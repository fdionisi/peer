use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use brain::embedder::Embedder;
use brain::language_model::{
    AssistantEvent, AssistantEventStream, LanguageModel, Prompt, PromptMessage, ToolCall,
};
use brain::models::message::{Content, Message, Role};
use brain::prompts::{PromptRegistry, render};
use brain::summarizer::{Summarizer, SummaryRequest};
use brain::tool::ToolDefinition;
use brain::topic_detector::{TopicDetector, TopicShift};

/// The vector dimension produced by `mistral-embed`. Used to validate live
/// responses and must match the HNSW index configuration in the SurrealDB
/// migration.
pub const MISTRAL_EMBED_DIMENSION: usize = 1024;

/// Topic detection is a classification, not a creative generation. It is sampled
/// at temperature zero so the same conversation yields the same decision: the
/// orchestrator's split behaviour stays stable and prompt-level evaluations are
/// reproducible rather than fighting run-to-run sampling noise.
const TOPIC_DETECTION_TEMPERATURE: f32 = 0.0;

use crate::MistralConfig;

pub struct MistralClient {
    client: Client,
    config: MistralConfig,
    prompts: Arc<dyn PromptRegistry>,
}

impl Clone for MistralClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            config: self.config.clone(),
            prompts: self.prompts.clone(),
        }
    }
}

impl MistralClient {
    pub fn new(config: MistralConfig, prompts: Arc<dyn PromptRegistry>) -> Result<Self> {
        let client = Client::builder()
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            client,
            config,
            prompts,
        })
    }

    fn chat_model(&self) -> &str {
        &self.config.chat_model
    }

    fn summarizer_model(&self) -> &str {
        self.config
            .summarizer_model
            .as_deref()
            .unwrap_or(&self.config.chat_model)
    }

    fn topic_detector_model(&self) -> &str {
        self.config
            .topic_detector_model
            .as_deref()
            .unwrap_or(&self.config.chat_model)
    }

    fn embed_model(&self) -> &str {
        self.config
            .embed_model
            .as_deref()
            .unwrap_or("mistral-embed")
    }

    fn format_messages(&self, messages: &[Message]) -> String {
        let blocks = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "system",
                };
                let text = Self::extract_text(&m.content);
                format!("<{role}>\n{text}\n</{role}>")
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        format!("<conversation>\n{blocks}\n</conversation>")
    }

    fn extract_text(content: &[Content]) -> String {
        content
            .iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(text.as_str()),
                Content::ToolCall { .. }
                | Content::ToolResult { .. }
                | Content::TemporalUpdate { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Renders a `PromptMessage` into the Mistral wire shape. `Content::ToolCall`
    /// becomes an assistant message carrying `tool_calls`; `Content::ToolResult`
    /// becomes a `tool` role message carrying the result and the correlating
    /// `tool_call_id`. A resolved pair replays correctly on the next turn.
    fn render_message(&self, msg: &PromptMessage) -> Vec<MistralMessage> {
        let mut out = Vec::new();
        let mut tool_calls: Vec<MistralToolCall> = Vec::new();
        let mut tool_results: Vec<MistralMessage> = Vec::new();

        for content in &msg.content {
            match content {
                Content::Text { text } => {
                    out.push(MistralMessage {
                        role: Self::mistral_role(msg.role).to_string(),
                        content: Some(text.clone()),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });
                }
                Content::ToolCall { id, name, input } => {
                    tool_calls.push(MistralToolCall {
                        id: id.clone(),
                        kind: "function".to_string(),
                        function: MistralFunctionCall {
                            name: name.clone(),
                            arguments: serde_json::to_string(&input)
                                .unwrap_or_else(|_| "null".to_string()),
                        },
                    });
                }
                Content::ToolResult {
                    id,
                    output,
                    is_error: _,
                } => {
                    tool_results.push(MistralMessage {
                        role: "tool".to_string(),
                        content: Some(output.clone()),
                        tool_calls: None,
                        tool_call_id: Some(id.clone()),
                        name: None,
                    });
                }
                Content::TemporalUpdate { timestamp } => {
                    out.push(MistralMessage {
                        role: "user".to_string(),
                        content: Some(format!("<temporal_update>{timestamp}</temporal_update>")),
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });
                }
            }
        }

        if !tool_calls.is_empty() {
            out.push(MistralMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(tool_calls),
                tool_call_id: None,
                name: None,
            });
        }
        out.extend(tool_results);
        out
    }

    fn mistral_role(role: Role) -> &'static str {
        match role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        }
    }

    /// Renders the `system` template, optionally injecting the prior-context
    /// summary and recalled summaries. This is where the generation system prompt
    /// is assembled — the orchestrator only supplies prompt data.
    fn render_system(&self, summary: Option<&str>, recalled: Option<&str>) -> Result<String> {
        match self.prompts.get("system") {
            Some(template) => {
                let mut vars: Vec<(&str, &str)> = Vec::new();
                if let Some(summary) = summary {
                    vars.push(("summary", summary));
                }
                if let Some(recalled) = recalled {
                    vars.push(("recalled", recalled));
                }
                render(template, &vars)
            }
            None => Ok(String::new()),
        }
    }

    fn render_recalled(recalled: &[String]) -> Option<String> {
        let summaries = recalled
            .iter()
            .map(|summary| summary.trim())
            .filter(|summary| !summary.is_empty())
            .collect::<Vec<_>>();

        if summaries.is_empty() {
            None
        } else {
            Some(summaries.join("\n\n---\n\n"))
        }
    }
}

#[derive(Debug, Serialize)]
struct MistralRequest {
    model: String,
    messages: Vec<MistralMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<MistralTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
}

#[derive(Debug, Serialize)]
struct MistralMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<MistralToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Debug, Serialize)]
struct MistralTool {
    #[serde(rename = "type")]
    kind: String,
    function: MistralFunction,
}

#[derive(Debug, Serialize)]
struct MistralFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct MistralToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: MistralFunctionCall,
}

#[derive(Debug, Serialize)]
struct MistralFunctionCall {
    name: String,
    arguments: String,
}

impl From<&ToolDefinition> for MistralTool {
    fn from(def: &ToolDefinition) -> Self {
        Self {
            kind: "function".to_string(),
            function: MistralFunction {
                name: def.name.clone(),
                description: def.description.clone(),
                parameters: def.input_schema.clone(),
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Deserialize)]
struct MistralResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: Delta,
}

#[derive(Debug, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    #[serde(rename = "type")]
    #[allow(dead_code)]
    kind: Option<String>,
    #[serde(default)]
    function: Option<FunctionCallDelta>,
}

#[derive(Debug, Deserialize)]
struct FunctionCallDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

// ── Streaming tool-call accumulator ──────────────────────────────────────────

/// A tool call being assembled from streaming fragments. Mistral sends each
/// tool call across multiple chunks: the first carries `id` and `name`, later
/// chunks carry pieces of the `arguments` JSON string. We accumulate by
/// `index` (the per-chunk identifier) and emit a complete `AssistantEvent::ToolCall`
/// when a boundary signal arrives — a new index, a content delta, or end of
/// stream.
#[derive(Debug, Default)]
struct BuildingToolCall {
    id: String,
    name: String,
    arguments: String,
}

fn building_to_event(call: BuildingToolCall) -> AssistantEvent {
    let input = serde_json::from_str(&call.arguments).unwrap_or(serde_json::Value::Null);
    AssistantEvent::ToolCall(ToolCall {
        id: call.id,
        name: call.name,
        input,
    })
}

/// Processes a single `Delta` against the accumulator and returns the events
/// to yield. A content delta flushes any accumulated tool calls (they are
/// complete before the model starts speaking); a tool-call delta with a new
/// `index` flushes the previous one (parallel calls arrive in order).
fn process_delta(
    delta: Delta,
    accumulated: &mut std::collections::HashMap<usize, BuildingToolCall>,
) -> Vec<AssistantEvent> {
    let mut events = Vec::new();

    if let Some(content) = delta.content {
        for (_, call) in accumulated.drain() {
            events.push(building_to_event(call));
        }
        events.push(AssistantEvent::Text(content));
    }

    if let Some(tool_calls) = delta.tool_calls {
        for tc in tool_calls {
            if !accumulated.contains_key(&tc.index) {
                for (_, call) in accumulated.drain() {
                    events.push(building_to_event(call));
                }
            }
            let entry = accumulated.entry(tc.index).or_default();
            if let Some(id) = tc.id {
                entry.id = id;
            }
            if let Some(name) = tc.function.as_ref().and_then(|f| f.name.clone()) {
                entry.name = name;
            }
            if let Some(args) = tc.function.as_ref().and_then(|f| f.arguments.clone()) {
                entry.arguments.push_str(&args);
            }
        }
    }

    events
}

// ── Embedding request / response ────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct EmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Debug, Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

// ── Topic-shift JSON response ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ShiftResponse {
    shifted: bool,
    #[serde(default)]
    new_topic: Option<String>,
    #[serde(default)]
    at_message_index: Option<usize>,
}

fn topic_shift_from_response(messages: &[Message], parsed: ShiftResponse) -> Option<TopicShift> {
    if !parsed.shifted {
        return None;
    }

    let Some(new_topic) = parsed.new_topic else {
        tracing::warn!("topic detector reported a shift without new_topic; ignoring");
        return None;
    };

    let Some(at_index) = parsed.at_message_index else {
        tracing::warn!("topic detector reported a shift without at_message_index; ignoring");
        return None;
    };

    let Some(at_message_id) = messages.get(at_index).map(|m| m.id) else {
        tracing::warn!(
            at_message_index = at_index,
            message_count = messages.len(),
            "topic detector returned an out-of-bounds at_message_index; ignoring"
        );
        return None;
    };

    Some(TopicShift {
        at_message_id,
        new_topic,
    })
}

/// Renders the outgoing wire payload as readable `[role]\n content` blocks for
/// tracing, so it is easy to confirm exactly what context Mistral receives.
fn format_mistral_messages(messages: &[MistralMessage]) -> String {
    messages
        .iter()
        .map(|m| {
            let content = m.content.as_deref().unwrap_or("");
            format!("[{}]\n{}", m.role, content)
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[async_trait]
impl LanguageModel for MistralClient {
    fn complete(&self, prompt: Prompt, tools: &[ToolDefinition]) -> AssistantEventStream {
        let client = self.client.clone();
        let config = self.config.clone();
        let model = self.chat_model().to_string();

        let recalled = Self::render_recalled(&prompt.recalled);
        let system = match self.render_system(prompt.summary.as_deref(), recalled.as_deref()) {
            Ok(system) => system,
            Err(e) => return Box::pin(async_stream::stream! { yield Err(e); }),
        };

        let system_chars = system.len();
        let mut messages = Vec::new();
        if !system.is_empty() {
            messages.push(MistralMessage {
                role: "system".to_string(),
                content: Some(system),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
        for msg in &prompt.messages {
            messages.extend(self.render_message(msg));
        }

        let wire_tools: Vec<MistralTool> = tools.iter().map(MistralTool::from).collect();
        let tool_choice = if wire_tools.is_empty() {
            None
        } else {
            Some("auto".to_string())
        };

        Box::pin(async_stream::stream! {
            tracing::info!(
                model = %model,
                system_chars,
                message_count = messages.len(),
                tool_count = wire_tools.len(),
                "mistral: chat completion request"
            );
            tracing::debug!(
                model = %model,
                "mistral: outgoing context:\n{}",
                format_mistral_messages(&messages)
            );

            let request = MistralRequest {
                model,
                messages,
                stream: Some(true),
                response_format: None,
                temperature: None,
                tools: if wire_tools.is_empty() { None } else { Some(wire_tools) },
                tool_choice,
            };

            let response = client
                .post(format!("{}/chat/completions", config.base_url))
                .bearer_auth(&config.api_key)
                .json(&request)
                .send()
                .await;

            let response = match response {
                Ok(r) => r,
                Err(e) => {
                    yield Err(anyhow::anyhow!("request failed: {e}"));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                yield Err(anyhow::anyhow!("API error {status}: {body}"));
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut accumulated: std::collections::HashMap<usize, BuildingToolCall> =
                std::collections::HashMap::new();

            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(anyhow::anyhow!("stream error: {e}"));
                        return;
                    }
                };

                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buffer.find("\n\n") {
                    let event = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    for line in event.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data == "[DONE]" {
                                for (_, call) in accumulated.drain() {
                                    yield Ok(building_to_event(call));
                                }
                                return;
                            }
                            match serde_json::from_str::<StreamChunk>(data) {
                                Ok(chunk) => {
                                    for choice in chunk.choices {
                                        for event in process_delta(choice.delta, &mut accumulated) {
                                            yield Ok(event);
                                        }
                                    }
                                }
                                Err(_) => continue,
                            }
                        }
                    }
                }
            }

            for (_, call) in accumulated.drain() {
                yield Ok(building_to_event(call));
            }
        })
    }

    async fn remaining_capacity(&self, _prompt: &Prompt) -> usize {
        4096
    }
}

#[async_trait]
impl Summarizer for MistralClient {
    async fn summarize(&self, messages: &[Message], request: &SummaryRequest) -> Result<String> {
        let conversation = self.format_messages(messages);

        let base_prompt = match self.prompts.get("summarize_base") {
            Some(template) => render(template, &[])?,
            None => String::new(),
        };

        // Map the request to its fine-tuning template. Prompt-template names
        // are an implementation detail of this adapter, not the orchestrator.
        let (template_name, vars): (&str, Vec<(&str, &str)>) = match request {
            SummaryRequest::TopicShift { new_topic } => (
                "summarize_topic_shift",
                vec![("new_topic", new_topic.as_str())],
            ),
            SummaryRequest::Compaction => ("summarize_compaction", Vec::new()),
        };
        let fine_tuning = match self.prompts.get(template_name) {
            Some(template) => render(template, &vars)?,
            None => String::new(),
        };

        let system_prompt = if fine_tuning.is_empty() {
            base_prompt
        } else {
            format!("{base_prompt}\n\n{fine_tuning}")
        };

        let request = MistralRequest {
            model: self.summarizer_model().to_string(),
            messages: vec![
                MistralMessage {
                    role: "system".to_string(),
                    content: Some(system_prompt),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                MistralMessage {
                    role: "user".to_string(),
                    content: Some(format!("Conversation to summarise:\n\n{conversation}")),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
            ],
            stream: None,
            response_format: None,
            temperature: None,
            tools: None,
            tool_choice: None,
        };

        tracing::info!(
            model = %request.model,
            message_count = messages.len(),
            "mistral: summarisation request"
        );
        tracing::debug!(
            model = %request.model,
            "mistral: summarisation input:\n{}",
            format_mistral_messages(&request.messages)
        );

        let response = self
            .client
            .post(format!("{}/chat/completions", self.config.base_url))
            .bearer_auth(&self.config.api_key)
            .json(&request)
            .send()
            .await
            .context("failed to send summarisation request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("API error {status}: {body}");
        }

        let body: MistralResponse = response
            .json()
            .await
            .context("failed to parse summarisation response")?;

        let summary = body
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .context("no choices in response")?;

        tracing::debug!("mistral: summary produced:\n{summary}");

        Ok(summary)
    }
}

#[async_trait]
impl Embedder for MistralClient {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request = EmbedRequest {
            model: self.embed_model().to_string(),
            input: vec![text.to_string()],
        };

        tracing::info!(
            model = %request.model,
            text_len = text.len(),
            "mistral: embedding request"
        );

        let response = self
            .client
            .post(format!("{}/embeddings", self.config.base_url))
            .bearer_auth(&self.config.api_key)
            .json(&request)
            .send()
            .await
            .context("failed to send embedding request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("API error {status}: {body}");
        }

        let body: EmbedResponse = response
            .json()
            .await
            .context("failed to parse embedding response")?;

        let embedding = body
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .context("no embedding data in response")?;

        tracing::debug!(dim = embedding.len(), "mistral: embedding received");

        Ok(embedding)
    }
}

#[async_trait]
impl TopicDetector for MistralClient {
    async fn detect_shift(&self, messages: &[Message]) -> Result<Option<TopicShift>> {
        if messages.len() < 2 {
            return Ok(None);
        }

        let system_prompt = match self.prompts.get("detect_topic_shift") {
            Some(template) => render(template, &[])?,
            None => String::new(),
        };

        let conversation = self.format_messages(messages);

        let request = MistralRequest {
            model: self.topic_detector_model().to_string(),
            messages: vec![
                MistralMessage {
                    role: "system".to_string(),
                    content: Some(system_prompt.to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                MistralMessage {
                    role: "user".to_string(),
                    content: Some(format!("Conversation:\n\n{conversation}")),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
            ],
            stream: None,
            response_format: Some(ResponseFormat {
                kind: "json_object".to_string(),
            }),
            temperature: Some(TOPIC_DETECTION_TEMPERATURE),
            tools: None,
            tool_choice: None,
        };

        tracing::info!(
            model = %request.model,
            message_count = messages.len(),
            "mistral: topic detection request"
        );
        tracing::debug!(
            model = %request.model,
            "mistral: topic detection input:\n{}",
            format_mistral_messages(&request.messages)
        );

        let response = self
            .client
            .post(format!("{}/chat/completions", self.config.base_url))
            .bearer_auth(&self.config.api_key)
            .json(&request)
            .send()
            .await
            .context("failed to send topic detection request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("API error {status}: {body}");
        }

        let body: MistralResponse = response
            .json()
            .await
            .context("failed to parse topic detection response")?;

        let content = body
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .context("no choices in response")?;

        let parsed: ShiftResponse =
            serde_json::from_str(&content).context("failed to parse shift response")?;

        tracing::info!(
            shifted = parsed.shifted,
            new_topic = parsed.new_topic.as_deref().unwrap_or("-"),
            at_message_index = parsed.at_message_index,
            "mistral: topic detection result"
        );

        Ok(topic_shift_from_response(messages, parsed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_client() -> MistralClient {
        use brain_prompts_embedded::EmbeddedPromptRegistry;

        MistralClient::new(
            MistralConfig::new("test-key", "mistral-small-latest"),
            Arc::new(EmbeddedPromptRegistry::new()),
        )
        .expect("client construction failed")
    }

    // ── Offline: system rendering ─────────────────────────────────────────────

    #[test]
    fn render_system_includes_recalled_block_when_recalled_text_is_supplied() {
        let client = test_client();
        let system = client
            .render_system(
                Some("Current branch summary"),
                Some("Earlier related summary"),
            )
            .unwrap();

        assert!(system.contains("<earlier_conversation_context>\nCurrent branch summary\n</earlier_conversation_context>"));
        assert!(system.contains("<recalled_conversation_context>"));
        assert!(system.contains("Earlier related summary"));
        assert!(system.contains("Use this as silent background knowledge."));
        assert!(system.contains("Do not announce that recall happened"));
        assert!(system.contains("</recalled_conversation_context>"));
    }

    #[test]
    fn render_system_omits_recalled_block_when_recalled_text_is_absent() {
        let client = test_client();
        let system = client
            .render_system(Some("Current branch summary"), None)
            .unwrap();

        assert!(system.contains("<earlier_conversation_context>"));
        assert!(!system.contains("<recalled_conversation_context>"));
        assert!(!system.contains("silent background knowledge"));
        assert!(!system.contains("Do not announce that recall happened"));
    }

    #[test]
    fn render_recalled_joins_non_empty_summaries_as_one_template_var() {
        let recalled = MistralClient::render_recalled(&[
            " First summary ".to_string(),
            "".to_string(),
            "Second summary".to_string(),
        ])
        .unwrap();

        assert_eq!(recalled, "First summary\n\n---\n\nSecond summary");
    }

    // ── Offline: topic-shift response handling ────────────────────────────────

    fn test_message(role: Role, text: &str) -> Message {
        Message {
            id: brain::models::message::MessageId::new(),
            role,
            content: vec![Content::Text {
                text: text.to_string(),
            }],
            timestamp: jiff::Timestamp::now(),
        }
    }

    #[test]
    fn format_messages_uses_xml_like_conversation_blocks() {
        let client = test_client();
        let formatted = client.format_messages(&[
            test_message(Role::User, "hello"),
            test_message(Role::Assistant, "hi"),
        ]);

        assert_eq!(
            formatted,
            "<conversation>\n<user>\nhello\n</user>\n\n<assistant>\nhi\n</assistant>\n</conversation>"
        );
    }

    #[test]
    fn topic_shift_response_maps_valid_index_to_message_id() {
        let first = test_message(Role::User, "old topic");
        let second = test_message(Role::Assistant, "old answer");
        let third = test_message(Role::User, "new topic");
        let messages = vec![first, second, third.clone()];

        let shift = topic_shift_from_response(
            &messages,
            ShiftResponse {
                shifted: true,
                new_topic: Some("new topic".to_string()),
                at_message_index: Some(2),
            },
        )
        .expect("expected valid shift");

        assert_eq!(shift.at_message_id, third.id);
        assert_eq!(shift.new_topic, "new topic");
    }

    #[test]
    fn topic_shift_response_ignores_out_of_bounds_index() {
        let messages = vec![
            test_message(Role::User, "old topic"),
            test_message(Role::Assistant, "old answer"),
            test_message(Role::User, "new topic"),
        ];

        let shift = topic_shift_from_response(
            &messages,
            ShiftResponse {
                shifted: true,
                new_topic: Some("new topic".to_string()),
                at_message_index: Some(4),
            },
        );

        assert!(shift.is_none());
    }

    #[test]
    fn topic_shift_response_ignores_missing_shift_fields() {
        let messages = vec![test_message(Role::User, "old topic")];

        assert!(
            topic_shift_from_response(
                &messages,
                ShiftResponse {
                    shifted: true,
                    new_topic: None,
                    at_message_index: Some(0),
                },
            )
            .is_none()
        );
        assert!(
            topic_shift_from_response(
                &messages,
                ShiftResponse {
                    shifted: true,
                    new_topic: Some("new topic".to_string()),
                    at_message_index: None,
                },
            )
            .is_none()
        );
    }

    // ── Offline: request shape ────────────────────────────────────────────────

    #[test]
    fn embed_request_serialises_correctly() {
        let model = "mistral-embed".to_string();
        let text = "hello world";
        let req = EmbedRequest {
            model: model.clone(),
            input: vec![text.to_string()],
        };
        let json = serde_json::to_value(&req).unwrap();

        assert_eq!(json["model"], model);
        assert_eq!(json["input"][0], text);
        assert_eq!(json["input"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn embed_request_uses_custom_model() {
        let req = EmbedRequest {
            model: "custom-embed-v1".to_string(),
            input: vec!["test".to_string()],
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "custom-embed-v1");
    }

    #[test]
    fn request_serialises_pinned_temperature_and_omits_when_absent() {
        let pinned = MistralRequest {
            model: "m".to_string(),
            messages: Vec::new(),
            stream: None,
            response_format: None,
            temperature: Some(TOPIC_DETECTION_TEMPERATURE),
            tools: None,
            tool_choice: None,
        };
        let json = serde_json::to_value(&pinned).unwrap();
        assert_eq!(json["temperature"], 0.0);

        let unset = MistralRequest {
            model: "m".to_string(),
            messages: Vec::new(),
            stream: None,
            response_format: None,
            temperature: None,
            tools: None,
            tool_choice: None,
        };
        let json = serde_json::to_value(&unset).unwrap();
        assert!(json.get("temperature").is_none());
    }

    // ── Offline: tool calling wire shape ──────────────────────────────────────

    #[test]
    fn tool_definition_serialises_into_mistral_function_shape() {
        let def = ToolDefinition {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }),
        };
        let wire = MistralTool::from(&def);
        let json = serde_json::to_value(&wire).unwrap();

        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "web_search");
        assert_eq!(json["function"]["description"], "Search the web");
        assert_eq!(json["function"]["parameters"]["type"], "object");
        assert_eq!(json["function"]["parameters"]["required"][0], "query");
    }

    #[test]
    fn request_serialises_tools_and_tool_choice_when_present() {
        let def = ToolDefinition {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let req = MistralRequest {
            model: "m".to_string(),
            messages: Vec::new(),
            stream: None,
            response_format: None,
            temperature: None,
            tools: Some(vec![MistralTool::from(&def)]),
            tool_choice: Some("auto".to_string()),
        };
        let json = serde_json::to_value(&req).unwrap();

        assert_eq!(json["tools"][0]["type"], "function");
        assert_eq!(json["tools"][0]["function"]["name"], "web_search");
        assert_eq!(json["tool_choice"], "auto");
    }

    #[test]
    fn request_omits_tools_and_tool_choice_when_absent() {
        let req = MistralRequest {
            model: "m".to_string(),
            messages: Vec::new(),
            stream: None,
            response_format: None,
            temperature: None,
            tools: None,
            tool_choice: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("tools").is_none());
        assert!(json.get("tool_choice").is_none());
    }

    #[test]
    fn render_message_maps_tool_call_to_assistant_with_tool_calls() {
        let client = test_client();
        let msg = PromptMessage {
            role: Role::Assistant,
            content: vec![Content::ToolCall {
                id: "call-1".to_string(),
                name: "web_search".to_string(),
                input: serde_json::json!({"query": "rust async"}),
            }],
        };

        let wire = client.render_message(&msg);
        assert_eq!(wire.len(), 1);
        let rendered = &wire[0];
        assert_eq!(rendered.role, "assistant");
        assert!(rendered.content.is_none());
        let calls = rendered.tool_calls.as_ref().expect("tool_calls present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call-1");
        assert_eq!(calls[0].kind, "function");
        assert_eq!(calls[0].function.name, "web_search");
        assert_eq!(calls[0].function.arguments, r#"{"query":"rust async"}"#);
    }

    #[test]
    fn render_message_maps_tool_result_to_tool_role_message() {
        let client = test_client();
        let msg = PromptMessage {
            role: Role::Assistant,
            content: vec![Content::ToolResult {
                id: "call-1".to_string(),
                output: "first hit".to_string(),
                is_error: false,
            }],
        };

        let wire = client.render_message(&msg);
        assert_eq!(wire.len(), 1);
        let rendered = &wire[0];
        assert_eq!(rendered.role, "tool");
        assert_eq!(rendered.content.as_deref(), Some("first hit"));
        assert_eq!(rendered.tool_call_id.as_deref(), Some("call-1"));
        assert!(rendered.tool_calls.is_none());
    }

    #[test]
    fn render_message_splits_text_and_tool_calls_into_separate_wire_messages() {
        let client = test_client();
        let msg = PromptMessage {
            role: Role::Assistant,
            content: vec![
                Content::Text {
                    text: "Let me look that up.".to_string(),
                },
                Content::ToolCall {
                    id: "call-1".to_string(),
                    name: "web_search".to_string(),
                    input: serde_json::json!({"query": "rust"}),
                },
            ],
        };

        let wire = client.render_message(&msg);
        assert_eq!(wire.len(), 2);
        assert_eq!(wire[0].role, "assistant");
        assert_eq!(wire[0].content.as_deref(), Some("Let me look that up."));
        assert!(wire[0].tool_calls.is_none());
        assert_eq!(wire[1].role, "assistant");
        assert!(wire[1].content.is_none());
        assert!(wire[1].tool_calls.is_some());
    }

    // ── Offline: streaming tool-call parsing ──────────────────────────────────

    #[test]
    fn process_delta_assembles_tool_call_from_fragmented_arguments() {
        let mut accumulated = std::collections::HashMap::new();

        // First chunk: id and name arrive.
        let events = process_delta(
            Delta {
                content: None,
                tool_calls: Some(vec![ToolCallDelta {
                    index: 0,
                    id: Some("call-1".to_string()),
                    kind: Some("function".to_string()),
                    function: Some(FunctionCallDelta {
                        name: Some("web_search".to_string()),
                        arguments: Some(r#"{"query":"#.to_string()),
                    }),
                }]),
            },
            &mut accumulated,
        );
        assert!(events.is_empty());

        // Second chunk: more arguments.
        let events = process_delta(
            Delta {
                content: None,
                tool_calls: Some(vec![ToolCallDelta {
                    index: 0,
                    id: None,
                    kind: None,
                    function: Some(FunctionCallDelta {
                        name: None,
                        arguments: Some(r#""rust async"}"#.to_string()),
                    }),
                }]),
            },
            &mut accumulated,
        );
        assert!(events.is_empty());

        // Third chunk: a content delta flushes the accumulated tool call.
        let events = process_delta(
            Delta {
                content: Some("Here you go.".to_string()),
                tool_calls: None,
            },
            &mut accumulated,
        );
        assert_eq!(events.len(), 2);

        match &events[0] {
            AssistantEvent::ToolCall(call) => {
                assert_eq!(call.id, "call-1");
                assert_eq!(call.name, "web_search");
                assert_eq!(call.input, serde_json::json!({"query": "rust async"}));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
        assert_eq!(events[1], AssistantEvent::Text("Here you go.".to_string()));
    }

    #[test]
    fn process_delta_flushes_previous_call_when_new_index_arrives() {
        let mut accumulated = std::collections::HashMap::new();

        // First call: id, name, complete arguments.
        process_delta(
            Delta {
                content: None,
                tool_calls: Some(vec![ToolCallDelta {
                    index: 0,
                    id: Some("call-1".to_string()),
                    kind: Some("function".to_string()),
                    function: Some(FunctionCallDelta {
                        name: Some("web_search".to_string()),
                        arguments: Some(r#"{"query":"a"}"#.to_string()),
                    }),
                }]),
            },
            &mut accumulated,
        );

        // Second call: new index 1 flushes the first.
        let events = process_delta(
            Delta {
                content: None,
                tool_calls: Some(vec![ToolCallDelta {
                    index: 1,
                    id: Some("call-2".to_string()),
                    kind: Some("function".to_string()),
                    function: Some(FunctionCallDelta {
                        name: Some("web_search".to_string()),
                        arguments: Some(r#"{"query":"b"}"#.to_string()),
                    }),
                }]),
            },
            &mut accumulated,
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            AssistantEvent::ToolCall(call) => {
                assert_eq!(call.id, "call-1");
                assert_eq!(call.input, serde_json::json!({"query": "a"}));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn process_delta_handles_malformed_arguments_as_null() {
        let mut accumulated = std::collections::HashMap::new();

        process_delta(
            Delta {
                content: None,
                tool_calls: Some(vec![ToolCallDelta {
                    index: 0,
                    id: Some("call-1".to_string()),
                    kind: Some("function".to_string()),
                    function: Some(FunctionCallDelta {
                        name: Some("web_search".to_string()),
                        arguments: Some("not-json".to_string()),
                    }),
                }]),
            },
            &mut accumulated,
        );

        let events = process_delta(
            Delta {
                content: Some("done".to_string()),
                tool_calls: None,
            },
            &mut accumulated,
        );
        match &events[0] {
            AssistantEvent::ToolCall(call) => {
                assert_eq!(call.id, "call-1");
                assert_eq!(call.input, serde_json::Value::Null);
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn stream_chunk_with_tool_calls_deserialises() {
        // Captured Mistral streaming payload: a tool-call delta with id, name,
        // and a fragment of arguments.
        let payload = r#"{
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "abc123",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": "{\"query\":\""
                            }
                        }]
                    }
                }]
            }"#;

        let chunk: StreamChunk = serde_json::from_str(payload).unwrap();
        let delta = &chunk.choices[0].delta;
        let calls = delta.tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].index, 0);
        assert_eq!(calls[0].id.as_deref(), Some("abc123"));
        assert_eq!(
            calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("web_search")
        );
        assert_eq!(
            calls[0].function.as_ref().unwrap().arguments.as_deref(),
            Some(r#"{"query":""#)
        );
    }

    // ── Live (requires MISTRAL_API_KEY) ───────────────────────────────────────────

    /// Calls the real Mistral `/embeddings` endpoint and asserts the returned
    /// vector has the expected dimension. Run with:
    ///   MISTRAL_API_KEY=<key> cargo test -p brain_mistralai embed_live -- --ignored
    #[tokio::test]
    #[ignore]
    async fn embed_live_dimension() {
        use brain_prompts_embedded::EmbeddedPromptRegistry;
        use std::sync::Arc;

        let api_key = std::env::var("MISTRAL_API_KEY")
            .expect("MISTRAL_API_KEY must be set to run live tests");

        let config = MistralConfig::new(api_key, "mistral-small-latest");
        let prompts = Arc::new(EmbeddedPromptRegistry::new());
        let client = MistralClient::new(config, prompts).expect("client construction failed");

        let embedding = client
            .embed("The quick brown fox jumps over the lazy dog")
            .await
            .expect("embedding request failed");

        assert_eq!(
            embedding.len(),
            MISTRAL_EMBED_DIMENSION,
            "expected {MISTRAL_EMBED_DIMENSION}-dimensional vector from mistral-embed, got {}",
            embedding.len()
        );
    }
}
