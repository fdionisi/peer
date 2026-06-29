pub mod models;

pub mod conversation_store;

pub mod embedder;
pub mod language_model;
pub mod memory_index;

pub mod prompts;
pub mod summarizer;
pub mod topic_detector;

pub mod tool;

#[cfg(any(test, feature = "test-harness"))]
pub mod testing;

#[cfg(test)]
mod brain_tests;

use std::{collections::HashSet, pin::Pin, sync::Arc, task::Poll};

use anyhow::Result;
use futures::{Stream, StreamExt};
use jiff::Timestamp;
use tokio::sync::mpsc::Receiver;

use crate::{
    conversation_store::ConversationStore,
    embedder::Embedder,
    language_model::{AssistantEvent, ContentStream, LanguageModel, Prompt, PromptMessage},
    memory_index::{MemoryIndex, RelatedConversation},
    models::{
        conversation::{Conversation, ConversationId, ConversationItem},
        message::{Content, Message, MessageId, Role},
        summary::Summary,
    },
    summarizer::{Summarizer, SummaryRequest},
    tool::{Policy, ToolDefinition, ToolOutput, ToolRegistry},
    topic_detector::TopicDetector,
};

pub type InputStream = Pin<Box<dyn Stream<Item = Content> + Send>>;

/// Number of trailing messages carried forward (as references) on a
/// compaction split, to preserve continuity for generation. Drift splits
/// carry no overlap: the trailing turns belong to the topic being left
/// behind and are already captured in the summary.
const OVERLAP_MESSAGES: usize = 2;

/// Number of candidates requested from the memory index when discovering a
/// recall target. The single best candidate is kept; widening this is a
/// follow-up once the threshold is trusted.
const RECALL_K: usize = 1;

/// Built-in tool name injected by the orchestrator into every `Confirm`
/// thread. The model calls this to close the thread with either `execute` or
/// `cancel`, avoiding the need to hard-wire a CLI command for resolution.
const RESOLVE_TOOL_NAME: &str = "__resolve";

pub struct Brain {
    conversations: Arc<dyn ConversationStore>,
    summarizer: Arc<dyn Summarizer>,
    topic_detector: Arc<dyn TopicDetector>,
    llm: Arc<dyn LanguageModel>,
    embedder: Arc<dyn Embedder>,
    memory_index: Arc<dyn MemoryIndex>,
    tools: Arc<dyn ToolRegistry>,
    compaction_threshold: usize,
    recall_threshold: f32,
}

#[derive(Debug, Clone)]
struct Context {
    summary: Option<Summary>,
    recalled: Vec<String>,
    messages: Vec<Message>,
}

impl Brain {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        conversations: Arc<dyn ConversationStore>,
        summarizer: Arc<dyn Summarizer>,
        topic_detector: Arc<dyn TopicDetector>,
        llm: Arc<dyn LanguageModel>,
        embedder: Arc<dyn Embedder>,
        memory_index: Arc<dyn MemoryIndex>,
        tools: Arc<dyn ToolRegistry>,
        compaction_threshold: usize,
        recall_threshold: f32,
    ) -> Self {
        Self {
            conversations,
            summarizer,
            topic_detector,
            llm,
            embedder,
            memory_index,
            tools,
            compaction_threshold,
            recall_threshold,
        }
    }

    pub async fn current_conversation(&self) -> Result<ConversationId> {
        self.conversations.current().await
    }

    pub async fn get_conversation(&self, id: ConversationId) -> Result<Option<Conversation>> {
        self.conversations.get(id).await
    }

    pub async fn list_messages(&self, conversation_id: ConversationId) -> Result<Vec<Message>> {
        self.conversations.list_messages(conversation_id).await
    }

    pub async fn say(&self, message: InputStream) -> Result<ContentStream> {
        let content: Vec<Content> = message.collect().await;
        let conversation_id = self.conversations.current().await?;
        tracing::info!(conversation_id = %conversation_id, "say called");

        let user_message = Message {
            id: MessageId::new(),
            role: Role::User,
            content,
            timestamp: Timestamp::now(),
        };

        let conversation_id = self.maybe_split(conversation_id, &user_message).await?;

        if self.should_inject_temporal_update(conversation_id).await? {
            let now = Timestamp::now();
            let temporal_message = Message {
                id: MessageId::new(),
                role: Role::User,
                content: vec![Content::TemporalUpdate {
                    timestamp: now.to_string(),
                }],
                timestamp: now,
            };
            self.conversations
                .append_message(conversation_id, temporal_message)
                .await?;
            tracing::debug!(%conversation_id, "injected temporal update");
        }

        self.conversations
            .append_message(conversation_id, user_message)
            .await?;

        let tool_defs = self.tools.visible_definitions();

        let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<String>>(32);

        let conversations = self.conversations.clone();
        let tools = self.tools.clone();
        let llm = self.llm.clone();

        let context = self.assemble_context(&conversation_id).await?;
        let initial_prompt = self.build_prompt(&context);

        tracing::info!(
            %conversation_id,
            has_summary = initial_prompt.summary.is_some(),
            recalled = initial_prompt.recalled.len(),
            message_count = initial_prompt.messages.len(),
            "assembled prompt for generation"
        );
        tracing::debug!(
            %conversation_id,
            "prompt sent to language model:\n{}",
            render_prompt(&initial_prompt)
        );

        tokio::spawn(async move {
            Self::run_tool_loop(
                conversations,
                llm,
                tools,
                conversation_id,
                initial_prompt,
                tool_defs,
                tx,
            )
            .await;
        });

        Ok(Box::pin(ReceiverStream(rx)))
    }

    /// Drives the tool loop for one `say` turn.
    ///
    /// Calls `complete`, forwards `Text` events to `tx`, and handles
    /// `ToolCall` events:
    ///
    /// - `Auto` policy: execute immediately, append the adjacent
    ///   `ToolCall`/`ToolResult` pair to the parent conversation, and loop.
    /// - `Confirm` policy: open a resolution thread (child conversation),
    ///   stream the model's confirmation question to the user, and return —
    ///   the next `say` lands in the thread. Resolution writes the adjacent
    ///   pair into the **parent** once decided.
    ///
    /// The built-in `resolve` sentinel (`{"action": "execute"|"cancel"}`) is
    /// injected alongside the registered definitions so that every `Confirm`
    /// thread can close itself without CLI wiring.
    async fn run_tool_loop(
        conversations: Arc<dyn ConversationStore>,
        llm: Arc<dyn LanguageModel>,
        tools: Arc<dyn ToolRegistry>,
        conversation_id: ConversationId,
        mut prompt: Prompt,
        tool_defs: Vec<ToolDefinition>,
        tx: tokio::sync::mpsc::Sender<anyhow::Result<String>>,
    ) {
        let effective_tools = Self::tools_with_resolve_sentinel(&tool_defs);

        loop {
            let stream = llm.complete(prompt.clone(), &effective_tools);

            let mut text_accumulated = String::new();
            let mut tool_calls: Vec<crate::language_model::ToolCall> = Vec::new();
            let mut stream_error = false;

            {
                let mut pinned = stream;
                while let Some(event) = pinned.next().await {
                    match event {
                        Ok(AssistantEvent::Text(text)) => {
                            text_accumulated.push_str(&text);
                            if tx.send(Ok(text)).await.is_err() {
                                tracing::debug!(
                                    %conversation_id,
                                    "response receiver dropped"
                                );
                                return;
                            }
                        }
                        Ok(AssistantEvent::ToolCall(call)) => {
                            tracing::debug!(
                                %conversation_id,
                                tool_call_id = %call.id,
                                tool_name = %call.name,
                                "tool call received"
                            );
                            tool_calls.push(call);
                        }
                        Err(e) => {
                            tracing::warn!(
                                %conversation_id,
                                error = %e,
                                "language model stream error"
                            );
                            let _ = tx.send(Err(e)).await;
                            stream_error = true;
                            break;
                        }
                    }
                }
            }

            if stream_error {
                return;
            }

            if !text_accumulated.is_empty() {
                tracing::info!(
                    %conversation_id,
                    response_chars = text_accumulated.len(),
                    "persisting assistant text response"
                );
                let assistant_message = Message {
                    id: MessageId::new(),
                    role: Role::Assistant,
                    content: vec![Content::Text {
                        text: text_accumulated,
                    }],
                    timestamp: Timestamp::now(),
                };
                if let Err(e) = conversations
                    .append_message(conversation_id, assistant_message)
                    .await
                {
                    tracing::error!(
                        %conversation_id,
                        error = %e,
                        "failed to persist assistant response"
                    );
                    return;
                }
            }

            if tool_calls.is_empty() {
                return;
            }

            let mut any_confirm = false;
            for call in tool_calls {
                if call.name == RESOLVE_TOOL_NAME {
                    let action = call
                        .input
                        .get("action")
                        .and_then(|v| v.as_str())
                        .unwrap_or("cancel");

                    tracing::info!(
                        %conversation_id,
                        action,
                        "resolve sentinel received"
                    );

                    let pending =
                        Self::pending_call_from_thread(&*conversations, conversation_id).await;

                    let payload = match (action, pending) {
                        ("execute", Some(pending_call)) => {
                            let result = tools
                                .execute(&pending_call.name, pending_call.input.clone())
                                .await
                                .unwrap_or_else(|e| ToolOutput {
                                    text: e.to_string(),
                                    is_error: true,
                                });
                            Self::tool_pair_messages(&pending_call, &result)
                        }
                        (_, Some(pending_call)) => {
                            let cancelled = ToolOutput {
                                text: "Tool call cancelled by user.".to_string(),
                                is_error: false,
                            };
                            Self::tool_pair_messages(&pending_call, &cancelled)
                        }
                        (_, None) => Vec::new(),
                    };

                    if let Err(e) = conversations.resolve_branch(conversation_id, payload).await {
                        tracing::error!(
                            %conversation_id,
                            error = %e,
                            "resolve_branch failed"
                        );
                    }

                    return;
                }

                let policy = tools.policy(&call.name).unwrap_or(Policy::Auto);

                match policy {
                    Policy::Auto => {
                        let result = tools
                            .execute(&call.name, call.input.clone())
                            .await
                            .unwrap_or_else(|e| ToolOutput {
                                text: e.to_string(),
                                is_error: true,
                            });

                        tracing::info!(
                            %conversation_id,
                            tool_name = %call.name,
                            is_error = result.is_error,
                            "auto tool executed"
                        );

                        let pair = Self::tool_pair_messages(&call, &result);
                        if let Err(e) = conversations.append_messages(conversation_id, pair).await {
                            tracing::error!(
                                %conversation_id,
                                error = %e,
                                "failed to persist tool call/result pair"
                            );
                            return;
                        }

                        let messages = match conversations.list_messages(conversation_id).await {
                            Ok(m) => m,
                            Err(e) => {
                                tracing::error!(
                                    %conversation_id,
                                    error = %e,
                                    "failed to list messages for prompt rebuild"
                                );
                                return;
                            }
                        };
                        prompt.messages = messages
                            .iter()
                            .map(|m| PromptMessage {
                                role: m.role,
                                content: m.content.clone(),
                            })
                            .collect();
                    }
                    Policy::Confirm => {
                        let thread = match conversations.create(Some(conversation_id)).await {
                            Ok(t) => t,
                            Err(e) => {
                                tracing::error!(
                                    %conversation_id,
                                    error = %e,
                                    "failed to create confirmation thread"
                                );
                                return;
                            }
                        };

                        let pending_msg = Message {
                            id: MessageId::new(),
                            role: Role::Assistant,
                            content: vec![Content::ToolCall {
                                id: call.id.clone(),
                                name: call.name.clone(),
                                input: call.input.clone(),
                            }],
                            timestamp: Timestamp::now(),
                        };
                        if let Err(e) = conversations.append_message(thread.id, pending_msg).await {
                            tracing::error!(
                                %conversation_id,
                                error = %e,
                                "failed to seed confirmation thread"
                            );
                            return;
                        }

                        if let Err(e) = conversations.set_current(thread.id).await {
                            tracing::error!(
                                %conversation_id,
                                error = %e,
                                "failed to set current to confirmation thread"
                            );
                            return;
                        }

                        tracing::info!(
                            parent_id = %conversation_id,
                            thread_id = %thread.id,
                            tool_name = %call.name,
                            "opened confirmation thread"
                        );

                        any_confirm = true;

                        let question_prompt = Self::build_confirm_prompt(&call, &effective_tools);
                        let q_stream = llm.complete(question_prompt, &effective_tools);
                        let mut q_text = String::new();
                        let mut pinned = q_stream;
                        while let Some(event) = pinned.next().await {
                            match event {
                                Ok(AssistantEvent::Text(text)) => {
                                    q_text.push_str(&text);
                                    if tx.send(Ok(text)).await.is_err() {
                                        return;
                                    }
                                }
                                Ok(AssistantEvent::ToolCall(_)) => {}
                                Err(e) => {
                                    let _ = tx.send(Err(e)).await;
                                    return;
                                }
                            }
                        }
                        if !q_text.is_empty() {
                            let q_msg = Message {
                                id: MessageId::new(),
                                role: Role::Assistant,
                                content: vec![Content::Text { text: q_text }],
                                timestamp: Timestamp::now(),
                            };
                            if let Err(e) = conversations.append_message(thread.id, q_msg).await {
                                tracing::error!(
                                    thread_id = %thread.id,
                                    error = %e,
                                    "failed to persist confirmation question"
                                );
                            }
                        }
                    }
                }
            }

            if any_confirm {
                return;
            }
        }
    }

    /// Builds the adjacent `ToolCall` + `ToolResult` message pair without
    /// touching the store. Callers persist it atomically via `append_messages`
    /// or `resolve_branch`.
    fn tool_pair_messages(
        call: &crate::language_model::ToolCall,
        result: &ToolOutput,
    ) -> Vec<Message> {
        vec![
            Message {
                id: MessageId::new(),
                role: Role::Assistant,
                content: vec![Content::ToolCall {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    input: call.input.clone(),
                }],
                timestamp: Timestamp::now(),
            },
            Message {
                id: MessageId::new(),
                role: Role::User,
                content: vec![Content::ToolResult {
                    id: call.id.clone(),
                    output: result.text.clone(),
                    is_error: result.is_error,
                }],
                timestamp: Timestamp::now(),
            },
        ]
    }

    /// Reads the first `Content::ToolCall` from the confirmation thread so
    /// the resolving turn knows what to execute (or cancel).
    async fn pending_call_from_thread(
        conversations: &dyn ConversationStore,
        thread_id: ConversationId,
    ) -> Option<crate::language_model::ToolCall> {
        let messages = conversations.list_messages(thread_id).await.ok()?;
        for msg in &messages {
            for content in &msg.content {
                if let Content::ToolCall { id, name, input } = content {
                    return Some(crate::language_model::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    });
                }
            }
        }
        None
    }

    /// Returns the registered tool definitions augmented with the built-in
    /// `resolve` sentinel used by `Confirm` threads.
    fn tools_with_resolve_sentinel(defs: &[ToolDefinition]) -> Vec<ToolDefinition> {
        let mut out: Vec<ToolDefinition> = defs.to_vec();
        out.push(ToolDefinition {
            name: RESOLVE_TOOL_NAME.to_string(),
            description: "Close a confirmation thread. Use action=\"execute\" to proceed with the \
                          tool call or action=\"cancel\" to abort it."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["execute", "cancel"]
                    }
                },
                "required": ["action"]
            }),
        });
        out
    }

    /// Builds a minimal prompt for the model's opening question in a
    /// confirmation thread. The model is asked to state what it intends to do
    /// and seek confirmation.
    fn build_confirm_prompt(
        call: &crate::language_model::ToolCall,
        _tools: &[ToolDefinition],
    ) -> Prompt {
        Prompt {
            summary: None,
            recalled: Vec::new(),
            messages: vec![PromptMessage {
                role: Role::User,
                content: vec![Content::Text {
                    text: format!(
                        "You are about to call the `{}` tool with the following input: {}. \
                         Describe what you are about to do and ask the user to confirm \
                         (reply \"yes\" to proceed, \"no\" to cancel).",
                        call.name,
                        serde_json::to_string_pretty(&call.input)
                            .unwrap_or_else(|_| call.input.to_string())
                    ),
                }],
            }],
        }
    }

    /// Whether a `TemporalUpdate` should be injected before the next user message.
    ///
    /// Injects on the first turn (no owned messages) or when the conversation
    /// has been idle for more than four hours.
    async fn should_inject_temporal_update(&self, conversation_id: ConversationId) -> Result<bool> {
        const IDLE_THRESHOLD: i64 = 4 * 60 * 60; // 4 hours in seconds

        let owned = self.owned_messages(&conversation_id).await?;

        let last_non_temporal = owned.iter().rev().find(|m| {
            !m.content
                .iter()
                .all(|c| matches!(c, Content::TemporalUpdate { .. }))
        });

        match last_non_temporal {
            None => Ok(true), // first turn
            Some(last) => {
                let elapsed = Timestamp::now().duration_since(last.timestamp).as_secs() as i64;
                Ok(elapsed >= IDLE_THRESHOLD)
            }
        }
    }

    /// Returns the conversation the incoming message should be recorded in.
    ///
    /// If the message represents a topic shift, or recording it would overflow
    /// the context window, the current conversation is closed (summarised into
    /// a fresh child) and the child's id is returned. Otherwise the current
    /// conversation id is returned unchanged.
    async fn maybe_split(
        &self,
        conversation_id: ConversationId,
        incoming: &Message,
    ) -> Result<ConversationId> {
        let owned = self.owned_messages(&conversation_id).await?;

        if let Some((split, new_topic)) = self.detect_drift(&owned, incoming).await? {
            tracing::info!(%conversation_id, new_topic = %new_topic, "topic shift detected");
            return self
                .split_conversation(
                    conversation_id,
                    &owned,
                    split,
                    SummaryRequest::TopicShift { new_topic },
                )
                .await;
        }

        if self.would_overflow(&conversation_id, incoming).await? {
            tracing::info!(%conversation_id, threshold = self.compaction_threshold, "compacting");
            return self
                .split_conversation(
                    conversation_id,
                    &owned,
                    owned.len(),
                    SummaryRequest::Compaction,
                )
                .await;
        }

        Ok(conversation_id)
    }

    /// Detects whether `incoming` moves the conversation onto a new topic.
    ///
    /// Returns the boundary index within `owned` (everything before it is the
    /// prior context to summarise) together with the detected topic, or `None`
    /// when the conversation stays on subject.
    async fn detect_drift(
        &self,
        owned: &[Message],
        incoming: &Message,
    ) -> Result<Option<(usize, String)>> {
        let mut probe = owned.to_vec();
        probe.push(incoming.clone());

        let Some(shift) = self.topic_detector.detect_shift(&probe).await? else {
            tracing::debug!(
                owned = owned.len(),
                probe = probe.len(),
                "drift detection: no shift"
            );
            return Ok(None);
        };

        let split = probe
            .iter()
            .position(|m| m.id == shift.at_message_id)
            .unwrap_or(owned.len())
            .min(owned.len());

        if split == 0 {
            tracing::debug!(
                new_topic = %shift.new_topic,
                "drift detection: shift at start, nothing to summarise; staying"
            );
            return Ok(None);
        }

        tracing::debug!(
            owned = owned.len(),
            split,
            new_topic = %shift.new_topic,
            "drift detection: shift detected"
        );
        Ok(Some((split, shift.new_topic)))
    }

    /// Whether recording `incoming` in `conversation_id` would push the prompt
    /// past the compaction threshold.
    async fn would_overflow(
        &self,
        conversation_id: &ConversationId,
        incoming: &Message,
    ) -> Result<bool> {
        let context = self.assemble_context(conversation_id).await?;
        let mut prompt = self.build_prompt(&context);
        prompt.messages.push(PromptMessage {
            role: incoming.role,
            content: incoming.content.clone(),
        });
        let remaining = self.llm.remaining_capacity(&prompt).await;
        let overflow = remaining < self.compaction_threshold;
        tracing::debug!(
            %conversation_id,
            remaining,
            threshold = self.compaction_threshold,
            overflow,
            "capacity check"
        );
        Ok(overflow)
    }

    /// Closes `parent` by summarising the prior context into a fresh child
    /// conversation and switching to it.
    ///
    /// `owned[..split]` is summarised into the child's summary (which the child
    /// surfaces through its system prompt). On *compaction* a small trailing
    /// overlap is carried forward as *referenced* messages: the topic is
    /// unchanged, so the last verbatim turns keep the active thread coherent
    /// across the summary boundary. They enrich generation but are excluded
    /// from the child's future drift detection and summarisation. On a *topic
    /// shift* no overlap is carried: the trailing turns belong to the topic
    /// being left behind, are already captured in the summary, and would only
    /// pollute the new topic's context. The drift child starts on the incoming
    /// message alone.
    ///
    /// On a topic shift, recall discovery (embed the new topic, search the
    /// memory index) runs concurrently with summarisation — both are known
    /// once drift is detected and neither depends on the other, so the drift
    /// turn pays the latency of the slower of the two, not their sum. The
    /// summary embedding is written to the index in a detached task after the
    /// child is created, so it never blocks the response.
    async fn split_conversation(
        &self,
        parent: ConversationId,
        owned: &[Message],
        split: usize,
        request: SummaryRequest,
    ) -> Result<ConversationId> {
        let (summary_result, recall_candidate) = match &request {
            SummaryRequest::TopicShift { new_topic } => {
                let summarizer_fut = self.summarizer.summarize(&owned[..split], &request);
                let recall_fut = self.discover_recall(new_topic, parent);
                let (summary, recall) = futures::join!(summarizer_fut, recall_fut);
                (summary, recall)
            }
            SummaryRequest::Compaction => {
                let summary = self.summarizer.summarize(&owned[..split], &request).await?;
                (Ok(summary), Ok(None))
            }
        };

        let content = summary_result?;
        let recall_candidate = match recall_candidate {
            Ok(candidate) => candidate,
            Err(e) => {
                tracing::warn!(%parent, error = %e, "recall discovery failed; continuing without recall");
                None
            }
        };

        let referenced: Vec<MessageId> = match &request {
            SummaryRequest::Compaction => {
                let overlap_start = split.saturating_sub(OVERLAP_MESSAGES);
                owned[overlap_start..]
                    .iter()
                    .skip_while(|m| !matches!(m.role, Role::User))
                    .map(|m| m.id)
                    .collect()
            }
            SummaryRequest::TopicShift { .. } => Vec::new(),
        };

        let new_conversation = self.conversations.create(Some(parent)).await?;
        tracing::info!(
            %parent,
            new = %new_conversation.id,
            reason = request.label(),
            referenced = referenced.len(),
            "split conversation"
        );

        self.conversations
            .update_summary(
                new_conversation.id,
                Summary {
                    content: content.clone(),
                    created_at: Timestamp::now(),
                },
            )
            .await?;

        self.conversations
            .reference_messages(new_conversation.id, referenced)
            .await?;

        self.conversations.set_current(new_conversation.id).await?;

        let summary_text = content;
        let embedder = self.embedder.clone();
        let memory_index = self.memory_index.clone();
        let child_id = new_conversation.id;
        tokio::spawn(async move {
            match embedder.embed(&summary_text).await {
                Ok(embedding) => {
                    if let Err(e) = memory_index.index_summary(child_id, embedding).await {
                        tracing::warn!(
                            %child_id,
                            error = %e,
                            "failed to index summary embedding"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(%child_id, error = %e, "failed to embed summary");
                }
            }
        });

        if let Some(candidate) = recall_candidate {
            tracing::info!(
                %child_id,
                recalled = %candidate.id,
                score = candidate.score,
                threshold = self.recall_threshold,
                "recording recall edge"
            );
            if let Err(e) = self
                .memory_index
                .record_recall(child_id, candidate.id, candidate.score)
                .await
            {
                tracing::warn!(error = %e, "failed to record recall edge");
            }
        }

        Ok(new_conversation.id)
    }

    /// Embeds `new_topic`, searches the memory index for related
    /// conversations (excluding `parent`), and returns the single best
    /// candidate whose score meets `recall_threshold`, or `None`.
    ///
    /// Runs concurrently with summarisation in `split_conversation`; both
    /// are known once drift is detected and neither depends on the other.
    async fn discover_recall(
        &self,
        new_topic: &str,
        parent: ConversationId,
    ) -> Result<Option<RelatedConversation>> {
        let embedding = self.embedder.embed(new_topic).await?;
        let results = self
            .memory_index
            .search(embedding, RECALL_K, Some(parent))
            .await?;
        let best = results.into_iter().next();
        let accepted = best.filter(|r| r.score >= self.recall_threshold);
        tracing::debug!(
            %parent,
            new_topic = %new_topic,
            threshold = self.recall_threshold,
            accepted = ?accepted.as_ref().map(|r| (r.id, r.score)),
            "recall discovery"
        );
        Ok(accepted)
    }

    /// The messages owned by a conversation, excluding referenced overlap.
    async fn owned_messages(&self, conversation_id: &ConversationId) -> Result<Vec<Message>> {
        let conv = self
            .conversations
            .get(*conversation_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("conversation not found"))?;

        let owned: HashSet<MessageId> = conv
            .items
            .iter()
            .filter_map(|item| match item {
                ConversationItem::Message(id) => Some(*id),
                ConversationItem::ReferencedMessage(_) => None,
            })
            .collect();

        let all = self.conversations.list_messages(*conversation_id).await?;
        Ok(all.into_iter().filter(|m| owned.contains(&m.id)).collect())
    }

    /// Assembles the generation context for a conversation: its summary (prior
    /// context), the summaries of any conversations linked by recall edges,
    /// and every message, owned and referenced, in order.
    async fn assemble_context(&self, conversation_id: &ConversationId) -> Result<Context> {
        let conv = self
            .conversations
            .get(*conversation_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("conversation not found"))?;
        let messages = self.conversations.list_messages(*conversation_id).await?;

        let recalled_ids = self.memory_index.recalled(*conversation_id).await?;
        let mut recalled = Vec::with_capacity(recalled_ids.len());
        for id in recalled_ids {
            match self.conversations.get(id).await? {
                Some(source) => match source.summary {
                    Some(summary) => recalled.push(summary.content),
                    None => {
                        tracing::debug!(
                            %conversation_id,
                            recalled_id = %id,
                            "recalled conversation has no summary; skipping"
                        );
                    }
                },
                None => {
                    tracing::debug!(
                        %conversation_id,
                        recalled_id = %id,
                        "recalled conversation not found; skipping"
                    );
                }
            }
        }

        let referenced = conv
            .items
            .iter()
            .filter(|i| matches!(i, ConversationItem::ReferencedMessage(_)))
            .count();
        tracing::debug!(
            %conversation_id,
            has_summary = conv.summary.is_some(),
            recalled = recalled.len(),
            owned = messages.len().saturating_sub(referenced),
            referenced,
            total = messages.len(),
            "assembled context"
        );

        Ok(Context {
            summary: conv.summary,
            recalled,
            messages,
        })
    }

    /// Builds the data the language model needs: the prior-context summary (if
    /// any), the recalled summaries from related conversations, and the
    /// messages. Rendering this into an actual system prompt is the language
    /// model implementation's responsibility, so prompt templates live in one
    /// place — the model adapter — rather than being split here.
    fn build_prompt(&self, context: &Context) -> Prompt {
        let messages: Vec<PromptMessage> = context
            .messages
            .iter()
            .map(|m| PromptMessage {
                role: m.role,
                content: m.content.clone(),
            })
            .collect();

        Prompt {
            summary: context.summary.as_ref().map(|s| s.content.clone()),
            recalled: context.recalled.clone(),
            messages,
        }
    }
}

/// Renders a prompt as readable `[role]\n content` blocks for tracing, so the
/// data handed to the language model can be inspected in the logs. The final
/// system prompt is rendered by the model adapter, which logs it separately.
fn render_prompt(prompt: &Prompt) -> String {
    let mut out = String::new();
    if let Some(summary) = &prompt.summary {
        out.push_str("[summary]\n");
        out.push_str(summary);
        out.push_str("\n\n");
    }
    if !prompt.recalled.is_empty() {
        out.push_str("[recalled]\n");
        for entry in &prompt.recalled {
            out.push_str(entry);
            out.push_str("\n\n");
        }
    }
    for message in &prompt.messages {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        };
        let text: String = message
            .content
            .iter()
            .map(|c| match c {
                Content::Text { text } => text.as_str(),
                Content::ToolCall { .. }
                | Content::ToolResult { .. }
                | Content::TemporalUpdate { .. } => "",
            })
            .collect();
        out.push_str(&format!("[{role}]\n{text}\n\n"));
    }
    out.trim_end().to_string()
}

struct ReceiverStream<T>(Receiver<T>);

impl<T> Stream for ReceiverStream<T> {
    type Item = T;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.0.poll_recv(cx)
    }
}
