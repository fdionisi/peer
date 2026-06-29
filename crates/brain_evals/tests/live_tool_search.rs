//! Live evaluation of `tool_search` invocation against the real Mistral
//! endpoint.
//!
//! This is the actual evaluation, not a format check: it runs the golden set
//! through a real `MistralClient` wrapped in a `RecordingLanguageModel`, with
//! fixture tools (a visible `web_search` and a hidden `echo`), and prints a
//! scored [`Report`]. It is `#[ignore]`d so the ordinary offline `cargo test`
//! stays fast and free.
//!
//! Run with:
//!   MISTRAL_API_KEY=<key> cargo test -p brain_evals --test live_tool_search -- --ignored --nocapture
//!
//! The model defaults to `mistral-small-latest` and can be overridden with
//! `MISTRAL_MODEL` to match whatever production runs.

use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;

use brain::Brain;
use brain::conversation_store::ConversationStore;
use brain::language_model::{
    AssistantEvent, AssistantEventStream, LanguageModel, Prompt, ToolCall,
};
use brain::models::message::Content;
use brain::tool::{Policy, StaticToolRegistry, Tool, ToolDefinition, ToolOutput, Visibility};
use brain::tool_index::ToolIndex;
use brain::tool_search::ToolSearch;
use brain_evals::tool_search::{
    CaseOutcome, InvocationOutcome, ObservedFirstCall, ToolSearchCase, ToolSearchMetrics,
    score_case,
};
use brain_evals::{Dataset, Report, RunMetadata};
use brain_mistralai::{MistralClient, MistralConfig};
use brain_prompts_embedded::EmbeddedPromptRegistry;

// ── Fixture tools ────────────────────────────────────────────────────────────
//
// Trivial by design: this eval tests *invocation* (does the model call
// `tool_search` when it should), not retrieval quality. The descriptions are
// just enough for the model to recognise the capability.

/// A visible `web_search` fixture. Returns a canned result so the tool loop
/// terminates cleanly if the model calls it.
struct VisibleWebSearch;

#[async_trait]
impl Tool for VisibleWebSearch {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for current information or facts you don't know."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" }
            },
            "required": ["query"]
        })
    }

    fn policy(&self) -> Policy {
        Policy::Auto
    }

    fn visibility(&self) -> Visibility {
        Visibility::Visible
    }

    async fn execute(&self, _input: serde_json::Value) -> Result<ToolOutput> {
        Ok(ToolOutput {
            text: "Search results: [fixture] no real results in eval mode.".to_string(),
            is_error: false,
        })
    }
}

/// A hidden `read_file` fixture. The model can only learn it exists by
/// calling `tool_search`; it is registered and callable but omitted from the
/// visible tool list. Unlike an `echo` fixture, this exercises a capability
/// the model provably lacks (filesystem access), so the only rational path
/// to satisfying the request is discovery — the model cannot just do it
/// natively.
struct HiddenReadFile;

#[async_trait]
impl Tool for HiddenReadFile {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file from the filesystem."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" }
            },
            "required": ["path"]
        })
    }

    fn policy(&self) -> Policy {
        Policy::Auto
    }

    fn visibility(&self) -> Visibility {
        Visibility::Hidden
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput> {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        Ok(ToolOutput {
            text: format!("[fixture] contents of {path}: hello world"),
            is_error: false,
        })
    }
}

// ── Recording LanguageModel ──────────────────────────────────────────────────

/// A `LanguageModel` wrapper that forwards every `complete` call to the inner
/// model and records the first `AssistantEvent::ToolCall` emitted on each
/// turn. The orchestrator consumes the forwarded stream unchanged; the wrapper
/// only observes.
///
/// The scorer reads the first tool call across the *entire* run — the
/// model's decision before any tool result has fed back. The per-turn slots
/// are kept so the runner can distinguish "no tool call on turn 0" from "tool
/// call on a later turn"; the first non-empty slot is the one that counts.
struct RecordingLanguageModel {
    inner: Arc<dyn LanguageModel>,
    first_calls: Arc<Mutex<Vec<ObservedFirstCall>>>,
}

impl RecordingLanguageModel {
    fn new(inner: Arc<dyn LanguageModel>) -> Self {
        Self {
            inner,
            first_calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The first tool call observed on each `complete` call, in call order.
    /// `tool_name` is `None` for a turn that produced no tool call.
    fn first_calls(&self) -> Vec<ObservedFirstCall> {
        self.first_calls.lock().unwrap().clone()
    }

    fn clear(&self) {
        self.first_calls.lock().unwrap().clear();
    }
}

#[async_trait]
impl LanguageModel for RecordingLanguageModel {
    fn complete(&self, prompt: Prompt, tools: &[ToolDefinition]) -> AssistantEventStream {
        let mut slots = self.first_calls.lock().unwrap();
        let turn_index = slots.len();
        slots.push(ObservedFirstCall { tool_name: None });
        drop(slots);

        let inner_stream = self.inner.complete(prompt, tools);
        let first_calls = self.first_calls.clone();

        Box::pin(async_stream::stream! {
            let mut pinned = inner_stream;
            while let Some(event) = pinned.next().await {
                if let Ok(AssistantEvent::ToolCall(ToolCall { ref name, .. })) = event {
                    let mut slots = first_calls.lock().unwrap();
                    let entry = slots
                        .get_mut(turn_index)
                        .expect("turn slot must exist; reserved before the stream started");
                    if entry.tool_name.is_none() {
                        entry.tool_name = Some(name.clone());
                    }
                }
                yield event;
            }
        })
    }

    async fn remaining_capacity(&self, prompt: &Prompt) -> usize {
        self.inner.remaining_capacity(prompt).await
    }
}

// ── Runner ───────────────────────────────────────────────────────────────────

/// Runs every case in `dataset` through `brain`, capturing the first tool
/// call the model emits per case via `recorder`, and grades each one.
///
/// The caller drives each case to completion (drains the `ContentStream`) so
/// the recorder's slots for that turn are settled before scoring. The first
/// tool call across the entire run is what the scorer reads: the model's
/// decision before any tool result has fed back. Later turns (after a
/// `tool_search` result, say) are downstream behaviour outside the scope of
/// this eval. The result is the scored outcomes, ready to aggregate with
/// [`ToolSearchMetrics::from_outcomes`] and wrap in a [`Report`].
async fn run_tool_search(
    brain: &Brain,
    recorder: &RecordingLanguageModel,
    dataset: &Dataset<ToolSearchCase>,
) -> Result<Vec<CaseOutcome>> {
    use brain::InputStream;
    use futures::stream;

    let mut outcomes = Vec::with_capacity(dataset.cases.len());

    for case in &dataset.cases {
        let input: InputStream = Box::pin(stream::iter(vec![Content::Text {
            text: case.user_message.clone(),
        }]));

        let stream = brain.say(input).await?;
        let mut s = stream;
        while let Some(chunk) = s.next().await {
            if chunk.is_err() {
                break;
            }
        }

        for _ in 0..32 {
            tokio::task::yield_now().await;
        }

        let first_calls = recorder.first_calls();

        let observed = first_calls
            .iter()
            .find_map(|fc| fc.tool_name.clone())
            .map(|name| ObservedFirstCall {
                tool_name: Some(name),
            })
            .unwrap_or(ObservedFirstCall { tool_name: None });
        outcomes.push(score_case(case, &observed));

        recorder.clear();
    }

    Ok(outcomes)
}

// ── Live eval ────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn tool_search_invocation_v1_live() {
    let api_key =
        std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY must be set to run live evals");
    let model =
        std::env::var("MISTRAL_MODEL").unwrap_or_else(|_| "mistral-small-latest".to_string());

    let prompts: Arc<dyn brain::prompts::PromptRegistry> = Arc::new(EmbeddedPromptRegistry::new());
    let prompt_hash = template_hash(prompts.as_ref(), "system");

    let mistral_config = MistralConfig::new(&api_key, &model).with_embed_model("mistral-embed");
    let mistral =
        Arc::new(MistralClient::new(mistral_config, prompts).expect("client construction"));
    let recorder = Arc::new(RecordingLanguageModel::new(mistral.clone()));

    let hidden_definitions = vec![ToolDefinition {
        name: "read_file".to_string(),
        description: HiddenReadFile.description().to_string(),
        input_schema: HiddenReadFile.input_schema(),
    }];

    let index = Arc::new(brain::testing::tool_index::InMemoryToolIndex::new());

    let tool_search = ToolSearch::new(
        mistral.clone() as Arc<dyn brain::embedder::Embedder>,
        index.clone() as Arc<dyn ToolIndex>,
        hidden_definitions,
    );

    let registry = Arc::new(StaticToolRegistry::new(vec![
        Box::new(VisibleWebSearch),
        Box::new(HiddenReadFile),
        Box::new(tool_search),
    ]));

    ToolIndex::index(
        &*index,
        &*registry,
        &*mistral as &dyn brain::embedder::Embedder,
    )
    .await
    .expect("indexing hidden tools");

    let store = Arc::new(brain::testing::conversation_store::InMemoryConversationStore::new());
    {
        let conv = store.create(None).await.expect("create conversation");
        store.set_current(conv.id).await.expect("set current");
    }

    let brain = Brain::new(
        store.clone(),
        Arc::new(brain::testing::summarizer::MockSummarizer::new("summary")),
        Arc::new(brain::testing::topic_detector::MockTopicDetector::no_shift()),
        recorder.clone() as Arc<dyn brain::language_model::LanguageModel>,
        mistral.clone() as Arc<dyn brain::embedder::Embedder>,
        Arc::new(brain::testing::memory_index::MockMemoryIndex::new()),
        registry,
        4096,
        0.8,
    );

    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join("tool_search_invocation.v1.jsonl");
    let dataset: Dataset<ToolSearchCase> = Dataset::load(&path).expect("dataset should load");

    let outcomes = run_tool_search(&brain, &recorder, &dataset)
        .await
        .expect("tool_search run failed");
    let metrics = ToolSearchMetrics::from_outcomes(&outcomes);

    let report = Report::new(
        RunMetadata::new(
            "tool_search_invocation",
            &dataset.version,
            &model,
            &prompt_hash,
        ),
        &metrics,
    );

    print_outcomes(&dataset, &outcomes);
    println!(
        "\n{}",
        serde_json::to_string_pretty(&report).expect("report serialises")
    );

    assert_eq!(metrics.total, dataset.len());
}

fn template_hash(registry: &dyn brain::prompts::PromptRegistry, name: &str) -> String {
    let template = registry
        .get(name)
        .unwrap_or_else(|| panic!("missing prompt template '{name}'"));
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    template.template.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Prints a per-case line so a disagreement is inspectable, not just a number.
fn print_outcomes(dataset: &Dataset<ToolSearchCase>, outcomes: &[CaseOutcome]) {
    println!("per-case outcomes ({} cases):", outcomes.len());
    for (case, outcome) in dataset.cases.iter().zip(outcomes) {
        let observed = outcome
            .observed
            .clone()
            .unwrap_or_else(|| "<none>".to_string());
        println!(
            "  {:<10} expected={:<12} observed={:<12} {:?}",
            case.id, outcome.expected, observed, outcome.outcome
        );
    }
}

// Keep the unused `InvocationOutcome` name in scope so the print macro's
// `{:?}` formatting has a stable type reference; the variant is part of the
// public scoring contract even though only `outcome.outcome` is read here.
#[allow(dead_code)]
fn _invocation_outcome_anchor(_: InvocationOutcome) {}
