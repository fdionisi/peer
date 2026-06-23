use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::{
    models::message::{Content, Role},
    tool::ToolDefinition,
};

/// The UI-facing stream: text chunks only. The orchestrator filters
/// `AssistantEvent::Text` events into this shape; tool calls are handled
/// internally and never reach the UI.
pub type ContentStream = Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send>>;

/// The internal stream the language model produces. The orchestrator consumes
/// this and decides what to do with each event: forward text to the UI, or
/// open a resolution thread for a tool call.
pub type AssistantEventStream = Pin<Box<dyn Stream<Item = anyhow::Result<AssistantEvent>> + Send>>;

/// A tool call emitted by the language model. The orchestrator uses `id` to
/// match results back to calls, `name` to look up the tool in the registry,
/// and `input` as the tool's argument.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// An event emitted by the language model during a completion.
#[derive(Debug, Clone, PartialEq)]
pub enum AssistantEvent {
    Text(String),
    ToolCall(ToolCall),
}

#[derive(Debug, Clone)]
pub struct Prompt {
    /// Summary of prior context, if any. The language model implementation is
    /// responsible for rendering this into a system prompt; the orchestrator
    /// only supplies the data.
    pub summary: Option<String>,
    /// Summaries from related prior conversations, surfaced via embedding
    /// similarity and resolved at prompt-assembly time.
    ///
    /// **System-prompt-only data.** This field must never be persisted as a
    /// `Message` or appear in any list of messages visible to the user. The
    /// model adapter renders it as a distinct fenced block inside the system
    /// prompt and nowhere else.
    pub recalled: Vec<String>,
    pub messages: Vec<PromptMessage>,
}

#[derive(Debug, Clone)]
pub struct PromptMessage {
    pub role: Role,
    pub content: Vec<Content>,
}

#[async_trait]
pub trait LanguageModel: Send + Sync {
    fn complete(&self, prompt: Prompt, tools: &[ToolDefinition]) -> AssistantEventStream;
    async fn remaining_capacity(&self, prompt: &Prompt) -> usize;
}
