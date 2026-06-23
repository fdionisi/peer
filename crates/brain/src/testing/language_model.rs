use std::sync::Mutex;

use async_trait::async_trait;
use futures::stream;

use crate::language_model::{AssistantEvent, AssistantEventStream, LanguageModel, Prompt};
use crate::tool::ToolDefinition;

/// A scripted language model for tests. The script is a sequence of turns;
/// each turn is a `Vec<AssistantEvent>` emitted in order on successive
/// `complete` calls. When the script is exhausted, additional calls emit a
/// single `Text("response")` chunk.
pub struct MockLanguageModel {
    pub remaining_capacity: usize,
    turns: Mutex<Vec<Vec<AssistantEvent>>>,
    pub prompts: Mutex<Vec<Prompt>>,
    pub tools: Mutex<Vec<Vec<ToolDefinition>>>,
}

impl MockLanguageModel {
    /// Text-only mock that emits a single "response" chunk on every call.
    pub fn new(remaining_capacity: usize) -> Self {
        Self::with_events(
            remaining_capacity,
            vec![AssistantEvent::Text("response".to_string())],
        )
    }

    /// Text-only mock with a custom script of text chunks (single turn).
    pub fn with_chunks(remaining_capacity: usize, chunks: Vec<String>) -> Self {
        Self::with_events(
            remaining_capacity,
            chunks.into_iter().map(AssistantEvent::Text).collect(),
        )
    }

    /// Mock with a fully custom event script for the first call — text, tool
    /// calls, or both. Additional calls emit `Text("response")`.
    pub fn with_events(remaining_capacity: usize, events: Vec<AssistantEvent>) -> Self {
        Self::with_turns(remaining_capacity, vec![events])
    }

    /// Mock with multiple turns. Each `say` / tool-loop iteration consumes the
    /// next turn. Once all turns are consumed, additional calls emit a single
    /// `Text("response")` chunk.
    pub fn with_turns(remaining_capacity: usize, turns: Vec<Vec<AssistantEvent>>) -> Self {
        Self {
            remaining_capacity,
            turns: Mutex::new(turns),
            prompts: Mutex::new(Vec::new()),
            tools: Mutex::new(Vec::new()),
        }
    }

    pub fn prompts(&self) -> Vec<Prompt> {
        self.prompts.lock().unwrap().clone()
    }

    /// The `tools` slice the mock received on each `complete` call, in order.
    pub fn tool_calls(&self) -> Vec<Vec<ToolDefinition>> {
        self.tools.lock().unwrap().clone()
    }
}

#[async_trait]
impl LanguageModel for MockLanguageModel {
    fn complete(&self, prompt: Prompt, tools: &[ToolDefinition]) -> AssistantEventStream {
        self.prompts.lock().unwrap().push(prompt);
        self.tools.lock().unwrap().push(tools.to_vec());
        let events = self
            .turns
            .lock()
            .unwrap()
            .first()
            .cloned()
            .unwrap_or_else(|| vec![AssistantEvent::Text("response".to_string())]);

        let mut turns = self.turns.lock().unwrap();
        if !turns.is_empty() {
            turns.remove(0);
        }

        Box::pin(stream::iter(events.into_iter().map(Ok)))
    }

    async fn remaining_capacity(&self, _prompt: &Prompt) -> usize {
        self.remaining_capacity
    }
}
