use anyhow::Result;
use async_trait::async_trait;

use crate::models::message::Message;

/// What kind of summary the orchestrator is asking for. The summariser
/// implementation maps this to its own prompts; the orchestrator stays unaware
/// of prompt templates.
#[derive(Debug, Clone)]
pub enum SummaryRequest {
    /// Summarise the prior topic because the conversation has moved on.
    TopicShift { new_topic: String },
    /// Summarise to compact a conversation approaching the context limit.
    Compaction,
}

impl SummaryRequest {
    /// A short, stable identifier for logging.
    pub fn label(&self) -> &'static str {
        match self {
            SummaryRequest::TopicShift { .. } => "topic_shift",
            SummaryRequest::Compaction => "compaction",
        }
    }
}

#[async_trait]
pub trait Summarizer: Send + Sync {
    async fn summarize(&self, messages: &[Message], request: &SummaryRequest) -> Result<String>;
}
