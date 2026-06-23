use anyhow::Result;
use async_trait::async_trait;

use crate::models::message::{Message, MessageId};

#[derive(Debug, Clone)]
pub struct TopicShift {
    pub at_message_id: MessageId,
    pub new_topic: String,
}

#[async_trait]
pub trait TopicDetector: Send + Sync {
    async fn detect_shift(&self, messages: &[Message]) -> Result<Option<TopicShift>>;
}
