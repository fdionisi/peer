use std::sync::Mutex;

use anyhow::Result;
use async_trait::async_trait;

use crate::models::message::Message;
use crate::topic_detector::{TopicDetector, TopicShift};

pub struct MockTopicDetector {
    pub shift: Mutex<Option<TopicShift>>,
    pub calls: Mutex<Vec<Vec<Message>>>,
}

impl MockTopicDetector {
    pub fn no_shift() -> Self {
        Self {
            shift: Mutex::new(None),
            calls: Mutex::new(Vec::new()),
        }
    }

    pub fn with_shift(shift: TopicShift) -> Self {
        Self {
            shift: Mutex::new(Some(shift)),
            calls: Mutex::new(Vec::new()),
        }
    }

    pub fn calls(&self) -> Vec<Vec<Message>> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl TopicDetector for MockTopicDetector {
    async fn detect_shift(&self, messages: &[Message]) -> Result<Option<TopicShift>> {
        self.calls.lock().unwrap().push(messages.to_vec());
        Ok(self.shift.lock().unwrap().take())
    }
}
