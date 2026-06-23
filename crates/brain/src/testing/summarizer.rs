use std::sync::Mutex;

use anyhow::Result;
use async_trait::async_trait;

use crate::models::message::Message;
use crate::summarizer::SummaryRequest;

pub struct MockSummarizer {
    pub response: String,
    pub calls: Mutex<Vec<Vec<Message>>>,
}

impl MockSummarizer {
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
            calls: Mutex::new(Vec::new()),
        }
    }

    pub fn calls(&self) -> Vec<Vec<Message>> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl crate::summarizer::Summarizer for MockSummarizer {
    async fn summarize(&self, messages: &[Message], _request: &SummaryRequest) -> Result<String> {
        self.calls.lock().unwrap().push(messages.to_vec());
        Ok(self.response.clone())
    }
}
