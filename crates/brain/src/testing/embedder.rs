use std::sync::Mutex;

use anyhow::Result;
use async_trait::async_trait;

use crate::embedder::Embedder;

pub struct MockEmbedder {
    pub dimension: usize,
    pub calls: Mutex<Vec<String>>,
}

impl MockEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            calls: Mutex::new(Vec::new()),
        }
    }

    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

impl Default for MockEmbedder {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.calls.lock().unwrap().push(text.to_string());
        Ok(vec![0.0; self.dimension])
    }
}
