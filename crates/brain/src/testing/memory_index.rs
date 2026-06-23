use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;

use crate::memory_index::{MemoryIndex, RelatedConversation};
use crate::models::conversation::ConversationId;

#[derive(Debug, Clone)]
pub struct IndexSummaryCall {
    pub id: ConversationId,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct RecordRecallCall {
    pub from: ConversationId,
    pub to: ConversationId,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct SearchCall {
    pub query: Vec<f32>,
    pub k: usize,
    pub exclude: Option<ConversationId>,
}

pub struct MockMemoryIndex {
    /// Seeded results returned by `search` in order. Each call to `search`
    /// pops the front of the queue; once exhausted, `search` returns empty.
    pub search_results: Mutex<Vec<Vec<RelatedConversation>>>,
    /// Recorded `index_summary` calls.
    pub indexed: Mutex<Vec<IndexSummaryCall>>,
    /// Recorded `record_recall` calls.
    pub recalls: Mutex<Vec<RecordRecallCall>>,
    /// Stored recall edges, keyed by `from`. `recalled(from)` returns the
    /// current values for `from` in insertion order.
    pub edges: Mutex<std::collections::HashMap<ConversationId, Vec<ConversationId>>>,
    /// Optional signal that fires when `index_summary` completes, so tests can
    /// deterministically wait for the detached write task.
    pub index_signal: Mutex<Option<Arc<tokio::sync::Notify>>>,
    pub search_calls: Mutex<Vec<SearchCall>>,
}

impl MockMemoryIndex {
    pub fn new() -> Self {
        Self {
            search_results: Mutex::new(Vec::new()),
            indexed: Mutex::new(Vec::new()),
            recalls: Mutex::new(Vec::new()),
            edges: Mutex::new(std::collections::HashMap::new()),
            index_signal: Mutex::new(None),
            search_calls: Mutex::new(Vec::new()),
        }
    }

    /// Seeds the next `search` call to return `results`.
    pub fn push_search(&self, results: Vec<RelatedConversation>) {
        self.search_results.lock().unwrap().push(results);
    }

    pub fn indexed(&self) -> Vec<IndexSummaryCall> {
        self.indexed.lock().unwrap().clone()
    }

    pub fn recalls(&self) -> Vec<RecordRecallCall> {
        self.recalls.lock().unwrap().clone()
    }

    pub fn search_calls(&self) -> Vec<SearchCall> {
        self.search_calls.lock().unwrap().clone()
    }

    /// Enables a `Notify` that fires after each `index_summary` call. Tests
    /// can `await` on it to observe the detached write deterministically.
    pub fn enable_index_signal(&self) {
        *self.index_signal.lock().unwrap() = Some(Arc::new(tokio::sync::Notify::new()));
    }

    pub fn index_notify(&self) -> Option<Arc<tokio::sync::Notify>> {
        self.index_signal.lock().unwrap().clone()
    }
}

impl Default for MockMemoryIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryIndex for MockMemoryIndex {
    async fn index_summary(&self, id: ConversationId, embedding: Vec<f32>) -> Result<()> {
        self.indexed
            .lock()
            .unwrap()
            .push(IndexSummaryCall { id, embedding });
        if let Some(notify) = self.index_signal.lock().unwrap().as_ref() {
            notify.notify_one();
        }
        Ok(())
    }

    async fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        exclude: Option<ConversationId>,
    ) -> Result<Vec<RelatedConversation>> {
        self.search_calls
            .lock()
            .unwrap()
            .push(SearchCall { query, k, exclude });
        let mut queue = self.search_results.lock().unwrap();
        if queue.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(queue.remove(0))
        }
    }

    async fn record_recall(
        &self,
        from: ConversationId,
        to: ConversationId,
        score: f32,
    ) -> Result<()> {
        self.recalls
            .lock()
            .unwrap()
            .push(RecordRecallCall { from, to, score });
        self.edges.lock().unwrap().entry(from).or_default().push(to);
        Ok(())
    }

    async fn recalled(&self, from: ConversationId) -> Result<Vec<ConversationId>> {
        Ok(self
            .edges
            .lock()
            .unwrap()
            .get(&from)
            .cloned()
            .unwrap_or_default())
    }
}
