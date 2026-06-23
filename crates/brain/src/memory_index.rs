use anyhow::Result;
use async_trait::async_trait;

use crate::models::conversation::ConversationId;

/// A conversation found by vector similarity search.
#[derive(Debug, Clone)]
pub struct RelatedConversation {
    pub id: ConversationId,
    /// Cosine similarity in [0, 1]; higher means more similar.
    pub score: f32,
}

/// Stores and retrieves per-conversation embedding vectors and the typed recall
/// edges that are derived from them.
///
/// Recall edges are memory-graph state, not conversation storage, which is why
/// they live here rather than on `ConversationStore`.
#[async_trait]
pub trait MemoryIndex: Send + Sync {
    /// Persists the embedding for a conversation's summary so it can be found
    /// by future similarity searches.
    async fn index_summary(&self, id: ConversationId, embedding: Vec<f32>) -> Result<()>;

    /// Returns up to `k` conversations whose embeddings are most similar to
    /// `query`, optionally excluding one conversation (typically the caller).
    async fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        exclude: Option<ConversationId>,
    ) -> Result<Vec<RelatedConversation>>;

    /// Persists a typed, scored recall edge from `from` to `to`. Idempotent:
    /// a second call with the same pair updates the score in place.
    async fn record_recall(
        &self,
        from: ConversationId,
        to: ConversationId,
        score: f32,
    ) -> Result<()>;

    /// Returns the ids of all conversations linked from `from` by a recall
    /// edge, in the order they were recorded.
    async fn recalled(&self, from: ConversationId) -> Result<Vec<ConversationId>>;
}
