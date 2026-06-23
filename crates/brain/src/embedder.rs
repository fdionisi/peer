use anyhow::Result;
use async_trait::async_trait;

/// Converts a text string into a dense vector representation suitable for
/// semantic similarity search. The dimension of the returned vector is
/// determined by the underlying model and must match whatever the `MemoryIndex`
/// implementation was configured with.
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}
