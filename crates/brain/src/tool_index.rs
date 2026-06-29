use anyhow::Result;
use async_trait::async_trait;

use crate::embedder::Embedder;
use crate::tool::ToolRegistry;

/// A hidden tool found by similarity search. `score` is cosine similarity in [0, 1].
#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveredTool {
    pub name: String,
    pub score: f32,
}

/// Stores embeddings for hidden tools and searches them by similarity.
#[async_trait]
pub trait ToolIndex: Send + Sync {
    /// Rebuilds the index from a registry's hidden tools.
    async fn index(&self, registry: &dyn ToolRegistry, embedder: &dyn Embedder) -> Result<()>;

    /// Returns up to `k` tools whose embeddings are most similar to `query`.
    async fn search(&self, query: Vec<f32>, k: usize) -> Result<Vec<DiscoveredTool>>;
}
