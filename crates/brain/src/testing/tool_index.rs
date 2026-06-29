use anyhow::Result;
use async_trait::async_trait;

use crate::embedder::Embedder;
use crate::tool::{ToolDefinition, ToolRegistry, Visibility};
use crate::tool_index::{DiscoveredTool, ToolIndex};

fn tool_embedding_text(definition: &ToolDefinition) -> String {
    format!("{}\n{}", definition.name, definition.description)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(0.0, 1.0)
}

/// In-memory `ToolIndex` for tests.
#[derive(Default)]
pub struct InMemoryToolIndex {
    entries: tokio::sync::Mutex<Vec<(String, Vec<f32>)>>,
}

impl InMemoryToolIndex {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ToolIndex for InMemoryToolIndex {
    async fn index(&self, registry: &dyn ToolRegistry, embedder: &dyn Embedder) -> Result<()> {
        let mut entries = Vec::new();

        for definition in registry.definitions() {
            if registry.visibility(&definition.name) != Some(Visibility::Hidden) {
                continue;
            }
            let text = tool_embedding_text(&definition);
            let embedding = embedder.embed(&text).await?;
            entries.push((definition.name.clone(), embedding));
        }

        let mut store = self.entries.lock().await;
        *store = entries;
        Ok(())
    }

    async fn search(&self, query: Vec<f32>, k: usize) -> Result<Vec<DiscoveredTool>> {
        let store = self.entries.lock().await;
        if store.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let mut scored: Vec<DiscoveredTool> = store
            .iter()
            .map(|(name, embedding)| DiscoveredTool {
                name: name.clone(),
                score: cosine_similarity(query.as_slice(), embedding.as_slice()),
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::embedder::MockEmbedder;
    use crate::testing::tool_registry::MockToolRegistry;
    use crate::tool::{Policy, ToolOutput};

    fn hidden_tool(name: &str, description: &str) -> MockToolRegistry {
        MockToolRegistry::new().register_hidden(
            name,
            description,
            Policy::Auto,
            vec![ToolOutput {
                text: "ok".to_string(),
                is_error: false,
            }],
        )
    }

    fn visible_tool(name: &str, description: &str) -> MockToolRegistry {
        MockToolRegistry::new().register(
            name,
            description,
            Policy::Auto,
            vec![ToolOutput {
                text: "ok".to_string(),
                is_error: false,
            }],
        )
    }

    #[tokio::test]
    async fn index_only_includes_hidden_tools() {
        let registry = MockToolRegistry::new()
            .register_hidden(
                "hidden_one",
                "A hidden tool.",
                Policy::Auto,
                vec![ToolOutput {
                    text: "ok".to_string(),
                    is_error: false,
                }],
            )
            .register(
                "visible_one",
                "A visible tool.",
                Policy::Auto,
                vec![ToolOutput {
                    text: "ok".to_string(),
                    is_error: false,
                }],
            );

        let embedder = MockEmbedder::new(4);
        let index = InMemoryToolIndex::new();
        index.index(&registry, &embedder).await.unwrap();

        assert_eq!(embedder.calls().len(), 1);
        assert!(embedder.calls()[0].contains("hidden_one"));

        let results = index.search(vec![0.0; 4], 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "hidden_one");
    }

    #[tokio::test]
    async fn index_skips_registry_with_no_hidden_tools() {
        let registry = visible_tool("only_visible", "The only tool.");
        let embedder = MockEmbedder::new(4);
        let index = InMemoryToolIndex::new();
        index.index(&registry, &embedder).await.unwrap();

        assert!(embedder.calls().is_empty());
        let results = index.search(vec![0.0; 4], 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_returns_top_k_ordered_by_descending_score() {
        let registry = MockToolRegistry::new()
            .register_hidden("a", "alpha", Policy::Auto, vec![])
            .register_hidden("b", "beta", Policy::Auto, vec![])
            .register_hidden("c", "gamma", Policy::Auto, vec![]);
        let embedder = MockEmbedder::new(2);
        let index = InMemoryToolIndex::new();
        index.index(&registry, &embedder).await.unwrap();

        let results = index.search(vec![0.0, 0.0], 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].name, "a");
        assert_eq!(results[1].name, "b");
    }

    #[tokio::test]
    async fn search_with_zero_k_returns_empty() {
        let registry = hidden_tool("solo", "the only hidden tool");
        let embedder = MockEmbedder::new(2);
        let index = InMemoryToolIndex::new();
        index.index(&registry, &embedder).await.unwrap();

        let results = index.search(vec![0.0, 0.0], 0).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_before_index_returns_empty() {
        let index = InMemoryToolIndex::new();
        let results = index.search(vec![1.0, 2.0], 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn reindex_replaces_previous_entries() {
        let first = hidden_tool("old", "the old hidden tool");
        let embedder = MockEmbedder::new(2);
        let index = InMemoryToolIndex::new();
        index.index(&first, &embedder).await.unwrap();

        let second = hidden_tool("new", "the new hidden tool");
        index.index(&second, &embedder).await.unwrap();

        let results = index.search(vec![0.0, 0.0], 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "new");
    }

    #[tokio::test]
    async fn index_embeds_name_and_description_together() {
        let registry = hidden_tool("desktop_click", "Click an element via AX press.");
        let embedder = MockEmbedder::new(2);
        let index = InMemoryToolIndex::new();
        index.index(&registry, &embedder).await.unwrap();

        let embedded = embedder.calls();
        assert_eq!(embedded.len(), 1);
        assert!(embedded[0].contains("desktop_click"));
        assert!(embedded[0].contains("Click an element via AX press."));
    }

    #[test]
    fn cosine_similarity_identical_vectors_is_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_zero_vector_returns_zero() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_similarity_clamps_negatives_to_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn tool_embedding_text_is_name_then_description() {
        let definition = ToolDefinition {
            name: "desktop_click".to_string(),
            description: "Click an element via accessibility press.".to_string(),
            input_schema: serde_json::json!({ "type": "object" }),
        };
        assert_eq!(
            tool_embedding_text(&definition),
            "desktop_click\nClick an element via accessibility press."
        );
    }
}
