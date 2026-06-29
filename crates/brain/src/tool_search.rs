use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;

use crate::embedder::Embedder;
use crate::tool::{Policy, Tool, ToolDefinition, ToolOutput, Visibility};
use crate::tool_index::ToolIndex;

/// How many hidden tools `tool_search` surfaces per call. Enough to give the
/// model a real choice without re-flooding the context the discovery mechanism
/// exists to keep lean.
const TOOL_SEARCH_K: usize = 5;

/// A visible tool the model calls to discover hidden tools by natural-language
/// query.
///
/// Unlike Anthropic's server-side `tool_search`, this is an ordinary `Auto`
/// tool: the orchestrator runs it through `ToolRegistry::execute` like any
/// other. Its `execute` embeds the query, searches the `ToolIndex`, resolves
/// the returned names to `ToolDefinition`s from the snapshot captured at
/// construction, and renders the results as text. The model then calls one of
/// the discovered tools by name; the registry already has it registered
/// (hidden ≠ unregistered), so the call succeeds without the orchestrator
/// growing any tool list.
///
/// `ToolSearch` holds a snapshot of the hidden tool definitions rather than a
/// live reference to the registry. This avoids a circular dependency — the tool
/// is owned by the registry, so it cannot also hold the registry — at the cost
/// of dynamism: tools registered after `ToolSearch` is constructed are not
/// discoverable. That's acceptable because `StaticToolRegistry` is immutable
/// by construction, so the snapshot never goes stale in practice. If dynamic
/// tool registration is added later, this design will need to be revisited
/// alongside the `ToolIndex` (which is also built once at startup).
///
/// The registry is the source of truth for execution; the index stores only
/// names and embeddings; `ToolSearch` carries the definitions it needs to
/// render results. Discovery is always on: if there are no hidden tools,
/// `index()` is a no-op and `search()` returns empty, so the tool is present
/// but inert.
pub struct ToolSearch {
    embedder: Arc<dyn Embedder>,
    index: Arc<dyn ToolIndex>,
    definitions: Vec<ToolDefinition>,
}

impl ToolSearch {
    /// Construct from the hidden tool definitions captured at startup.
    ///
    /// The caller is responsible for filtering to hidden tools — typically by
    /// calling `registry.definitions()` and keeping those whose
    /// `registry.visibility(name) == Some(Visibility::Hidden)`. Passing visible
    /// tools here is harmless but wasteful: they'd be discoverable twice (once
    /// in the visible list, once via search).
    pub fn new(
        embedder: Arc<dyn Embedder>,
        index: Arc<dyn ToolIndex>,
        definitions: Vec<ToolDefinition>,
    ) -> Self {
        Self {
            embedder,
            index,
            definitions,
        }
    }
}

#[async_trait]
impl Tool for ToolSearch {
    fn name(&self) -> &str {
        "tool_search"
    }

    fn description(&self) -> &str {
        "Search the deferred tool catalog for capabilities not in the visible tool list. \
         Call this before assuming a capability is unavailable. Returns the names, \
         descriptions, and input schemas of matching tools so you can call them \
         directly by name."
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language description of the capability you need."
                }
            },
            "required": ["query"]
        })
    }

    fn policy(&self) -> Policy {
        Policy::Auto
    }

    fn visibility(&self) -> Visibility {
        Visibility::Visible
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .context("input must contain a `query` string")?;

        let embedding = self.embedder.embed(query).await?;
        let discovered = self.index.search(embedding, TOOL_SEARCH_K).await?;

        if discovered.is_empty() {
            return Ok(ToolOutput {
                text: "~No matching tools found.".to_string(),
                is_error: false,
            });
        }

        let definitions: Vec<_> = discovered
            .iter()
            .filter_map(|d| {
                self.definitions
                    .iter()
                    .find(|def| def.name == d.name)
                    .map(|def| (d.score, def.clone()))
            })
            .collect();

        if definitions.is_empty() {
            return Ok(ToolOutput {
                text: "No matching tools found.".to_string(),
                is_error: false,
            });
        }

        let mut text = String::from("Discovered tools:\n\n");
        for (score, def) in definitions {
            text.push_str(&format!(
                "- {} (score: {:.3}): {}\n  schema: {}\n",
                def.name, score, def.description, def.input_schema
            ));
        }
        text.push_str("\nCall any of these tools by name to use it.");

        Ok(ToolOutput {
            text,
            is_error: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::embedder::MockEmbedder;
    use crate::testing::tool_index::InMemoryToolIndex;
    use crate::testing::tool_registry::MockToolRegistry;
    use crate::tool::{ToolDefinition, ToolOutput, Visibility};

    /// Build a registry with a single hidden `echo` tool.
    fn hidden_echo_registry() -> MockToolRegistry {
        MockToolRegistry::new().register_hidden(
            "echo",
            "Echo text back",
            Policy::Auto,
            vec![ToolOutput {
                text: "ok".to_string(),
                is_error: false,
            }],
        )
    }

    /// Extract the hidden tool definitions from a registry, as the CLI will
    /// do at startup when constructing `ToolSearch`.
    fn hidden_definitions(registry: &dyn crate::tool::ToolRegistry) -> Vec<ToolDefinition> {
        registry
            .definitions()
            .into_iter()
            .filter(|d| registry.visibility(&d.name) == Some(Visibility::Hidden))
            .collect()
    }

    fn make_search(
        embedder: Arc<MockEmbedder>,
        index: Arc<InMemoryToolIndex>,
        definitions: Vec<ToolDefinition>,
    ) -> ToolSearch {
        ToolSearch::new(embedder, index, definitions)
    }

    #[tokio::test]
    async fn execute_returns_text_describing_discovered_tools() {
        let embedder = Arc::new(MockEmbedder::new(2));
        let index = Arc::new(InMemoryToolIndex::new());
        let registry = hidden_echo_registry();
        index.index(&registry, &*embedder).await.unwrap();

        let tool = make_search(embedder, index, hidden_definitions(&registry));
        let result = tool
            .execute(serde_json::json!({ "query": "repeat what I say" }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.text.contains("echo"));
        assert!(result.text.contains("Echo text back"));
        assert!(result.text.contains("schema"));
    }

    #[tokio::test]
    async fn execute_returns_no_match_when_index_is_empty() {
        let embedder = Arc::new(MockEmbedder::new(2));
        let index = Arc::new(InMemoryToolIndex::new());

        let tool = make_search(embedder, index, Vec::new());
        let result = tool
            .execute(serde_json::json!({ "query": "anything" }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(result.text, "No matching tools found.");
    }

    #[tokio::test]
    async fn execute_errors_when_query_is_missing() {
        let embedder = Arc::new(MockEmbedder::new(2));
        let index = Arc::new(InMemoryToolIndex::new());

        let tool = make_search(embedder, index, Vec::new());
        let err = tool.execute(serde_json::json!({})).await.unwrap_err();

        assert!(err.to_string().contains("query"));
    }

    #[tokio::test]
    async fn execute_returns_no_match_when_discovered_name_not_in_snapshot() {
        let embedder = Arc::new(MockEmbedder::new(2));
        let index = Arc::new(InMemoryToolIndex::new());
        let registry = hidden_echo_registry();
        index.index(&registry, &*embedder).await.unwrap();

        let tool = make_search(embedder, index, Vec::new());
        let result = tool
            .execute(serde_json::json!({ "query": "echo" }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(result.text, "No matching tools found.");
    }

    #[test]
    fn definition_is_honest_about_schema() {
        let embedder = Arc::new(MockEmbedder::new(2));
        let index = Arc::new(InMemoryToolIndex::new());
        let tool = make_search(embedder, index, Vec::new());

        let def = tool.definition();
        assert_eq!(def.name, "tool_search");
        assert!(def.description.contains("deferred tool catalog"));
        assert_eq!(
            def.input_schema,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A natural-language description of the capability you need."
                    }
                },
                "required": ["query"]
            })
        );
    }

    #[test]
    fn is_visible_and_auto() {
        let embedder = Arc::new(MockEmbedder::new(2));
        let index = Arc::new(InMemoryToolIndex::new());
        let tool = make_search(embedder, index, Vec::new());

        assert_eq!(tool.visibility(), Visibility::Visible);
        assert_eq!(tool.policy(), Policy::Auto);
        assert!(tool.is_visible());
    }

    #[tokio::test]
    async fn execute_includes_score_in_output() {
        let embedder = Arc::new(MockEmbedder::new(2));
        let index = Arc::new(InMemoryToolIndex::new());
        let registry = hidden_echo_registry();
        index.index(&registry, &*embedder).await.unwrap();

        let tool = make_search(embedder, index, hidden_definitions(&registry));
        let result = tool
            .execute(serde_json::json!({ "query": "echo" }))
            .await
            .unwrap();

        assert!(result.text.contains("score:"));
    }

    #[tokio::test]
    async fn execute_resolves_multiple_tools() {
        let embedder = Arc::new(MockEmbedder::new(2));
        let index = Arc::new(InMemoryToolIndex::new());
        let registry = MockToolRegistry::new()
            .register_hidden(
                "echo",
                "Echo text back",
                Policy::Auto,
                vec![ToolOutput {
                    text: "ok".to_string(),
                    is_error: false,
                }],
            )
            .register_hidden(
                "reverse",
                "Reverse text",
                Policy::Auto,
                vec![ToolOutput {
                    text: "ok".to_string(),
                    is_error: false,
                }],
            );
        index.index(&registry, &*embedder).await.unwrap();

        let tool = make_search(embedder, index, hidden_definitions(&registry));
        let result = tool
            .execute(serde_json::json!({ "query": "text" }))
            .await
            .unwrap();

        assert!(result.text.contains("echo"));
        assert!(result.text.contains("reverse"));
    }

    #[tokio::test]
    async fn execute_truncates_to_k_results() {
        let embedder = Arc::new(MockEmbedder::new(2));
        let index = Arc::new(InMemoryToolIndex::new());
        let registry = MockToolRegistry::new()
            .register_hidden("a", "alpha", Policy::Auto, vec![])
            .register_hidden("b", "beta", Policy::Auto, vec![])
            .register_hidden("c", "gamma", Policy::Auto, vec![])
            .register_hidden("d", "delta", Policy::Auto, vec![])
            .register_hidden("e", "epsilon", Policy::Auto, vec![])
            .register_hidden("f", "zeta", Policy::Auto, vec![]);
        index.index(&registry, &*embedder).await.unwrap();

        let tool = make_search(embedder, index, hidden_definitions(&registry));
        let result = tool
            .execute(serde_json::json!({ "query": "greek" }))
            .await
            .unwrap();

        let count = result.text.matches("- ").count();
        assert!(count <= TOOL_SEARCH_K, "got {count} results");
    }

    #[test]
    fn tool_search_k_is_five() {
        assert_eq!(TOOL_SEARCH_K, 5);
    }

    #[test]
    fn tool_definition_has_expected_shape() {
        let embedder = Arc::new(MockEmbedder::new(2));
        let index = Arc::new(InMemoryToolIndex::new());
        let tool = make_search(embedder, index, Vec::new());

        let def: ToolDefinition = tool.definition();
        assert!(!def.name.is_empty());
        assert!(!def.description.is_empty());
        assert!(def.input_schema.is_object());
    }
}
