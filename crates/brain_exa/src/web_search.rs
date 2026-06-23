use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use brain::tool::{Policy, Tool, ToolDefinition, ToolOutput};

use crate::ExaConfig;

/// A web search tool backed by the Exa `/search` endpoint.
///
/// The tool is read-only: it returns a compact text rendering of the top
/// results (`title`, `url`, `snippet` per result) and never mutates anything
/// on the user's behalf. `policy()` is therefore `Auto` — no confirmation
/// thread is needed.
pub struct ExaWebSearch {
    client: Client,
    config: ExaConfig,
}

impl ExaWebSearch {
    pub fn new(config: ExaConfig) -> Result<Self> {
        let client = Client::builder()
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self { client, config })
    }

    fn endpoint(&self) -> String {
        format!("{}/search", self.config.base_url.trim_end_matches('/'))
    }
}

#[async_trait]
impl Tool for ExaWebSearch {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for current events, recent information, or anything you don't know. \
         Use this when the user's question requires up-to-date facts that are not in your training data."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                }
            },
            "required": ["query"]
        })
    }

    fn policy(&self) -> Policy {
        Policy::Auto
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .context("input must contain a `query` string")?;

        let request = SearchRequest {
            query: query.to_string(),
            contents: Contents {
                highlights: true,
                summary: true,
            },
        };

        let response = self
            .client
            .post(self.endpoint())
            .header("x-api-key", &self.config.api_key)
            .json(&request)
            .send()
            .await
            .context("exa search request failed")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("failed to read exa response body")?;

        if !status.is_success() {
            return Ok(ToolOutput {
                text: format!("Exa search failed (HTTP {status}): {body}"),
                is_error: true,
            });
        }

        let parsed: SearchResponse = serde_json::from_str(&body)
            .with_context(|| format!("failed to parse exa response: {body}"))?;

        Ok(ToolOutput {
            text: render_results(&parsed.results),
            is_error: false,
        })
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }
}

/// Render the result list as a compact, model-readable block. Each result gets
/// a numbered heading with title and URL, followed by the highlight or summary
/// snippet. Empty result sets render as a single line so the model can tell
/// the search returned nothing rather than failing.
fn render_results(results: &[SearchResult]) -> String {
    if results.is_empty() {
        return "No results.".to_string();
    }

    let mut out = String::new();
    for (i, r) in results.iter().enumerate() {
        let snippet = r
            .highlights
            .as_ref()
            .and_then(|h| h.first().cloned())
            .or_else(|| r.summary.clone())
            .unwrap_or_default();

        out.push_str(&format!(
            "{}. {}\n   {}\n   {}\n\n",
            i + 1,
            r.title,
            r.url,
            snippet.trim()
        ));
    }
    out.trim_end().to_string()
}

#[derive(Debug, Serialize)]
struct SearchRequest {
    query: String,
    contents: Contents,
}

#[derive(Debug, Serialize)]
struct Contents {
    highlights: bool,
    summary: bool,
}

#[derive(Debug, Deserialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
}

#[derive(Debug, Deserialize)]
struct SearchResult {
    title: String,
    url: String,
    #[serde(default)]
    highlights: Option<Vec<String>>,
    #[serde(default)]
    summary: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> ExaWebSearch {
        ExaWebSearch::new(ExaConfig::new("test-key")).expect("client construction failed")
    }

    // ── Offline: metadata ────────────────────────────────────────────────────

    #[test]
    fn name_is_web_search() {
        assert_eq!(tool().name(), "web_search");
    }

    #[test]
    fn policy_is_auto() {
        assert_eq!(tool().policy(), Policy::Auto);
    }

    #[test]
    fn input_schema_requires_query_string() {
        let schema = tool().input_schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["query"]["type"], "string");
        assert_eq!(schema["required"][0], "query");
    }

    #[test]
    fn definition_assembles_metadata() {
        let t = tool();
        let def = t.definition();
        assert_eq!(def.name, t.name());
        assert_eq!(def.description, t.description());
        assert_eq!(def.input_schema, t.input_schema());
    }

    // ── Offline: request shape ───────────────────────────────────────────────

    #[test]
    fn search_request_serialises_query_and_contents() {
        let req = SearchRequest {
            query: "rust async traits".to_string(),
            contents: Contents {
                highlights: true,
                summary: true,
            },
        };
        let json = serde_json::to_value(&req).unwrap();

        assert_eq!(json["query"], "rust async traits");
        assert_eq!(json["contents"]["highlights"], true);
        assert_eq!(json["contents"]["summary"], true);
    }

    #[test]
    fn endpoint_strips_trailing_slash() {
        let t =
            ExaWebSearch::new(ExaConfig::new("k").with_base_url("https://api.exa.ai/")).unwrap();
        assert_eq!(t.endpoint(), "https://api.exa.ai/search");
    }

    // ── Offline: response rendering ──────────────────────────────────────────

    #[test]
    fn render_results_emits_title_url_and_snippet_per_result() {
        let results = vec![
            SearchResult {
                title: "Async traits in Rust".to_string(),
                url: "https://example.com/a".to_string(),
                highlights: Some(vec!["A short highlight.".to_string()]),
                summary: Some("A summary.".to_string()),
            },
            SearchResult {
                title: "Tokio docs".to_string(),
                url: "https://example.com/b".to_string(),
                highlights: None,
                summary: Some("Tokio summary.".to_string()),
            },
        ];

        let rendered = render_results(&results);

        assert!(rendered.contains("1. Async traits in Rust"));
        assert!(rendered.contains("https://example.com/a"));
        assert!(rendered.contains("A short highlight."));
        assert!(rendered.contains("2. Tokio docs"));
        assert!(rendered.contains("Tokio summary."));
    }

    #[test]
    fn render_results_falls_back_to_summary_when_highlights_absent() {
        let results = vec![SearchResult {
            title: "Only summary".to_string(),
            url: "https://example.com/c".to_string(),
            highlights: None,
            summary: Some("Just a summary.".to_string()),
        }];

        let rendered = render_results(&results);
        assert!(rendered.contains("Just a summary."));
    }

    #[test]
    fn render_results_handles_empty_list() {
        assert_eq!(render_results(&[]), "No results.");
    }

    #[test]
    fn render_results_handles_missing_snippet() {
        let results = vec![SearchResult {
            title: "Bare result".to_string(),
            url: "https://example.com/d".to_string(),
            highlights: None,
            summary: None,
        }];

        let rendered = render_results(&results);
        assert!(rendered.contains("1. Bare result"));
        assert!(rendered.contains("https://example.com/d"));
    }

    #[test]
    fn search_response_parses_results_array() {
        let body = r#"{
            "results": [
                {
                    "title": "First",
                    "url": "https://example.com/1",
                    "highlights": ["h1"]
                }
            ]
        }"#;

        let parsed: SearchResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.results.len(), 1);
        assert_eq!(parsed.results[0].title, "First");
        assert_eq!(parsed.results[0].highlights.as_deref().unwrap()[0], "h1");
        assert!(parsed.results[0].summary.is_none());
    }

    // ── Live (requires EXA_API_KEY) ──────────────────────────────────────────

    /// Calls the real Exa `/search` endpoint with a stable query and asserts a
    /// non-empty result set. Run with:
    ///   EXA_API_KEY=<key> cargo test -p brain_exa search_live -- --ignored
    #[tokio::test]
    #[ignore]
    async fn search_live_returns_results() {
        let api_key =
            std::env::var("EXA_API_KEY").expect("EXA_API_KEY must be set to run live tests");

        let tool = ExaWebSearch::new(ExaConfig::new(api_key)).expect("client construction failed");

        let output = tool
            .execute(serde_json::json!({ "query": "rust programming language" }))
            .await
            .expect("search request failed");

        assert!(
            !output.is_error,
            "live search returned an error: {}",
            output.text
        );
        assert!(
            !output.text.contains("No results."),
            "live search returned no results"
        );
    }
}
