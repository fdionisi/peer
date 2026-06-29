//! Evaluation of `tool_search` invocation behaviour.
//!
//! # What the action decides
//!
//! Tool discovery is a context-management strategy: the visible tool list is
//! partial, and the model is told `tool_search` is free and should be called
//! before assuming a capability is unavailable. This eval answers one question
//! per case: did the model's *first* tool call match the expected tool?
//!
//! - When the request needs a hidden capability, the first call should be
//!   `tool_search` — the model searches before giving up.
//! - When a visible tool already covers the request, the first call should be
//!   that tool directly — the model does not search for alternatives it
//!   already has.
//!
//! # Why the first call, and only the first
//!
//! The discovery mechanism is invoked (or not) at the very first tool call.
//! Anything after that — executing the discovered tool, calling a visible
//! tool's result — is downstream behaviour outside the scope of this eval.
//! Retrieval quality (does `tool_search` return the right tool?) is a
//! separate concern, deferred to a later eval against a real tool corpus.
//!
//! # How the first call is observed
//!
//! The orchestrator filters tool calls out of the public `ContentStream`, so
//! the eval cannot read them from `Brain::say`. Instead the live test wraps
//! the `LanguageModel` in a recording adapter (defined in the test file)
//! that forwards every `complete` call and captures the first
//! `AssistantEvent::ToolCall` per turn. The scorer then reads the first
//! non-empty slot across the whole run — the model's decision before any
//! tool result has fed back. This observes the model's behaviour at the
//! trait boundary — the same boundary `Brain` consumes — without coupling
//! to orchestrator internals.
//!
//! This module holds only the pure scoring pieces: the dataset case schema,
//! the scorer, and the metrics. The runner that drives `Brain` and the
//! recording adapter live in the test file because they depend on `tokio`,
//! `futures`, and `async-stream`, which are dev-dependencies — the scorer
//! itself is pure and dependency-free.

use serde::{Deserialize, Serialize};

use crate::dataset::Dataset;

// ── Dataset case schema ──────────────────────────────────────────────────────

/// One golden case for `tool_search` invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSearchCase {
    pub id: String,
    pub user_message: String,
    pub expect: ToolSearchExpectation,
}

/// The expected first tool call for a case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSearchExpectation {
    pub first_tool_call: String,
}

// ── Scoring ──────────────────────────────────────────────────────────────────

/// The first tool call the model emitted on a case, or `None` if it produced
/// no tool call before the turn ended.
#[derive(Debug, Clone, Serialize)]
pub struct ObservedFirstCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

/// Where a single case landed: did the first tool call match the expectation?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InvocationOutcome {
    /// The first tool call matched the expected tool name.
    Match,
    /// The model made a tool call, but it was not the expected one.
    Mismatch,
    /// The model produced no tool call at all before the turn ended.
    NoCall,
}

/// The graded result of a single case. A pure function of the case and the
/// observed first call — it records no model, threshold, or timing.
#[derive(Debug, Clone, Serialize)]
pub struct CaseOutcome {
    pub case_id: String,
    pub outcome: InvocationOutcome,
    pub expected: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed: Option<String>,
}

/// Grades a single case against the observed first tool call. Pure and
/// deterministic.
pub fn score_case(case: &ToolSearchCase, observed: &ObservedFirstCall) -> CaseOutcome {
    let expected = case.expect.first_tool_call.as_str();
    let (outcome, observed_name) = match &observed.tool_name {
        Some(name) => {
            let o = if name == expected {
                InvocationOutcome::Match
            } else {
                InvocationOutcome::Mismatch
            };
            (o, Some(name.clone()))
        }
        None => (InvocationOutcome::NoCall, None),
    };

    CaseOutcome {
        case_id: case.id.clone(),
        outcome,
        expected: expected.to_string(),
        observed: observed_name,
    }
}

// ── Aggregate metrics ────────────────────────────────────────────────────────

/// Aggregated `tool_search` invocation metrics over a run.
///
/// `match_rate` is the headline number: of all cases, how often did the model
/// make the expected first call. Read it alongside `total` to know it is not
/// vacuous, and alongside `no_call` / `mismatch` counts to know what the
/// failures looked like.
#[derive(Debug, Clone, Serialize)]
pub struct ToolSearchMetrics {
    pub total: usize,
    pub matches: usize,
    pub mismatches: usize,
    pub no_calls: usize,
    pub match_rate: f64,
}

impl ToolSearchMetrics {
    pub fn from_outcomes(outcomes: &[CaseOutcome]) -> Self {
        let mut matches = 0;
        let mut mismatches = 0;
        let mut no_calls = 0;

        for outcome in outcomes {
            match outcome.outcome {
                InvocationOutcome::Match => matches += 1,
                InvocationOutcome::Mismatch => mismatches += 1,
                InvocationOutcome::NoCall => no_calls += 1,
            }
        }

        let total = outcomes.len();
        let match_rate = if total == 0 {
            0.0
        } else {
            matches as f64 / total as f64
        };

        Self {
            total,
            matches,
            mismatches,
            no_calls,
            match_rate,
        }
    }
}

// ── Compile-time anchor: Dataset is parametric over the case type ────────────

#[allow(dead_code)]
fn _dataset_is_parametric(dataset: &Dataset<ToolSearchCase>) -> usize {
    dataset.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(id: &str, msg: &str, expected: &str) -> ToolSearchCase {
        ToolSearchCase {
            id: id.to_string(),
            user_message: msg.to_string(),
            expect: ToolSearchExpectation {
                first_tool_call: expected.to_string(),
            },
        }
    }

    #[test]
    fn score_case_match_when_observed_equals_expected() {
        let c = case("a", "anything", "tool_search");
        let observed = ObservedFirstCall {
            tool_name: Some("tool_search".to_string()),
        };
        let outcome = score_case(&c, &observed);
        assert_eq!(outcome.outcome, InvocationOutcome::Match);
        assert_eq!(outcome.observed.as_deref(), Some("tool_search"));
    }

    #[test]
    fn score_case_mismatch_when_observed_differs() {
        let c = case("a", "anything", "tool_search");
        let observed = ObservedFirstCall {
            tool_name: Some("web_search".to_string()),
        };
        let outcome = score_case(&c, &observed);
        assert_eq!(outcome.outcome, InvocationOutcome::Mismatch);
        assert_eq!(outcome.observed.as_deref(), Some("web_search"));
    }

    #[test]
    fn score_case_no_call_when_observed_is_none() {
        let c = case("a", "anything", "tool_search");
        let observed = ObservedFirstCall { tool_name: None };
        let outcome = score_case(&c, &observed);
        assert_eq!(outcome.outcome, InvocationOutcome::NoCall);
        assert!(outcome.observed.is_none());
    }

    #[test]
    fn metrics_count_each_quadrant() {
        let outcomes = vec![
            score_case(
                &case("a", "m", "tool_search"),
                &ObservedFirstCall {
                    tool_name: Some("tool_search".to_string()),
                },
            ),
            score_case(
                &case("b", "m", "tool_search"),
                &ObservedFirstCall {
                    tool_name: Some("web_search".to_string()),
                },
            ),
            score_case(
                &case("c", "m", "tool_search"),
                &ObservedFirstCall { tool_name: None },
            ),
        ];
        let metrics = ToolSearchMetrics::from_outcomes(&outcomes);
        assert_eq!(metrics.total, 3);
        assert_eq!(metrics.matches, 1);
        assert_eq!(metrics.mismatches, 1);
        assert_eq!(metrics.no_calls, 1);
        assert!((metrics.match_rate - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn metrics_empty_run_is_zero_not_vacuous_perfect() {
        let metrics = ToolSearchMetrics::from_outcomes(&[]);
        assert_eq!(metrics.total, 0);
        assert_eq!(metrics.match_rate, 0.0);
    }
}
