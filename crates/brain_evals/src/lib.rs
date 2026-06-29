//! Offline evaluation harness for the brain's AI-backed invocation actions.
//!
//! Every AI behaviour in the system is a trait in `brain` (`TopicDetector`,
//! `Summarizer`, `Embedder`, `LanguageModel`) implemented by an adapter and
//! backed by a prompt template. This crate measures those behaviours at the
//! trait boundary — the same boundary `Brain` consumes — so an evaluation is
//! an equal-citizen client of the action, not a backdoor into the adapter.
//!
//! # The four-part loop
//!
//! Following current practice, an evaluation is built from four separable
//! pieces, each with one job:
//!
//! - a **dataset** of versioned, representative inputs paired with an expected
//!   result or rubric ([`Dataset`]);
//! - a **scorer** that grades a single output into a per-case outcome — pure,
//!   deterministic, and independent of how the output was produced;
//! - a **runner** that feeds the dataset through an action and collects the
//!   per-case outcomes;
//! - a **report** that aggregates outcomes into metrics and stamps the run with
//!   the dataset version, model id and prompt hash so a score is reproducible
//!   and a regression is attributable ([`Report`], [`RunMetadata`]).
//!
//! Dashboards, drift monitoring and observability platforms are deliberately
//! out of scope: they are built on top of this foundation, not in place of it.
//!
//! # What "good" means per action
//!
//! The actions differ in output shape, so they differ in how they are scored.
//! Drift detection ([`drift`]) emits a structured decision and is graded
//! deterministically. Embedding is graded on relative ranking. Summarisation
//! and chat are open-ended and will need a pinned judge; they are deferred.
//!
//! Scorers are pure functions of an output. Model and threshold — the things
//! that vary — live in [`RunMetadata`] and dataset files, never in the scorer.

pub mod dataset;
pub mod drift;
pub mod report;
pub mod tool_search;

pub use dataset::Dataset;
pub use report::{Report, RunMetadata};
