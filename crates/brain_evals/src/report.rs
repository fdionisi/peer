//! Run reports: metrics stamped with the provenance needed to attribute a
//! regression.
//!
//! A [`Report`] pairs a metrics payload with the metadata that makes a score
//! reproducible — which dataset version ran, against which model, with which
//! prompt. Pinning the prompt hash and model id is what lets a later run say
//! "this regressed when the template changed" rather than "the model drifted
//! under us". The metadata is supplied by the caller because the model id and
//! prompt hash are properties of the adapter, kept out of this crate.

use jiff::Timestamp;
use serde::Serialize;

/// Provenance for a single evaluation run.
#[derive(Debug, Clone, Serialize)]
pub struct RunMetadata {
    /// The action evaluated, e.g. `detect_topic_shift`.
    pub action: String,
    /// The dataset version the cases came from, e.g. `v1`.
    pub dataset_version: String,
    /// The model id the action ran against, e.g. `mistral-small-latest`.
    pub model: String,
    /// A hash of the prompt template(s) the action used, so a template change
    /// is distinguishable from a model change.
    pub prompt_hash: String,
    /// When the run happened.
    pub ran_at: Timestamp,
}

impl RunMetadata {
    pub fn new(
        action: impl Into<String>,
        dataset_version: impl Into<String>,
        model: impl Into<String>,
        prompt_hash: impl Into<String>,
    ) -> Self {
        Self {
            action: action.into(),
            dataset_version: dataset_version.into(),
            model: model.into(),
            prompt_hash: prompt_hash.into(),
            ran_at: Timestamp::now(),
        }
    }
}

/// A scored run: provenance plus an action-specific metrics payload.
///
/// Generic over the metrics type so each action reports the metrics that fit
/// its output shape while sharing one snapshot format.
#[derive(Debug, Clone, Serialize)]
pub struct Report<M> {
    #[serde(flatten)]
    pub metadata: RunMetadata,
    pub metrics: M,
}

impl<M> Report<M> {
    pub fn new(metadata: RunMetadata, metrics: M) -> Self {
        Self { metadata, metrics }
    }
}
