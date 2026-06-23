//! Live evaluation of topic-drift detection against the real Mistral endpoint.
//!
//! This is the actual evaluation, not a format check: it runs the golden set
//! through a real `MistralClient` and prints a scored [`Report`]. It is
//! `#[ignore]`d so the ordinary offline `cargo test` stays fast and free.
//!
//! Run with:
//!   MISTRAL_API_KEY=<key> cargo test -p brain_evals --test live_drift -- --ignored --nocapture
//!
//! The model defaults to `mistral-small-latest` and can be overridden with
//! `MISTRAL_MODEL` to match whatever production runs.

use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;

use brain::prompts::PromptRegistry;
use brain_evals::drift::{CaseOutcome, DriftCase, DriftMetrics};
use brain_evals::{Dataset, Report, RunMetadata};
use brain_mistralai::{MistralClient, MistralConfig};
use brain_prompts_embedded::EmbeddedPromptRegistry;

#[tokio::test]
#[ignore]
async fn detect_topic_shift_v1_live() {
    let api_key =
        std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY must be set to run live evals");
    let model =
        std::env::var("MISTRAL_MODEL").unwrap_or_else(|_| "mistral-small-latest".to_string());

    let prompts = Arc::new(EmbeddedPromptRegistry::new());
    let prompt_hash = template_hash(prompts.as_ref(), "detect_topic_shift");

    let client = MistralClient::new(MistralConfig::new(api_key, model.clone()), prompts)
        .expect("client construction failed");

    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join("detect_topic_shift.v1.jsonl");
    let dataset: Dataset<DriftCase> = Dataset::load(&path).expect("dataset should load");

    let outcomes = brain_evals::drift::run_drift(&client, &dataset)
        .await
        .expect("drift run failed");
    let metrics = DriftMetrics::from_outcomes(&outcomes);

    let report = Report::new(
        RunMetadata::new("detect_topic_shift", &dataset.version, &model, &prompt_hash),
        &metrics,
    );

    print_outcomes(&dataset, &outcomes);
    println!(
        "\n{}",
        serde_json::to_string_pretty(&report).expect("report serialises")
    );

    // A run that produced no outcomes means nothing was evaluated.
    assert_eq!(metrics.total, dataset.len());
}

fn template_hash(registry: &dyn PromptRegistry, name: &str) -> String {
    let template = registry
        .get(name)
        .unwrap_or_else(|| panic!("missing prompt template '{name}'"));
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    template.template.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Prints a per-case line so a disagreement is inspectable, not just a number.
fn print_outcomes(dataset: &Dataset<DriftCase>, outcomes: &[CaseOutcome]) {
    println!("per-case outcomes ({} cases):", outcomes.len());
    for (case, outcome) in dataset.cases.iter().zip(outcomes) {
        let boundary = match outcome.boundary {
            Some(b) => format!(
                " boundary expected={} predicted={}",
                b.expected, b.predicted
            ),
            None => String::new(),
        };
        let predicted = match &outcome.predicted_topic {
            Some(topic) => format!(" predicted_topic={topic:?}"),
            None => String::new(),
        };
        println!(
            "  {:<13?} {}{}{}",
            outcome.confusion, case.id, boundary, predicted
        );
    }
}
