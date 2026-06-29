//! Guards that the shipped golden datasets parse and that each case is
//! internally consistent (a shift carries a boundary index within range). This
//! is a format regression test, not a model evaluation — it never calls a
//! model.

use std::path::Path;

use brain_evals::Dataset;
use brain_evals::drift::DriftCase;
use brain_evals::tool_search::ToolSearchCase;

fn dataset_path(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join(name)
}

#[test]
fn detect_topic_shift_v1_parses_and_is_consistent() {
    let dataset: Dataset<DriftCase> =
        Dataset::load(dataset_path("detect_topic_shift.v1.jsonl")).expect("dataset should parse");

    assert_eq!(dataset.action, "detect_topic_shift");
    assert_eq!(dataset.version, "v1");
    assert!(
        dataset.len() >= 10,
        "expected the bootstrapped set, got {}",
        dataset.len()
    );

    let shifts = dataset
        .cases
        .iter()
        .filter(|case| case.expect.shifted)
        .count();
    let stays = dataset.len() - shifts;
    assert!(shifts > 0, "dataset must contain shift cases");
    assert!(stays > 0, "dataset must contain no-shift cases");

    for case in &dataset.cases {
        if case.expect.shifted {
            let at = case
                .expect
                .at
                .unwrap_or_else(|| panic!("case '{}' shifts but has no boundary index", case.id));
            assert!(
                at < case.turns.len(),
                "case '{}' boundary {at} is out of range for {} turns",
                case.id,
                case.turns.len()
            );
        }
    }
}

#[test]
fn tool_search_invocation_v1_parses_and_is_consistent() {
    let dataset: Dataset<ToolSearchCase> =
        Dataset::load(dataset_path("tool_search_invocation.v1.jsonl"))
            .expect("dataset should parse");

    assert_eq!(dataset.action, "tool_search_invocation");
    assert_eq!(dataset.version, "v1");
    assert!(!dataset.is_empty(), "dataset must not be empty");

    for case in &dataset.cases {
        assert!(
            !case.expect.first_tool_call.is_empty(),
            "case '{}' has no expected first_tool_call",
            case.id
        );
        assert!(
            !case.user_message.is_empty(),
            "case '{}' has no user_message",
            case.id
        );
    }

    let positives = dataset
        .cases
        .iter()
        .filter(|c| c.expect.first_tool_call == "tool_search")
        .count();
    let negatives = dataset.len() - positives;
    assert!(
        positives > 0,
        "dataset must contain tool_search-positive cases"
    );
    assert!(
        negatives > 0,
        "dataset must contain tool_search-negative cases"
    );
}
