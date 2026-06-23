//! Evaluation of topic-drift detection (`TopicDetector::detect_shift`).
//!
//! # What the action decides
//!
//! Drift detection makes one binary decision and, when it fires, one
//! localisation: *did the conversation shift onto a new topic*, and if so *at
//! which message does the new topic begin*. It also emits a free-text label for
//! the new topic, which downstream becomes the recall query string.
//!
//! # The metrics we care about, and why
//!
//! The binary decision is a classification problem, so it is scored with a
//! confusion matrix and the usual derived rates. The two error modes are not
//! equally expensive, and the prompt already encodes the asymmetry: *"When only
//! a subtle shift happens, or you are unsure, prefer not to shift."* The system
//! is deliberately biased away from spurious splits.
//!
//! - A **false positive** is a spurious split: it fragments a coherent
//!   conversation and writes a noisy summary into the recall index. This is the
//!   expensive failure, so **precision** — of the splits we made, how many were
//!   right — is the headline guardrail.
//! - A **false negative** is a missed shift: the conversation bloats and mixes
//!   topics, degrading summary quality. We track **recall** to know what the
//!   precision bias is costing us. Note a malformed model response (an
//!   out-of-bounds index, a missing field) collapses to "no shift" at the trait
//!   boundary and therefore shows up here as a missed shift — which is honest,
//!   because a shift the system cannot act on is a shift it did not make. The
//!   adapter's own unit tests cover structural validity; this harness measures
//!   behaviour.
//!
//! Precision alone is not a sufficient gate: a detector that never fires scores
//! perfect precision. Gate on precision holding *and* recall not falling below
//! a floor, and read both against `boundary_total` to know they are not
//! vacuous.
//!
//! When the detector and the gold label agree a shift happened, we also score
//! **boundary placement**. An off-by-one boundary leaks the tail of the old
//! topic into the new conversation (or strands the head of the new one), which
//! hurts both summaries and the carried-forward overlap, so exact and
//! within-one placement are tracked separately.
//!
//! The **topic label** is fuzzy and feeds recall retrieval; scoring it well
//! needs semantic comparison (embedding similarity or a judge) and is deferred.
//! The expected and predicted labels are carried on each outcome so a future
//! scorer can grade them without re-running the action.
//!
//! Evaluation happens at the trait boundary: we run [`run_drift`] against any
//! [`TopicDetector`], map the returned message id back to its index, and score
//! the [`Option<TopicShift>`] the orchestrator would itself receive.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use brain::models::message::{Content, Message, MessageId, Role};
use brain::topic_detector::TopicDetector;

use crate::dataset::Dataset;

// ── Dataset case schema ──────────────────────────────────────────────────────

/// A conversation turn in a golden case, in a form convenient to author. It is
/// materialised into a real [`Message`] at run time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub role: TurnRole,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TurnRole {
    User,
    Assistant,
    System,
}

impl From<TurnRole> for Role {
    fn from(role: TurnRole) -> Self {
        match role {
            TurnRole::User => Role::User,
            TurnRole::Assistant => Role::Assistant,
            TurnRole::System => Role::System,
        }
    }
}

/// The expected outcome for a case.
///
/// When `shifted` is true, `at` is the 0-based index of the turn at which the
/// new topic begins and should be provided so boundary placement can be scored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftExpectation {
    pub shifted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topic: Option<String>,
}

/// One golden case for drift detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftCase {
    pub id: String,
    pub turns: Vec<Turn>,
    pub expect: DriftExpectation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl DriftCase {
    /// Materialises the authored turns into real messages, assigning ids so the
    /// returned shift can be mapped back to a turn index.
    pub fn messages(&self) -> Vec<Message> {
        self.turns
            .iter()
            .map(|turn| Message {
                id: MessageId::new(),
                role: turn.role.into(),
                content: vec![Content::Text {
                    text: turn.text.clone(),
                }],
                timestamp: jiff::Timestamp::now(),
            })
            .collect()
    }
}

// ── Scoring ──────────────────────────────────────────────────────────────────

/// A detected shift expressed as a turn index, after mapping the returned
/// message id back into the conversation.
#[derive(Debug, Clone)]
pub struct DetectedShift {
    pub at_index: usize,
    pub topic: String,
}

/// Where a single case landed on the confusion matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Confusion {
    TruePositive,
    FalsePositive,
    TrueNegative,
    FalseNegative,
}

/// Predicted versus expected boundary index, present only for true positives
/// whose gold label carried an expected index.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Boundary {
    pub expected: usize,
    pub predicted: usize,
}

impl Boundary {
    pub fn distance(&self) -> usize {
        self.expected.abs_diff(self.predicted)
    }

    pub fn is_exact(&self) -> bool {
        self.distance() == 0
    }

    pub fn within(&self, tolerance: usize) -> bool {
        self.distance() <= tolerance
    }
}

/// The graded result of a single case. A pure function of the case and the
/// detector's output — it records no model, threshold, or timing.
#[derive(Debug, Clone, Serialize)]
pub struct CaseOutcome {
    pub case_id: String,
    pub confusion: Confusion,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boundary: Option<Boundary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_topic: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_topic: Option<String>,
}

/// Grades a single case against the detector's output. Pure and deterministic.
pub fn score_case(case: &DriftCase, predicted: Option<&DetectedShift>) -> CaseOutcome {
    let expected = &case.expect;
    let (confusion, boundary) = match (expected.shifted, predicted) {
        (true, Some(shift)) => {
            let boundary = expected.at.map(|expected| Boundary {
                expected,
                predicted: shift.at_index,
            });
            (Confusion::TruePositive, boundary)
        }
        (true, None) => (Confusion::FalseNegative, None),
        (false, Some(_)) => (Confusion::FalsePositive, None),
        (false, None) => (Confusion::TrueNegative, None),
    };

    CaseOutcome {
        case_id: case.id.clone(),
        confusion,
        boundary,
        expected_topic: expected.topic.clone(),
        predicted_topic: predicted.map(|shift| shift.topic.clone()),
    }
}

// ── Aggregate metrics ────────────────────────────────────────────────────────

/// Aggregated drift-detection metrics over a run.
///
/// Rate conventions: `precision` and `recall` are defined as `1.0` when their
/// denominator is zero (no predicted positives / no actual positives), so a
/// dataset must contain shift cases for `recall` to mean anything — see the
/// module docs on why precision is not a sufficient gate on its own. The
/// boundary rates are `0.0` when `boundary_total` is zero; read that count to
/// tell a genuine zero from a vacuous one.
#[derive(Debug, Clone, Serialize)]
pub struct DriftMetrics {
    pub total: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub accuracy: f64,
    /// True positives that carried an expected boundary index.
    pub boundary_total: usize,
    pub boundary_exact: usize,
    pub boundary_within_one: usize,
    pub boundary_exact_rate: f64,
    pub mean_boundary_distance: f64,
}

impl DriftMetrics {
    pub fn from_outcomes(outcomes: &[CaseOutcome]) -> Self {
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut true_negatives = 0;
        let mut false_negatives = 0;

        let mut boundary_total = 0;
        let mut boundary_exact = 0;
        let mut boundary_within_one = 0;
        let mut distance_sum = 0;

        for outcome in outcomes {
            match outcome.confusion {
                Confusion::TruePositive => true_positives += 1,
                Confusion::FalsePositive => false_positives += 1,
                Confusion::TrueNegative => true_negatives += 1,
                Confusion::FalseNegative => false_negatives += 1,
            }

            if let Some(boundary) = outcome.boundary {
                boundary_total += 1;
                if boundary.is_exact() {
                    boundary_exact += 1;
                }
                if boundary.within(1) {
                    boundary_within_one += 1;
                }
                distance_sum += boundary.distance();
            }
        }

        let precision = rate(true_positives, true_positives + false_positives);
        let recall = rate(true_positives, true_positives + false_negatives);
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        let accuracy = if outcomes.is_empty() {
            1.0
        } else {
            (true_positives + true_negatives) as f64 / outcomes.len() as f64
        };

        let boundary_exact_rate = if boundary_total == 0 {
            0.0
        } else {
            boundary_exact as f64 / boundary_total as f64
        };
        let mean_boundary_distance = if boundary_total == 0 {
            0.0
        } else {
            distance_sum as f64 / boundary_total as f64
        };

        Self {
            total: outcomes.len(),
            true_positives,
            false_positives,
            true_negatives,
            false_negatives,
            precision,
            recall,
            f1,
            accuracy,
            boundary_total,
            boundary_exact,
            boundary_within_one,
            boundary_exact_rate,
            mean_boundary_distance,
        }
    }
}

/// A ratio that treats a zero denominator as a perfect (`1.0`) score — used for
/// precision and recall, where an empty denominator means "no opportunity to be
/// wrong". See [`DriftMetrics`] for why this is safe only alongside the counts.
fn rate(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f64 / denominator as f64
    }
}

// ── Runner ───────────────────────────────────────────────────────────────────

/// Runs every case in `dataset` through `detector` and grades each one.
///
/// The detector returns a shift keyed by message id; this maps that id back to
/// the turn index so it can be compared against the gold boundary. The result
/// is the scored outcomes, ready to aggregate with
/// [`DriftMetrics::from_outcomes`] and wrap in a [`crate::Report`].
pub async fn run_drift(
    detector: &dyn TopicDetector,
    dataset: &Dataset<DriftCase>,
) -> Result<Vec<CaseOutcome>> {
    let mut outcomes = Vec::with_capacity(dataset.cases.len());

    for case in &dataset.cases {
        let messages = case.messages();
        let shift = detector.detect_shift(&messages).await?;

        let detected = shift.and_then(|shift| {
            messages
                .iter()
                .position(|message| message.id == shift.at_message_id)
                .map(|at_index| DetectedShift {
                    at_index,
                    topic: shift.new_topic,
                })
        });

        outcomes.push(score_case(case, detected.as_ref()));
    }

    Ok(outcomes)
}

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use brain::topic_detector::TopicShift;

    fn case(id: &str, expect: DriftExpectation) -> DriftCase {
        DriftCase {
            id: id.to_string(),
            turns: vec![
                Turn {
                    role: TurnRole::User,
                    text: "first".to_string(),
                },
                Turn {
                    role: TurnRole::Assistant,
                    text: "second".to_string(),
                },
                Turn {
                    role: TurnRole::User,
                    text: "third".to_string(),
                },
            ],
            expect,
            notes: None,
        }
    }

    fn shift_at(index: usize) -> DetectedShift {
        DetectedShift {
            at_index: index,
            topic: "topic".to_string(),
        }
    }

    #[test]
    fn score_case_classifies_the_four_quadrants() {
        let shift_expect = DriftExpectation {
            shifted: true,
            at: Some(2),
            topic: None,
        };
        let no_shift_expect = DriftExpectation {
            shifted: false,
            at: None,
            topic: None,
        };

        let tp = score_case(&case("tp", shift_expect.clone()), Some(&shift_at(2)));
        let fn_ = score_case(&case("fn", shift_expect), None);
        let fp = score_case(&case("fp", no_shift_expect.clone()), Some(&shift_at(1)));
        let tn = score_case(&case("tn", no_shift_expect), None);

        assert_eq!(tp.confusion, Confusion::TruePositive);
        assert_eq!(fn_.confusion, Confusion::FalseNegative);
        assert_eq!(fp.confusion, Confusion::FalsePositive);
        assert_eq!(tn.confusion, Confusion::TrueNegative);

        let boundary = tp.boundary.expect("true positive carries a boundary");
        assert!(boundary.is_exact());
    }

    #[test]
    fn score_case_records_off_by_one_boundary() {
        let expect = DriftExpectation {
            shifted: true,
            at: Some(2),
            topic: None,
        };
        let outcome = score_case(&case("c", expect), Some(&shift_at(1)));
        let boundary = outcome.boundary.unwrap();

        assert!(!boundary.is_exact());
        assert!(boundary.within(1));
        assert_eq!(boundary.distance(), 1);
    }

    #[test]
    fn metrics_compute_precision_recall_and_boundary_rates() {
        let outcomes = vec![
            CaseOutcome {
                case_id: "a".into(),
                confusion: Confusion::TruePositive,
                boundary: Some(Boundary {
                    expected: 2,
                    predicted: 2,
                }),
                expected_topic: None,
                predicted_topic: None,
            },
            CaseOutcome {
                case_id: "b".into(),
                confusion: Confusion::TruePositive,
                boundary: Some(Boundary {
                    expected: 3,
                    predicted: 4,
                }),
                expected_topic: None,
                predicted_topic: None,
            },
            CaseOutcome {
                case_id: "c".into(),
                confusion: Confusion::FalsePositive,
                boundary: None,
                expected_topic: None,
                predicted_topic: None,
            },
            CaseOutcome {
                case_id: "d".into(),
                confusion: Confusion::FalseNegative,
                boundary: None,
                expected_topic: None,
                predicted_topic: None,
            },
            CaseOutcome {
                case_id: "e".into(),
                confusion: Confusion::TrueNegative,
                boundary: None,
                expected_topic: None,
                predicted_topic: None,
            },
        ];

        let metrics = DriftMetrics::from_outcomes(&outcomes);

        assert_eq!(metrics.total, 5);
        assert_eq!(metrics.true_positives, 2);
        assert_eq!(metrics.false_positives, 1);
        assert_eq!(metrics.false_negatives, 1);
        assert_eq!(metrics.true_negatives, 1);

        // precision = 2 / (2 + 1), recall = 2 / (2 + 1)
        assert!((metrics.precision - 2.0 / 3.0).abs() < 1e-9);
        assert!((metrics.recall - 2.0 / 3.0).abs() < 1e-9);
        assert!((metrics.f1 - 2.0 / 3.0).abs() < 1e-9);
        assert!((metrics.accuracy - 3.0 / 5.0).abs() < 1e-9);

        assert_eq!(metrics.boundary_total, 2);
        assert_eq!(metrics.boundary_exact, 1);
        assert_eq!(metrics.boundary_within_one, 2);
        assert!((metrics.boundary_exact_rate - 0.5).abs() < 1e-9);
        assert!((metrics.mean_boundary_distance - 0.5).abs() < 1e-9);
    }

    #[test]
    fn empty_run_does_not_divide_by_zero() {
        let metrics = DriftMetrics::from_outcomes(&[]);
        assert_eq!(metrics.total, 0);
        assert_eq!(metrics.precision, 1.0);
        assert_eq!(metrics.recall, 1.0);
        assert_eq!(metrics.accuracy, 1.0);
        assert_eq!(metrics.boundary_exact_rate, 0.0);
    }

    /// A detector that shifts on the final turn whenever its text contains the
    /// marker `NEW`, used to prove the runner's id-to-index mapping end to end.
    struct MarkerDetector;

    #[async_trait]
    impl TopicDetector for MarkerDetector {
        async fn detect_shift(&self, messages: &[Message]) -> Result<Option<TopicShift>> {
            let Some(last) = messages.last() else {
                return Ok(None);
            };
            let Content::Text { text } = &last.content[0] else {
                return Ok(None);
            };
            if text.contains("NEW") {
                Ok(Some(TopicShift {
                    at_message_id: last.id,
                    new_topic: "new topic".to_string(),
                }))
            } else {
                Ok(None)
            }
        }
    }

    #[tokio::test]
    async fn run_drift_maps_message_id_back_to_turn_index() {
        let dataset = Dataset {
            action: "detect_topic_shift".to_string(),
            version: "test".to_string(),
            cases: vec![
                DriftCase {
                    id: "shifts".into(),
                    turns: vec![
                        Turn {
                            role: TurnRole::User,
                            text: "talk about cats".into(),
                        },
                        Turn {
                            role: TurnRole::User,
                            text: "NEW: now taxes".into(),
                        },
                    ],
                    expect: DriftExpectation {
                        shifted: true,
                        at: Some(1),
                        topic: None,
                    },
                    notes: None,
                },
                DriftCase {
                    id: "stays".into(),
                    turns: vec![
                        Turn {
                            role: TurnRole::User,
                            text: "talk about cats".into(),
                        },
                        Turn {
                            role: TurnRole::User,
                            text: "more about cats".into(),
                        },
                    ],
                    expect: DriftExpectation {
                        shifted: false,
                        at: None,
                        topic: None,
                    },
                    notes: None,
                },
            ],
        };

        let outcomes = run_drift(&MarkerDetector, &dataset).await.unwrap();
        let metrics = DriftMetrics::from_outcomes(&outcomes);

        assert_eq!(metrics.true_positives, 1);
        assert_eq!(metrics.true_negatives, 1);
        assert_eq!(metrics.boundary_exact, 1);
        assert!(outcomes[0].boundary.unwrap().is_exact());
    }
}
