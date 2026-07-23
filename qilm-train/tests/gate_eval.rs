//! gate_eval — unit tests for the H3 (collapse) and H2 (g0) gate arms wired in
//! Phase 1. Thresholds come from the FROZEN gates.toml (erank_ratio_min = 0.50,
//! meanstd_ratio_min = 0.50, bpb_ratio_max = 1.10 — METRICS-AND-GATES.md §7),
//! never hand-typed here; the tests read whatever the committed file says and
//! assert the PASS/FAIL logic, so they can't drift from the frozen numbers.

use qilm_train::gate::{evaluate, load_gates, GateError, GateOutcome, GatesConfig};
use serde_json::json;
use std::path::PathBuf;

fn gates() -> GatesConfig {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    load_gates(&root.join("gates.toml")).expect("load frozen gates.toml")
}

#[test]
fn collapse_passes_when_both_ratios_clear_mins() {
    let g = gates();
    let m = json!({ "erank_ratio": 0.80, "meanstd_ratio": 0.75 });
    assert!(matches!(
        evaluate("collapse", &g, &m).unwrap(),
        GateOutcome::Pass { .. }
    ));
}

#[test]
fn collapse_fails_when_erank_ratio_below_min() {
    let g = gates();
    // erank below its min, meanstd fine → FAIL, and the binding (reported)
    // sub-metric must be the erank ratio (the one that's under).
    let m = json!({ "erank_ratio": 0.10, "meanstd_ratio": 0.90 });
    match evaluate("collapse", &g, &m).unwrap() {
        GateOutcome::Fail { measured, .. } => {
            assert!(
                (measured - 0.10).abs() < 1e-12,
                "binding sub-metric should be the failing erank_ratio, got {measured}"
            );
        }
        other => panic!("expected Fail, got {other:?}"),
    }
}

#[test]
fn collapse_fails_when_meanstd_ratio_below_min() {
    let g = gates();
    let m = json!({ "erank_ratio": 0.95, "meanstd_ratio": 0.05 });
    match evaluate("collapse", &g, &m).unwrap() {
        GateOutcome::Fail { measured, .. } => {
            assert!(
                (measured - 0.05).abs() < 1e-12,
                "binding sub-metric should be the failing meanstd_ratio, got {measured}"
            );
        }
        other => panic!("expected Fail, got {other:?}"),
    }
}

#[test]
fn collapse_missing_field_errors_not_fabricated() {
    let g = gates();
    let m = json!({ "erank_ratio": 0.80 }); // meanstd_ratio absent
    assert!(matches!(
        evaluate("collapse", &g, &m),
        Err(GateError::MissingMetric(_))
    ));
}

#[test]
fn g0_passes_and_fails_around_bpb_ratio_max() {
    let g = gates();
    let max = g.g0.bpb_ratio_max;
    // just under the max → PASS; just over → FAIL.
    assert!(matches!(
        evaluate("g0", &g, &json!({ "bpb_ratio": max - 0.05 })).unwrap(),
        GateOutcome::Pass { .. }
    ));
    assert!(matches!(
        evaluate("g0", &g, &json!({ "bpb_ratio": max + 0.05 })).unwrap(),
        GateOutcome::Fail { .. }
    ));
}

#[test]
fn g0_missing_field_errors() {
    let g = gates();
    assert!(matches!(
        evaluate("g0", &g, &json!({})),
        Err(GateError::MissingMetric(_))
    ));
}
