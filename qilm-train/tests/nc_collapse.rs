//! nc_collapse — the Phase-1 anti-collapse negative control (T1.3). A CONSTANT
//! encoder (the canonical collapsed model) must drive the collapse metric to
//! ~0 relative to the targets AND make the H3 gate FAIL. If this ever passed,
//! the metric would be blind — the exact scar AGENTS.md was written to prevent
//! (VERIFICATION.md §5). This is the metric→gate wiring's anti-vacuity canary.

use qilm_train::gate::{evaluate, load_gates, GateOutcome};
use qilm_train::metrics::collapse::collapse_ratios;
use qilm_train::testkit::constant_encoder;
use serde_json::json;
use std::path::PathBuf;

fn gates() -> qilm_train::gate::GatesConfig {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    load_gates(&root.join("gates.toml")).expect("load frozen gates.toml")
}

#[test]
fn nc_collapse_canary() {
    let d = 32; // realistic representation dim
    let n = 16;

    // Z from a CONSTANT encoder: every input maps to the same d-vector, so all
    // n rows are identical (rank 1, zero per-dim variance).
    let enc = constant_encoder(vec![0.42; d]);
    let mut z = Vec::with_capacity(n * d);
    for i in 0..n {
        // Feed genuinely different inputs; a constant encoder ignores them.
        let input = vec![i as f64, (i as f64) * 2.0 - 1.0];
        z.extend_from_slice(&enc(&input));
    }

    // Z* = INDEPENDENT full-spread reference (I_d): erank d, nonzero meanstd.
    // (Rule 1: the expectation is the targets' own spread, not Z's.)
    let mut z_star = vec![0.0; d * d];
    for i in 0..d {
        z_star[i * d + i] = 1.0;
    }

    let (erank_ratio, meanstd_ratio) = collapse_ratios(&z, n, &z_star, d, d);

    // The collapsed encoder is caught by the metric itself...
    assert!(
        erank_ratio <= 0.05,
        "constant encoder erank_ratio must be ≤ 0.05, got {erank_ratio}"
    );
    assert!(
        meanstd_ratio.abs() < 1e-9,
        "constant encoder meanstd_ratio must be 0, got {meanstd_ratio}"
    );

    // ...AND the H3 gate must FAIL on this metrics blob (both ratios < 0.50).
    let metrics = json!({ "erank_ratio": erank_ratio, "meanstd_ratio": meanstd_ratio });
    match evaluate("collapse", &gates(), &metrics).unwrap() {
        GateOutcome::Fail { .. } => {} // correct: collapse is caught
        GateOutcome::Pass {
            measured,
            threshold,
        } => panic!(
            "BLIND METRIC: constant encoder PASSED the H3 gate (measured {measured}, \
             threshold {threshold}) — this is the scar; fix the metric, not the test"
        ),
    }
}

/// Positive control: a genuinely full-spread encoder (rows = distinct spread-out
/// vectors matching the target's scale) must PASS the H3 gate, proving the gate
/// is not just always-FAIL. Without this, `nc_collapse_canary` alone could be
/// satisfied by a gate that rejects everything.
#[test]
fn nc_collapse_healthy_encoder_passes() {
    let d = 32;
    let n = 32;
    // Z = I_d rows scaled — same spread as the target ⇒ ratios ≈ 1.
    let mut z = vec![0.0; n * d];
    for i in 0..n {
        z[i * d + (i % d)] = 1.0;
    }
    let mut z_star = vec![0.0; d * d];
    for i in 0..d {
        z_star[i * d + i] = 1.0;
    }
    let (erank_ratio, meanstd_ratio) = collapse_ratios(&z, n, &z_star, d, d);
    let metrics = json!({ "erank_ratio": erank_ratio, "meanstd_ratio": meanstd_ratio });
    assert!(
        matches!(
            evaluate("collapse", &gates(), &metrics).unwrap(),
            GateOutcome::Pass { .. }
        ),
        "healthy full-spread encoder must PASS H3 (erank_ratio {erank_ratio}, \
         meanstd_ratio {meanstd_ratio})"
    );
}
