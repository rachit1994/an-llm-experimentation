//! negative_controls.rs — the nc_* battery (T0.8, VERIFICATION.md §5).
//!
//! Phase 0 only wires the harness itself; the two controls it can actually
//! run now are the KAT-level identities (`nc_born_sum`, `nc_unitary_norm`),
//! re-exercised here (independent of qilm-oracle's copies) to prove the
//! negative-control file itself reaches the real qilm-core kernels. The
//! model-level controls (nc_collapse_canary, nc_shuffled_labels, ...) need a
//! real model/metrics module that doesn't exist until Phase 1+; per T0.8's
//! note, this file authors the deliberately-broken-model helpers
//! (`qilm_train::testkit`) now and exercises what's testable of them today
//! (that they behave as specified) without fabricating a Phase 1 metric.

use qilm_core::born::born;
use qilm_core::complex::Complex;
use qilm_core::unitary::{cayley, givens, skew_hermitian_part, ComplexMat};
use qilm_train::testkit::{constant_encoder, random_init_model, shuffle_labels, strip_term};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// nc_born_sum — Born-rule probabilities sum to 1 within 1e-6 (VERIFICATION
/// §5's "L0 identities, also run here" row; also directly kills the
/// "remove Born normalization" mutant from the §6 catalog).
#[test]
fn nc_born_sum() {
    let mut rng = ChaCha8Rng::seed_from_u64(2001);
    let d = 12;
    let psi = Complex::new(
        (0..d).map(|_| rng.random_range(-1.0..1.0)).collect(),
        (0..d).map(|_| rng.random_range(-1.0..1.0)).collect(),
    );
    let measurements: Vec<Complex> = (0..64)
        .map(|_| {
            Complex::new(
                (0..d).map(|_| rng.random_range(-1.0..1.0)).collect(),
                (0..d).map(|_| rng.random_range(-1.0..1.0)).collect(),
            )
        })
        .collect();

    let probs = born(&psi, &measurements);
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Born probabilities must sum to 1, got {sum}"
    );
}

/// nc_unitary_norm — ||U psi|| == ||psi|| within 1e-6, for both Cayley and
/// Givens (VERIFICATION §5's "L0 identities, also run here" row).
#[test]
fn nc_unitary_norm() {
    let mut rng = ChaCha8Rng::seed_from_u64(2002);
    let d = 5;

    let mut raw = ComplexMat::zeros(d);
    for i in 0..d {
        for j in 0..d {
            raw.set(
                i,
                j,
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            );
        }
    }
    let u_cayley = cayley(&skew_hermitian_part(&raw));
    let angles: Vec<f64> = (0..d - 1)
        .map(|_| rng.random_range(0.0..std::f64::consts::TAU))
        .collect();
    let u_givens = givens(&angles, d);

    for u in [&u_cayley, &u_givens] {
        let psi = Complex::new(
            (0..d).map(|_| rng.random_range(-1.0..1.0)).collect(),
            (0..d).map(|_| rng.random_range(-1.0..1.0)).collect(),
        );
        let before = psi.norm();
        let after = u.apply(&psi).norm();
        assert!(
            (before - after).abs() < 1e-6,
            "unitary must preserve norm: {before} vs {after}"
        );
    }
}

/// The "metric side" of nc_collapse_canary, testable now per T0.8's note: a
/// constant_encoder() genuinely produces zero per-dimension spread across a
/// batch of distinct inputs -- the collapse signature Phase 1's real
/// erank/meanstd metric will assert against. This is NOT the real collapse
/// gate (that needs Phase 1's metrics module); it only proves the fixture
/// (constant_encoder) is a genuine collapse fixture for that metric to fire on.
#[test]
fn nc_collapse_canary_fixture_has_zero_spread() {
    let value = vec![0.3, -0.1, 0.7];
    let enc = constant_encoder(value.clone());
    let outputs: Vec<Vec<f64>> = (0..10).map(|i| enc(&[i as f64; 3])).collect();

    for out in &outputs {
        assert_eq!(out, &value, "constant_encoder must ignore its input");
    }
    for j in 0..value.len() {
        let mean: f64 = outputs.iter().map(|o| o[j]).sum::<f64>() / outputs.len() as f64;
        let var: f64 =
            outputs.iter().map(|o| (o[j] - mean).powi(2)).sum::<f64>() / outputs.len() as f64;
        assert!(
            var.abs() < 1e-12,
            "constant_encoder output has nonzero variance in dim {j}: {var}"
        );
    }
}

/// testkit smoke coverage for the remaining T0.8 helpers, exercised from the
/// negative-control file itself (not just testkit.rs's own unit tests) so a
/// future Phase-1 nc_* author can see exactly how they're expected to be used.
#[test]
fn testkit_helpers_are_wired_correctly() {
    let a = random_init_model(42, 8);
    let b = random_init_model(42, 8);
    assert_eq!(a, b, "random_init_model must be seed-deterministic (C4)");

    let labels = vec!["cat", "dog", "cat", "bird"];
    let shuffled = shuffle_labels(&labels, 5);
    let mut sorted_shuffled = shuffled.clone();
    sorted_shuffled.sort();
    let mut sorted_labels = labels.clone();
    sorted_labels.sort();
    assert_eq!(
        sorted_shuffled, sorted_labels,
        "shuffle_labels must be a permutation"
    );

    let terms = ["L_pattern", "L_inv", "anti_collapse", "L_byte_ce"];
    let stripped = strip_term(&terms, "L_inv");
    assert!(!stripped.contains(&"L_inv".to_string()));
    assert_eq!(stripped.len(), terms.len() - 1);
}
