//! T0.6 — Modern-Hopfield retrieval KAT + permutation-invariance property.
//!
//! retrieve(X, xi, beta) = X * softmax(beta * X^T xi)  (dense modern-Hopfield
//! attractor readout, Ramsauer et al. 2020).

use proptest::prelude::*;
use qilm_core::hopfield::retrieve;
use rand::{seq::SliceRandom, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn random_pattern(rng: &mut ChaCha8Rng, d: usize) -> Vec<f64> {
    (0..d).map(|_| rng.random_range(-1.0..1.0)).collect()
}

/// kat_hopfield_recall — store 8 random patterns, query with a stored pattern
/// plus small noise; the retrieved vector's cosine similarity to the
/// original stored pattern must be >= 0.99.
///
/// beta is picked well above the 1/sqrt(d) "flat" default (which is a
/// stability floor, not a retrieval-sharpness target): a modern-Hopfield
/// network needs its inverse-temperature large enough that softmax(beta *
/// scores) concentrates on the best-matching stored pattern for recall to be
/// sharp — beta=8 at d=128 (score magnitudes O(1) after the dot product with
/// a unit-ish-scale noisy query) is well inside the regime literature (e.g.
/// Ramsauer et al. 2020) uses for high-capacity retrieval.
#[test]
fn kat_hopfield_recall() {
    let mut rng = ChaCha8Rng::seed_from_u64(4242);
    let d = 128;
    let n = 8;
    let beta = 8.0;

    let patterns: Vec<Vec<f64>> = (0..n).map(|_| random_pattern(&mut rng, d)).collect();

    let target_idx = 3;
    let noise_scale = 0.05;
    let query: Vec<f64> = patterns[target_idx]
        .iter()
        .map(|v| v + noise_scale * rng.random_range(-1.0..1.0))
        .collect();

    let retrieved = retrieve(&patterns, &query, beta);
    let cos = cosine(&retrieved, &patterns[target_idx]);
    assert!(
        cos >= 0.99,
        "cosine(retrieved, stored) = {cos}, need >= 0.99"
    );
}

/// kat_hopfield_recall_can_say_no (anti-vacuity canary) — the >= 0.99 bar is
/// not clearable by construction: with beta -> 0, `retrieve` degenerates to
/// an (almost) uniform average over ALL n stored patterns regardless of the
/// query (softmax(beta * scores) -> uniform as beta -> 0), which is diluted
/// by the other n-1 unrelated random patterns and should NOT recover the
/// target with high cosine. Same patterns/query/seed as `kat_hopfield_recall`
/// -- only beta changes -- proving the sharp-softmax mechanism (not the KAT's
/// threshold) is what makes recall succeed.
#[test]
fn kat_hopfield_recall_can_say_no() {
    let mut rng = ChaCha8Rng::seed_from_u64(4242);
    let d = 128;
    let n = 8;
    let near_zero_beta = 1e-6;

    let patterns: Vec<Vec<f64>> = (0..n).map(|_| random_pattern(&mut rng, d)).collect();

    let target_idx = 3;
    let noise_scale = 0.05;
    let query: Vec<f64> = patterns[target_idx]
        .iter()
        .map(|v| v + noise_scale * rng.random_range(-1.0..1.0))
        .collect();

    let retrieved = retrieve(&patterns, &query, near_zero_beta);
    let cos = cosine(&retrieved, &patterns[target_idx]);
    assert!(
        cos < 0.99,
        "beta -> 0 should degrade recall (got cosine {cos} >= 0.99) -- if this \
         passes, kat_hopfield_recall's threshold can't tell sharp retrieval from \
         a uniform blend of unrelated patterns"
    );
}

/// prop_hopfield_perm — permuting the stored pattern set does not change the
/// retrieval output for a clean (unpermuted) query: retrieve() is a weighted
/// sum over the pattern set, and both the weights (softmax of scores) and
/// the weighted sum are invariant to the order patterns are supplied in.
#[test]
fn prop_hopfield_perm_smoke() {
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let d = 32;
    let n = 8;
    let beta = 4.0;
    let patterns: Vec<Vec<f64>> = (0..n).map(|_| random_pattern(&mut rng, d)).collect();
    let query = patterns[2].clone();

    let out_original = retrieve(&patterns, &query, beta);

    let mut permuted = patterns.clone();
    permuted.shuffle(&mut rng);
    let out_permuted = retrieve(&permuted, &query, beta);

    for i in 0..d {
        assert!(
            (out_original[i] - out_permuted[i]).abs() < 1e-9,
            "index {i}: original={} permuted={}",
            out_original[i],
            out_permuted[i]
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    /// prop_hopfield_perm — the property test proper: over 100 random
    /// pattern sets + permutations, retrieval is permutation-invariant.
    #[test]
    fn prop_hopfield_perm(
        pattern_entries in prop::collection::vec(-1.0f64..1.0f64, 8 * 6),
        perm_seed in any::<u64>(),
    ) {
        let d = 6;
        let n = 8;
        let patterns: Vec<Vec<f64>> = pattern_entries.chunks(d).map(|c| c.to_vec()).collect();
        let query = patterns[0].clone();
        let beta = 2.0;

        let out_original = retrieve(&patterns, &query, beta);

        let mut rng = ChaCha8Rng::seed_from_u64(perm_seed);
        let mut permuted = patterns.clone();
        permuted.shuffle(&mut rng);
        prop_assert_eq!(permuted.len(), n);
        let out_permuted = retrieve(&permuted, &query, beta);

        for i in 0..d {
            prop_assert!((out_original[i] - out_permuted[i]).abs() < 1e-9);
        }
    }
}
