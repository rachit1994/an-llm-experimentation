//! testkit — deliberately-broken model/data constructors for the
//! negative-control battery (T0.8, VERIFICATION.md §5 / tests/README.md H2).
//!
//! Phase 0 has no model yet (that's Phase 1's next-pattern predictor), so
//! these are generic, minimal, well-documented utilities that Phase 1+'s
//! `nc_*` tests plug into a real encoder/loss. The KAT-level negative
//! controls this phase actually exercises (`nc_born_sum`, `nc_unitary_norm`
//! in `tests/negative_controls.rs`) just re-run the qilm-core kernels
//! directly and don't need these helpers.

use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// A "collapsed" encoder: ignores its input and always returns the same
/// fixed output vector. This is the H2 harness's `constant_encoder()` —
/// Phase 1+'s `nc_collapse_canary` / `nc_collapsed_invariance` controls plug
/// this in wherever a real encoder is expected, to prove the collapse metric
/// actually fires (erank ~= 1, meanstd ~= 0) on a model that has collapsed.
pub fn constant_encoder(output: Vec<f64>) -> impl Fn(&[f64]) -> Vec<f64> {
    move |_input: &[f64]| output.clone()
}

/// A deterministic, seeded "random-init" parameter vector — a stand-in for
/// "a freshly initialized, untrained model" until Phase 1's real model
/// exists. Used by `nc_untrained_bpb` (VERIFICATION.md §5): evaluating a
/// random-init model should give BPB >= 7.5 (near log2(256) = 8), proving BPB
/// isn't accidentally measuring something else.
pub fn random_init_model(seed: u64, num_params: usize) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..num_params)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect()
}

/// Deterministically permute a slice of labels using a seeded RNG — the
/// `shuffle_labels()` used by `nc_shuffled_labels` (train on permuted
/// labels; correct code should then score at chance, proving no label
/// leakage). A true permutation (Fisher-Yates via `rand::seq::SliceRandom`),
/// not an independent resample, so every label from the input appears
/// exactly once in the output.
pub fn shuffle_labels<T: Clone>(labels: &[T], seed: u64) -> Vec<T> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = labels.to_vec();
    out.shuffle(&mut rng);
    out
}

/// Remove a named term from a set of active loss-term names — the
/// `strip_term(name)` ablation used by `nc_invariance_needs_Linv` (train
/// without `L_inv`) and similar Phase 1+ ablations. Phase 0 has no loss
/// composition yet, so this operates generically on term names; Phase 1's
/// training loop is expected to consult the resulting list to decide which
/// loss terms to include.
pub fn strip_term(active_terms: &[&str], name: &str) -> Vec<String> {
    active_terms
        .iter()
        .filter(|&&t| t != name)
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_encoder_ignores_input() {
        let value = vec![1.0, 2.0, 3.0];
        let enc = constant_encoder(value.clone());
        assert_eq!(enc(&[0.0]), value);
        assert_eq!(enc(&[9.0, 9.0, 9.0, 9.0]), value);
    }

    #[test]
    fn random_init_model_is_deterministic_given_seed() {
        let a = random_init_model(7, 16);
        let b = random_init_model(7, 16);
        assert_eq!(
            a, b,
            "random_init_model must be deterministic for a fixed seed (C4)"
        );
        let c = random_init_model(8, 16);
        assert_ne!(
            a, c,
            "different seeds should (almost surely) give different params"
        );
    }

    #[test]
    fn shuffle_labels_is_a_permutation_not_a_resample() {
        let labels = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let shuffled = shuffle_labels(&labels, 3);
        let mut sorted = shuffled.clone();
        sorted.sort();
        assert_eq!(sorted, labels);
    }

    #[test]
    fn strip_term_removes_only_the_named_term() {
        let terms = ["L_pattern", "L_inv", "anti_collapse"];
        let stripped = strip_term(&terms, "L_inv");
        assert_eq!(
            stripped,
            vec!["L_pattern".to_string(), "anti_collapse".to_string()]
        );
    }
}
