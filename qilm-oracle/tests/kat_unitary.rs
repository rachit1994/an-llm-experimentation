//! T0.5 — Unitary layer (Cayley + Givens) KATs and property test.

use qilm_core::complex::Complex;
use qilm_core::unitary::{cayley, givens, skew_hermitian_part, ComplexMat};
use proptest::prelude::*;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn random_complex_mat(rng: &mut ChaCha8Rng, d: usize) -> ComplexMat {
    let mut m = ComplexMat::zeros(d);
    for i in 0..d {
        for j in 0..d {
            m.set(i, j, rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
        }
    }
    m
}

fn random_complex_vec(rng: &mut ChaCha8Rng, d: usize) -> Complex {
    let re = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
    let im = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
    Complex::new(re, im)
}

fn max_abs_diag_off_identity(m: &ComplexMat) -> f64 {
    let d = m.d;
    let mut max_err: f64 = 0.0;
    for i in 0..d {
        for j in 0..d {
            let (re, im) = m.get(i, j);
            let expected_re = if i == j { 1.0 } else { 0.0 };
            let err = ((re - expected_re).powi(2) + im.powi(2)).sqrt();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

/// kat_cayley_unitary — U^†U == I within 1e-6, for a random skew-Hermitian
/// generator built via the required projection (raw -> (raw - raw^†)/2).
#[test]
fn kat_cayley_unitary() {
    let mut rng = ChaCha8Rng::seed_from_u64(2024);
    let d = 5;
    let raw = random_complex_mat(&mut rng, d);
    let a_skew = skew_hermitian_part(&raw);
    let u = cayley(&a_skew);

    let udag_u = u.conj_transpose().matmul(&u);
    let err = max_abs_diag_off_identity(&udag_u);
    assert!(err < 1e-6, "U^†U deviates from I by {err}, need < 1e-6");
}

/// Kills-mutant: "Cayley without the skew-Hermitian constraint". Feeding a
/// generic (non-skew-Hermitian) raw matrix directly into `cayley` (skipping
/// `skew_hermitian_part`) must NOT generally be unitary — this proves the
/// projection step is load-bearing, so a mutant that deletes/breaks it is
/// caught by `kat_cayley_unitary` (which always goes through the projection).
#[test]
fn kat_cayley_needs_skew_hermitian_constraint() {
    let mut rng = ChaCha8Rng::seed_from_u64(77);
    let d = 5;
    let raw = random_complex_mat(&mut rng, d);
    // Deliberately skip skew_hermitian_part.
    let u_bad = cayley(&raw);
    let udag_u = u_bad.conj_transpose().matmul(&u_bad);
    let err = max_abs_diag_off_identity(&udag_u);
    assert!(
        err > 1e-3,
        "cayley() on a non-skew-Hermitian matrix should NOT be unitary (err={err}); \
         if this passes, kat_cayley_unitary can't distinguish a correct implementation \
         from one that dropped the skew-Hermitian projection"
    );
}

/// kat_unitary_norm — ||U psi|| == ||psi|| within 1e-6, for BOTH
/// parameterizations (Cayley and Givens).
#[test]
fn kat_unitary_norm() {
    let mut rng = ChaCha8Rng::seed_from_u64(555);
    let d = 6;

    let raw = random_complex_mat(&mut rng, d);
    let u_cayley = cayley(&skew_hermitian_part(&raw));

    let angles: Vec<f64> = (0..d - 1).map(|_| rng.random_range(0.0..std::f64::consts::TAU)).collect();
    let u_givens = givens(&angles, d);

    for (name, u) in [("cayley", &u_cayley), ("givens", &u_givens)] {
        let psi = random_complex_vec(&mut rng, d);
        let norm_before = psi.norm();
        let out = u.apply(&psi);
        let norm_after = out.norm();
        assert!(
            (norm_before - norm_after).abs() < 1e-6,
            "{name}: ||psi||={norm_before}, ||U psi||={norm_after}, diff >= 1e-6"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    /// prop_unitary_norm — over 100 random inputs, both Cayley and Givens
    /// preserve the L2 norm (same 1e-6 tolerance as the KAT).
    #[test]
    fn prop_unitary_norm(
        raw_entries in prop::collection::vec(-1.0f64..1.0f64, 5 * 5 * 2),
        angle_entries in prop::collection::vec(0.0f64..std::f64::consts::TAU, 4),
        psi_entries in prop::collection::vec(-1.0f64..1.0f64, 5 * 2),
    ) {
        let d = 5;

        let mut raw = ComplexMat::zeros(d);
        let mut idx = 0;
        for i in 0..d {
            for j in 0..d {
                raw.set(i, j, raw_entries[idx], raw_entries[idx + 1]);
                idx += 2;
            }
        }
        let u_cayley = cayley(&skew_hermitian_part(&raw));
        let u_givens = givens(&angle_entries, d);

        let psi = Complex::new(psi_entries[0..d].to_vec(), psi_entries[d..2 * d].to_vec());
        let norm_before = psi.norm();

        for u in [&u_cayley, &u_givens] {
            let out = u.apply(&psi);
            let norm_after = out.norm();
            prop_assert!((norm_before - norm_after).abs() < 1e-6);
        }
    }
}
