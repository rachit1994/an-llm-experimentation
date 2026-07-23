//! T0.4 — Born readout KATs.
//!
//! born(psi, measurements)_k = |<m_k|psi>|^2 / sum_j |<m_j|psi>|^2

use qilm_core::born::born;
use qilm_core::complex::Complex;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn random_complex(rng: &mut ChaCha8Rng, d: usize) -> Complex {
    let re = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
    let im = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
    Complex::new(re, im)
}

/// kat_born_sum — probabilities from a random state against 256 random
/// measurement vectors sum to 1 within 1e-6, regardless of the (nonzero,
/// generic) inputs. This is the normalization identity, not a fact about any
/// particular state.
#[test]
fn kat_born_sum() {
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let d = 16;
    let psi = random_complex(&mut rng, d);
    let measurements: Vec<Complex> = (0..256).map(|_| random_complex(&mut rng, d)).collect();

    let probs = born(&psi, &measurements);
    assert_eq!(probs.len(), 256);
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "sum of Born probabilities = {sum}, expected 1.0"
    );
    for p in &probs {
        assert!(*p >= 0.0, "Born probability must be nonnegative, got {p}");
    }
}

/// kat_born_hand_values — a small, fully hand-computed example.
/// psi = (1, 0) (i.e. psi = 1 + 0i in a 1-dim Hilbert space... use d=2 for a
/// genuine two-outcome measurement):
///   psi = (re=[1,0], im=[0,0])              -- the state |0>
///   m0  = (re=[1,0], im=[0,0])              -- measures |0>
///   m1  = (re=[0,1], im=[0,0])              -- measures |1>
/// <m0|psi> = conj(m0) . psi = 1*1 + 0*0 = 1        -> |<m0|psi>|^2 = 1
/// <m1|psi> = conj(m1) . psi = 0*1 + 1*0 = 0        -> |<m1|psi>|^2 = 0
/// Born probabilities (hand-computed): [1/(1+0), 0/(1+0)] = [1.0, 0.0]
#[test]
fn kat_born_hand_values() {
    let psi = Complex::new(vec![1.0, 0.0], vec![0.0, 0.0]);
    let m0 = Complex::new(vec![1.0, 0.0], vec![0.0, 0.0]);
    let m1 = Complex::new(vec![0.0, 1.0], vec![0.0, 0.0]);

    let probs = born(&psi, &[m0, m1]);
    assert!((probs[0] - 1.0).abs() < 1e-9, "probs[0] = {}", probs[0]);
    assert!((probs[1] - 0.0).abs() < 1e-9, "probs[1] = {}", probs[1]);
}

/// kat_born_hand_values_superposition — an equal superposition measured in
/// the computational basis.
///   psi = (1/sqrt(2), 1/sqrt(2)) real, i.e. re=[0.70710678, 0.70710678], im=[0,0]
///   m0 = |0> = (re=[1,0], im=[0,0]);  m1 = |1> = (re=[0,1], im=[0,0])
/// <m0|psi> = 0.70710678  -> |.|^2 = 0.5
/// <m1|psi> = 0.70710678  -> |.|^2 = 0.5
/// sum = 1.0, so normalized probabilities are exactly [0.5, 0.5] (hand-computed).
#[test]
fn kat_born_hand_values_superposition() {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let psi = Complex::new(vec![s, s], vec![0.0, 0.0]);
    let m0 = Complex::new(vec![1.0, 0.0], vec![0.0, 0.0]);
    let m1 = Complex::new(vec![0.0, 1.0], vec![0.0, 0.0]);

    let probs = born(&psi, &[m0, m1]);
    assert!((probs[0] - 0.5).abs() < 1e-9, "probs[0] = {}", probs[0]);
    assert!((probs[1] - 0.5).abs() < 1e-9, "probs[1] = {}", probs[1]);
}
