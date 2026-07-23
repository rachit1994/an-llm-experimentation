//! gradcheck_born — finite-difference check for the Born readout's learnable
//! parameters (the measurement vectors `m_k`).
//!
//! Not explicitly named in PHASE-0.md's T0.4 "Done-when" bullet, but added
//! per the general rule (VERIFICATION.md §3 / 00-stack-and-principles.md):
//! "every kernel with parameters" needs a gradcheck. `measurements` are
//! learned in the full model (they are what "measures to speak" — the 256-way
//! Born head in METRICS-AND-GATES.md §1), so they are parameters.
//!
//! Setup: 2-dimensional psi and two measurement vectors m0 (parameterized,
//! theta = [m0_re0, m0_re1, m0_im0, m0_im1]), m1 (fixed). Loss = Born
//! probability of the m0 outcome, L = P0 / (P0 + P1) where P_k = |<m_k|psi>|^2.
//!
//! Hand-derived analytic gradient (chain rule through A = <m0|psi>):
//!   Are = sum_i (m0_re_i*psi_re_i + m0_im_i*psi_im_i)
//!   Aim = sum_i (m0_re_i*psi_im_i - m0_im_i*psi_re_i)
//!   P0 = Are^2 + Aim^2,  L = P0 / (P0 + P1)
//!   dL/dP0 = P1 / (P0+P1)^2
//!   dP0/dm0_re_i = 2*Are*psi_re_i + 2*Aim*psi_im_i
//!   dP0/dm0_im_i = 2*Are*psi_im_i - 2*Aim*psi_re_i
//!   dL/dm0_re_i  = dL/dP0 * dP0/dm0_re_i   (and similarly for m0_im_i)

use qilm_core::born::born;
use qilm_core::complex::Complex;
use qilm_oracle::gradcheck::gradcheck;

const PSI: [f64; 2] = [0.6, -0.2]; // psi_re
const PSI_IM: [f64; 2] = [0.3, 0.5]; // psi_im
const M1: [f64; 2] = [0.1, 0.9]; // m1_re (fixed second measurement)
const M1_IM: [f64; 2] = [-0.4, 0.2]; // m1_im

fn p0_of(theta: &[f64]) -> f64 {
    let (m0_re, m0_im) = (&theta[0..2], &theta[2..4]);
    let mut are = 0.0;
    let mut aim = 0.0;
    for i in 0..2 {
        are += m0_re[i] * PSI[i] + m0_im[i] * PSI_IM[i];
        aim += m0_re[i] * PSI_IM[i] - m0_im[i] * PSI[i];
    }
    are * are + aim * aim
}

fn p1_fixed() -> f64 {
    let mut are = 0.0;
    let mut aim = 0.0;
    for i in 0..2 {
        are += M1[i] * PSI[i] + M1_IM[i] * PSI_IM[i];
        aim += M1[i] * PSI_IM[i] - M1_IM[i] * PSI[i];
    }
    are * are + aim * aim
}

fn loss(theta: &[f64]) -> f64 {
    let p0 = p0_of(theta);
    let p1 = p1_fixed();
    p0 / (p0 + p1)
}

fn analytic_grad(theta: &[f64]) -> Vec<f64> {
    let (m0_re, m0_im) = (&theta[0..2], &theta[2..4]);
    let mut are = 0.0;
    let mut aim = 0.0;
    for i in 0..2 {
        are += m0_re[i] * PSI[i] + m0_im[i] * PSI_IM[i];
        aim += m0_re[i] * PSI_IM[i] - m0_im[i] * PSI[i];
    }
    let p0 = are * are + aim * aim;
    let p1 = p1_fixed();
    let dl_dp0 = p1 / (p0 + p1).powi(2);

    let mut grad = vec![0.0; 4];
    for i in 0..2 {
        let dp0_dre = 2.0 * are * PSI[i] + 2.0 * aim * PSI_IM[i];
        let dp0_dim = 2.0 * are * PSI_IM[i] - 2.0 * aim * PSI[i];
        grad[i] = dl_dp0 * dp0_dre; // d L / d m0_re[i]
        grad[2 + i] = dl_dp0 * dp0_dim; // d L / d m0_im[i]
    }
    grad
}

#[test]
fn gradcheck_born() {
    let cases: Vec<Vec<f64>> = vec![
        vec![0.5, 0.3, -0.2, 0.7],
        vec![-1.1, 0.4, 0.9, -0.6],
        vec![0.05, -0.9, 1.3, 0.2],
    ];
    for theta in cases {
        let max_rel = gradcheck(loss, analytic_grad, &theta, 1e-4);
        assert!(max_rel < 1e-4, "gradcheck_born: max relative error {max_rel} >= 1e-4 for theta {theta:?}");
    }
}

/// Ties the scalar oracle above to the actual tensor kernel (doc 00: "the
/// tensor kernel must match the oracle to float tolerance"): the oracle's
/// hand-written P0/(P0+P1) formula must equal qilm_core::born::born's output
/// for the same inputs, so gradcheck_born is validating the real kernel's
/// math, not a disconnected formula.
#[test]
fn gradcheck_born_oracle_matches_kernel() {
    let theta = vec![0.5, 0.3, -0.2, 0.7];
    let oracle_l = loss(&theta);

    let psi = Complex::new(PSI.to_vec(), PSI_IM.to_vec());
    let m0 = Complex::new(theta[0..2].to_vec(), theta[2..4].to_vec());
    let m1 = Complex::new(M1.to_vec(), M1_IM.to_vec());
    let probs = born(&psi, &[m0, m1]);

    assert!(
        (probs[0] - oracle_l).abs() < 1e-9,
        "kernel born()[0] = {}, oracle loss = {}",
        probs[0],
        oracle_l
    );
}
