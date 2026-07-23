//! gradcheck_givens — finite-difference check for the Givens unitary layer's
//! learnable parameters (the rotation angles).
//!
//! Not explicitly named in PHASE-0.md's T0.5 "Done-when" bullet (which lists
//! kat_cayley_unitary, kat_unitary_norm, prop_unitary_norm), but added per the
//! general rule that every kernel with parameters gets a gradcheck (angles
//! are literally the trained parameters of the unitary-dynamics layer used in
//! Phase 5). Cayley's parameters go through a matrix inverse; a correct
//! closed-form gradient for that is out of scope for Phase 0 (see final
//! report) — Cayley's correctness instead rests on the exact algebraic
//! identities (kat_cayley_unitary, kat_unitary_norm, prop_unitary_norm).
//!
//! Setup: d=3, a single sweep U(theta0, theta1) = G1(theta1) * G0(theta0),
//! where G0 rotates the (0,1) plane and G1 rotates the (1,2) plane. Since
//! every Givens rotation is real, U is real, and it acts on a complex vector
//! psi by applying the same real matrix to psi's re and im parts separately:
//! (U psi)_re = U * psi_re, (U psi)_im = U * psi_im.
//!
//! Loss: L(theta) = |<target | U psi>|^2 for fixed target, psi.
//!
//! Hand-derived analytic gradient (product rule through U = G1 * G0):
//!   dU/dtheta0 = G1 * dG0/dtheta0,   dU/dtheta1 = dG1/dtheta1 * G0
//! where dG_i/dtheta_i is the elementwise derivative of the 2x2 rotation
//! block (d/dtheta [c -s; s c] = [-s -c; c -s]), zero elsewhere.
//! Then, writing f_re = U*psi_re, f_im = U*psi_im:
//!   inner_re = target_re . f_re + target_im . f_im
//!   inner_im = target_re . f_im - target_im . f_re
//!   L = inner_re^2 + inner_im^2
//!   dL/dtheta_k = 2*inner_re*(target_re . df_re + target_im . df_im)
//!               + 2*inner_im*(target_re . df_im - target_im . df_re)
//! with df_re/dtheta_k = (dU/dtheta_k) * psi_re, df_im/dtheta_k = (dU/dtheta_k) * psi_im.

use qilm_oracle::gradcheck::gradcheck;

type Mat3 = [[f64; 3]; 3];

fn identity3() -> Mat3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// Rotation matrix embedding a 2D rotation by `theta` in plane (i, i+1) of a
/// 3x3 identity.
fn rot(theta: f64, i: usize) -> Mat3 {
    let mut m = identity3();
    let (c, s) = (theta.cos(), theta.sin());
    m[i][i] = c;
    m[i][i + 1] = -s;
    m[i + 1][i] = s;
    m[i + 1][i + 1] = c;
    m
}

/// d/dtheta of `rot(theta, i)`: zero everywhere except the same 2x2 block.
fn rot_deriv(theta: f64, i: usize) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    let (c, s) = (theta.cos(), theta.sin());
    m[i][i] = -s;
    m[i][i + 1] = -c;
    m[i + 1][i] = c;
    m[i + 1][i + 1] = -s;
    m
}

fn matmul3(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i][k] * b[k][j];
            }
            out[i][j] = s;
        }
    }
    out
}

fn matvec3(a: &Mat3, v: &[f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2];
    }
    out
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

const PSI_RE: [f64; 3] = [0.4, -0.7, 0.2];
const PSI_IM: [f64; 3] = [0.1, 0.3, -0.5];
const TARGET_RE: [f64; 3] = [0.6, 0.2, -0.3];
const TARGET_IM: [f64; 3] = [-0.2, 0.5, 0.1];

fn loss(theta: &[f64]) -> f64 {
    let (theta0, theta1) = (theta[0], theta[1]);
    let u = matmul3(&rot(theta1, 1), &rot(theta0, 0));
    let f_re = matvec3(&u, &PSI_RE);
    let f_im = matvec3(&u, &PSI_IM);
    let inner_re = dot3(&TARGET_RE, &f_re) + dot3(&TARGET_IM, &f_im);
    let inner_im = dot3(&TARGET_RE, &f_im) - dot3(&TARGET_IM, &f_re);
    inner_re * inner_re + inner_im * inner_im
}

fn analytic_grad(theta: &[f64]) -> Vec<f64> {
    let (theta0, theta1) = (theta[0], theta[1]);
    let g0 = rot(theta0, 0);
    let g1 = rot(theta1, 1);
    let u = matmul3(&g1, &g0);

    let f_re = matvec3(&u, &PSI_RE);
    let f_im = matvec3(&u, &PSI_IM);
    let inner_re = dot3(&TARGET_RE, &f_re) + dot3(&TARGET_IM, &f_im);
    let inner_im = dot3(&TARGET_RE, &f_im) - dot3(&TARGET_IM, &f_re);

    let du_dtheta0 = matmul3(&g1, &rot_deriv(theta0, 0));
    let du_dtheta1 = matmul3(&rot_deriv(theta1, 1), &g0);

    let mut grad = vec![0.0; 2];
    for (k, du) in [du_dtheta0, du_dtheta1].iter().enumerate() {
        let df_re = matvec3(du, &PSI_RE);
        let df_im = matvec3(du, &PSI_IM);
        let dinner_re = dot3(&TARGET_RE, &df_re) + dot3(&TARGET_IM, &df_im);
        let dinner_im = dot3(&TARGET_RE, &df_im) - dot3(&TARGET_IM, &df_re);
        grad[k] = 2.0 * inner_re * dinner_re + 2.0 * inner_im * dinner_im;
    }
    grad
}

#[test]
fn gradcheck_givens() {
    let cases: Vec<Vec<f64>> = vec![
        vec![0.3, -0.8],
        vec![1.2, 2.5],
        vec![-0.6, 0.05],
    ];
    for theta in cases {
        let max_rel = gradcheck(loss, analytic_grad, &theta, 1e-4);
        assert!(max_rel < 1e-4, "gradcheck_givens: max relative error {max_rel} >= 1e-4 for theta {theta:?}");
    }
}

/// Ties the oracle's rotation-matrix formula to the real `givens` kernel:
/// applying U built by the oracle's `rot`/`matmul3` must match
/// qilm_core::unitary::givens applied via ComplexMat::apply.
#[test]
fn gradcheck_givens_oracle_matches_kernel() {
    use qilm_core::complex::Complex;
    use qilm_core::unitary::givens;

    let theta = [0.3_f64, -0.8_f64];
    let u_oracle = matmul3(&rot(theta[1], 1), &rot(theta[0], 0));
    let f_re_oracle = matvec3(&u_oracle, &PSI_RE);
    let f_im_oracle = matvec3(&u_oracle, &PSI_IM);

    let u_kernel = givens(&theta, 3);
    let psi = Complex::new(PSI_RE.to_vec(), PSI_IM.to_vec());
    let out = u_kernel.apply(&psi);

    for i in 0..3 {
        assert!((out.re[i] - f_re_oracle[i]).abs() < 1e-9, "re[{i}]: kernel={} oracle={}", out.re[i], f_re_oracle[i]);
        assert!((out.im[i] - f_im_oracle[i]).abs() < 1e-9, "im[{i}]: kernel={} oracle={}", out.im[i], f_im_oracle[i]);
    }
}
