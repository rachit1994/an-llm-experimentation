//! T0.2 — Finite-difference gradient check for complex multiply.
//!
//! L(a, b) = abs2(mul(a, b)) = (a_re*b_re - a_im*b_im)^2 + (a_re*b_im + a_im*b_re)^2
//!
//! The analytic gradient below is hand-derived (not autodiff, not the kernel's
//! own backward pass) by the chain rule through c = a*b, L = c_re^2 + c_im^2:
//!   dL/dc_re = 2*c_re,  dL/dc_im = 2*c_im
//!   dc_re/da_re =  b_re,  dc_re/da_im = -b_im,  dc_re/db_re =  a_re,  dc_re/db_im = -a_im
//!   dc_im/da_re =  b_im,  dc_im/da_im =  b_re,  dc_im/db_re =  a_im,  dc_im/db_im =  a_re
//! giving (params ordered [a_re, a_im, b_re, b_im]):
//!   dL/da_re =  2*c_re*b_re + 2*c_im*b_im
//!   dL/da_im = -2*c_re*b_im + 2*c_im*b_re
//!   dL/db_re =  2*c_re*a_re + 2*c_im*a_im
//!   dL/db_im = -2*c_re*a_im + 2*c_im*a_re
//!
//! This is checked against a central finite difference; the firewall that
//! catches a wrong backward pass before it poisons any training number.

use qilm_oracle::gradcheck::gradcheck;

fn loss(p: &[f64]) -> f64 {
    let (a_re, a_im, b_re, b_im) = (p[0], p[1], p[2], p[3]);
    let c_re = a_re * b_re - a_im * b_im;
    let c_im = a_re * b_im + a_im * b_re;
    c_re * c_re + c_im * c_im
}

fn analytic_grad(p: &[f64]) -> Vec<f64> {
    let (a_re, a_im, b_re, b_im) = (p[0], p[1], p[2], p[3]);
    let c_re = a_re * b_re - a_im * b_im;
    let c_im = a_re * b_im + a_im * b_re;
    vec![
        2.0 * c_re * b_re + 2.0 * c_im * b_im,
        -2.0 * c_re * b_im + 2.0 * c_im * b_re,
        2.0 * c_re * a_re + 2.0 * c_im * a_im,
        -2.0 * c_re * a_im + 2.0 * c_im * a_re,
    ]
}

#[test]
fn gradcheck_complex() {
    // Several fixed, deterministic (not kernel-derived) parameter sets.
    let cases: Vec<Vec<f64>> = vec![
        vec![0.7, -0.3, 0.5, 0.9],
        vec![1.3, 2.1, -0.8, 0.4],
        vec![-2.0, 0.6, 1.1, -1.4],
        vec![0.05, 3.2, -2.3, 0.02],
    ];

    for params in cases {
        let max_rel = gradcheck(loss, analytic_grad, &params, 1e-4);
        assert!(
            max_rel < 1e-4,
            "gradcheck_complex: max relative error {} >= 1e-4 for params {:?}",
            max_rel,
            params
        );
    }
}
