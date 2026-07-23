//! prop_autodiff_sum_squares — the mandated property test for the autodiff
//! tape (implementation/tests/PHASE-1.md increment 1, prior report requirement
//! "backward of a sum-of-squares loss w.r.t. a leaf equals 2*leaf"), and the
//! FIRST test the arena/backward machinery has to pass before any real op
//! (add/matmul/tanh/log_softmax) is layered on top.
//!
//! `L(x) = sum_i x_i^2`, `dL/dx_i = 2*x_i` for every `i` — a closed form
//! independent of the tape (elementary calculus, no tape call involved in
//! deriving it). Checked three ways:
//!   1. the closed form against `qilm_oracle::gradcheck`'s central finite
//!      difference (the numeric oracle, never touches backward code);
//!   2. the closed form against `Tape::sum_squares` + `Tape::backward`'s
//!      actual output, for several leaves (including one containing a zero,
//!      the boundary case for a squared term);
//!   3. a canary: a deliberately wrong "grad = leaf" (missing the factor of
//!      2, the single easiest bug in this kernel) must NOT match the tape's
//!      real gradient, proving the check in (2) can say "no".

use qilm_core::autodiff::{Shape, Tape};
use qilm_oracle::gradcheck::gradcheck;

fn loss(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

fn analytic_grad(x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| 2.0 * v).collect()
}

#[test]
fn prop_autodiff_sum_squares_gradcheck() {
    let cases: Vec<Vec<f64>> = vec![vec![0.5, -1.3, 2.0, 0.0], vec![-0.7, 0.02, -2.4], vec![3.1]];
    for x in cases {
        let max_rel = gradcheck(loss, analytic_grad, &x, 1e-4);
        assert!(
            max_rel < 1e-4,
            "prop_autodiff_sum_squares: max relative error {max_rel} >= 1e-4 for x {x:?}"
        );
    }
}

#[test]
fn prop_autodiff_sum_squares_matches_tape() {
    let cases: Vec<Vec<f64>> = vec![
        vec![0.5, -1.3, 2.0, 0.0],
        vec![-0.7, 0.02, -2.4],
        vec![3.1],
        vec![0.0, 0.0, 0.0],
    ];
    for x in cases {
        let expected_grad = analytic_grad(&x);
        let expected_loss = loss(&x);

        let mut tape = Tape::new();
        let leaf = tape.leaf(x.clone(), Shape::row(x.len()));
        let out = tape.sum_squares(leaf);
        tape.backward(out);

        assert!(
            (tape.value(out)[0] - expected_loss).abs() < 1e-9,
            "tape loss {} != closed-form loss {} for x {:?}",
            tape.value(out)[0],
            expected_loss,
            x
        );
        for (i, (g, e)) in tape.grad(leaf).iter().zip(&expected_grad).enumerate() {
            assert!(
                (g - e).abs() < 1e-9,
                "grad[{i}] = {g}, expected 2*x = {e} for x {x:?}"
            );
        }
    }
}

/// Anti-vacuity canary: the single easiest wrong implementation of this
/// kernel — forgetting the factor of 2 (`grad = x` instead of `grad = 2x`) —
/// must NOT match the real tape's gradient. If it did, the equality check
/// above would be trivially satisfiable and prove nothing.
#[test]
fn prop_autodiff_sum_squares_can_say_no_on_missing_factor_of_two() {
    let x = vec![0.5, -1.3, 2.0];
    let wrong_grad = x.clone(); // missing the factor of 2 — the bug

    let mut tape = Tape::new();
    let leaf = tape.leaf(x.clone(), Shape::row(x.len()));
    let out = tape.sum_squares(leaf);
    tape.backward(out);
    let real_grad = tape.grad(leaf);

    let mut max_abs_diff: f64 = 0.0;
    for (r, w) in real_grad.iter().zip(&wrong_grad) {
        max_abs_diff = max_abs_diff.max((r - w).abs());
    }
    assert!(
        max_abs_diff > 0.1,
        "the wrong (factor-of-2-missing) gradient should visibly disagree with the tape's real \
         gradient, got max abs diff {max_abs_diff}"
    );
}
