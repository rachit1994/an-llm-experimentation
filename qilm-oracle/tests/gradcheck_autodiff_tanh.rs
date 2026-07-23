//! gradcheck_autodiff_tanh — finite-difference check for `Tape::tanh`
//! (implementation/tests/PHASE-1.md increment 1, op 3/4: the nonlinearity).
//!
//! Setup: a leaf `x` of length 4, `y = tanh(x)`, scalar loss `L = sum(y_i^2)`.
//!
//! Hand-derived analytic gradient (independent of the tape's own backward;
//! chain rule through y = tanh(x), L = sum y_i^2):
//!   dL/dy_i = 2*y_i
//!   dy_i/dx_i = 1 - tanh(x_i)^2 = 1 - y_i^2
//!   dL/dx_i = 2*y_i*(1 - y_i^2)

use qilm_core::autodiff::{Shape, Tape};
use qilm_oracle::gradcheck::gradcheck;

fn loss(x: &[f64]) -> f64 {
    x.iter().map(|v| v.tanh().powi(2)).sum()
}

fn analytic_grad(x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|v| {
            let y = v.tanh();
            2.0 * y * (1.0 - y * y)
        })
        .collect()
}

fn cases() -> Vec<Vec<f64>> {
    vec![
        vec![0.5, -1.3, 2.0, 0.0],
        vec![-3.0, 0.02, 4.5, -0.7],
        vec![10.0, -10.0, 0.1], // saturating regime, tanh(±10) ~ ±1
    ]
}

#[test]
fn gradcheck_autodiff_tanh() {
    for x in cases() {
        let max_rel = gradcheck(loss, analytic_grad, &x, 1e-4);
        assert!(
            max_rel < 1e-4,
            "gradcheck_autodiff_tanh: max relative error {max_rel} >= 1e-4 for x {x:?}"
        );
    }
}

/// Ties the hand-derived oracle to the real tape's `tanh`: same forward
/// value, same backward-computed gradient.
#[test]
fn gradcheck_autodiff_tanh_oracle_matches_tape() {
    for x in cases() {
        let oracle_loss = loss(&x);
        let oracle_grad = analytic_grad(&x);

        let mut tape = Tape::new();
        let leaf = tape.leaf(x.clone(), Shape::row(x.len()));
        let y = tape.tanh(leaf);
        let sq = tape.sum_squares(y);
        tape.backward(sq);

        assert!(
            (tape.value(sq)[0] - oracle_loss).abs() < 1e-9,
            "tape loss {} != oracle loss {} for x {x:?}",
            tape.value(sq)[0],
            oracle_loss
        );
        for (i, (t, o)) in tape.grad(leaf).iter().zip(&oracle_grad).enumerate() {
            assert!(
                (t - o).abs() < 1e-9,
                "grad[{i}]: tape={t} oracle={o} for x {x:?}"
            );
        }
    }
}

/// Anti-vacuity: the identity-derivative bug (using `dy/dx = 1` instead of
/// `1 - tanh(x)^2`, i.e. forgetting tanh saturates) must NOT match the real
/// tape's gradient away from x=0, proving the check above can say "no".
#[test]
fn gradcheck_autodiff_tanh_can_say_no_on_identity_derivative() {
    let x = vec![2.0, -2.0, 3.0];

    let mut tape = Tape::new();
    let leaf = tape.leaf(x.clone(), Shape::row(x.len()));
    let y = tape.tanh(leaf);
    let sq = tape.sum_squares(y);
    tape.backward(sq);
    let real_grad = tape.grad(leaf).to_vec();

    // Deliberately WRONG: dL/dx_i = 2*y_i * 1 (missing the (1 - y_i^2) term).
    let wrong_grad: Vec<f64> = x.iter().map(|v| 2.0 * v.tanh()).collect();

    let mut max_abs_diff: f64 = 0.0;
    for (r, w) in real_grad.iter().zip(&wrong_grad) {
        max_abs_diff = max_abs_diff.max((r - w).abs());
    }
    assert!(
        max_abs_diff > 0.1,
        "the wrong (identity-derivative) gradient should visibly disagree with the tape's real \
         gradient away from x=0, got max abs diff {max_abs_diff}"
    );
}
