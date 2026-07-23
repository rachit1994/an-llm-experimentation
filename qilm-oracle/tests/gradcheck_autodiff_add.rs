//! gradcheck_autodiff_add — finite-difference check for `Tape::add`
//! (implementation/tests/PHASE-1.md increment 1: the autodiff tape, op 1/4).
//!
//! Setup: a (2x3) matrix `A` and a (1x3) bias row `b`; `Y = A + b` (row
//! broadcast), then a scalar loss `L = sum(Y_ij^2)` so the test exercises a
//! non-trivial downstream gradient, not just `dL/dY = 1`.
//!
//! Hand-derived analytic gradient (independent of the tape's own backward,
//! chain rule through Y = A + b, L = sum Y_ij^2):
//!   dL/dY_ij = 2*Y_ij
//!   dY_ij/dA_ij = 1        =>  dL/dA_ij = 2*Y_ij
//!   dY_ij/db_j  = 1 (every row) => dL/db_j = sum_i 2*Y_ij
//!
//! Params are flattened as [A (6 entries, row-major), b (3 entries)].
//!
//! This is checked two ways: (1) the oracle above against
//! `qilm_oracle::gradcheck`'s central finite difference (a numeric oracle
//! that never touches any backward code — the standard independent check),
//! and (2) the oracle's forward VALUE and analytic GRADIENT against the real
//! `qilm_core::autodiff::Tape`'s forward output and `backward()` output, at
//! matching float precision — this is what proves the tape's `add` backward
//! (not just this test file's hand-derived formula) is correct.

use qilm_core::autodiff::{Shape, Tape};
use qilm_oracle::gradcheck::gradcheck;

const ROWS: usize = 2;
const COLS: usize = 3;

fn split(theta: &[f64]) -> (&[f64], &[f64]) {
    theta.split_at(ROWS * COLS)
}

fn loss(theta: &[f64]) -> f64 {
    let (a, b) = split(theta);
    let mut total = 0.0;
    for r in 0..ROWS {
        for c in 0..COLS {
            let y = a[r * COLS + c] + b[c];
            total += y * y;
        }
    }
    total
}

fn analytic_grad(theta: &[f64]) -> Vec<f64> {
    let (a, b) = split(theta);
    let mut ga = vec![0.0; ROWS * COLS];
    let mut gb = vec![0.0; COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            let y = a[r * COLS + c] + b[c];
            ga[r * COLS + c] = 2.0 * y;
            gb[c] += 2.0 * y;
        }
    }
    let mut grad = ga;
    grad.extend(gb);
    grad
}

#[test]
fn gradcheck_autodiff_add() {
    let cases: Vec<Vec<f64>> = vec![
        vec![0.3, -1.1, 0.7, 2.0, -0.4, 0.1, 0.5, -0.2, 0.9],
        vec![-0.6, 0.2, 1.4, -0.9, 0.3, 0.05, -1.3, 0.8, -0.1],
    ];
    for theta in cases {
        let max_rel = gradcheck(loss, analytic_grad, &theta, 1e-4);
        assert!(
            max_rel < 1e-4,
            "gradcheck_autodiff_add: max relative error {max_rel} >= 1e-4 for theta {theta:?}"
        );
    }
}

/// Ties the hand-derived oracle to the real tape: same forward value, same
/// backward-computed gradient (to near machine precision).
#[test]
fn gradcheck_autodiff_add_oracle_matches_tape() {
    let theta = vec![0.3, -1.1, 0.7, 2.0, -0.4, 0.1, 0.5, -0.2, 0.9];
    let (a_vals, b_vals) = split(&theta);
    let oracle_loss = loss(&theta);
    let oracle_grad = analytic_grad(&theta);

    let mut tape = Tape::new();
    let a = tape.leaf(a_vals.to_vec(), Shape::mat(ROWS, COLS));
    let b = tape.leaf(b_vals.to_vec(), Shape::row(COLS));
    let y = tape.add(a, b);
    let sq = tape.sum_squares(y);
    tape.backward(sq);

    assert!(
        (tape.value(sq)[0] - oracle_loss).abs() < 1e-9,
        "tape loss {} != oracle loss {}",
        tape.value(sq)[0],
        oracle_loss
    );

    let tape_grad: Vec<f64> = tape
        .grad(a)
        .iter()
        .chain(tape.grad(b).iter())
        .cloned()
        .collect();
    for (i, (t, o)) in tape_grad.iter().zip(&oracle_grad).enumerate() {
        assert!((t - o).abs() < 1e-9, "grad[{i}]: tape={t} oracle={o}");
    }
}

/// Anti-vacuity: a broken `add` (drops the bias entirely) must make the
/// tape-vs-oracle forward-value check above fail — proves the comparison can
/// say "no", not just "yes" on whatever the tape happens to compute.
#[test]
fn gradcheck_autodiff_add_can_say_no_on_missing_bias() {
    let theta = vec![0.3, -1.1, 0.7, 2.0, -0.4, 0.1, 0.5, -0.2, 0.9];
    let (a_vals, b_vals) = split(&theta);
    let oracle_loss = loss(&theta);

    // Deliberately WRONG: forward without adding the bias at all (b unused).
    let mut broken_sq = 0.0;
    for &v in a_vals {
        broken_sq += v * v;
    }
    let _ = b_vals; // bias never applied — the bug under test

    assert!(
        (broken_sq - oracle_loss).abs() > 1e-3,
        "broken (bias-dropped) forward should visibly disagree with the correct oracle value"
    );
}
