//! gradcheck_autodiff_linear — finite-difference check for `Tape::matmul` /
//! `Tape::linear` (implementation/tests/PHASE-1.md increment 1, op 2/4:
//! "matmul / linear (x·W + b)").
//!
//! Setup: `x: (2,3)` (batch=2, in=3), `W: (3,2)` (in=3, out=2), `b: (1,2)`.
//! `Y = x*W + b` (shape (2,2)); scalar loss `L = sum(Y_ij^2)`.
//!
//! Hand-derived analytic gradient (independent of the tape's own backward;
//! chain rule through Y = xW + b, L = sum Y_ij^2):
//!   Y[i][j]    = sum_k x[i][k]*W[k][j] + b[j]
//!   dL/dY[i][j] = 2*Y[i][j]
//!   dL/dx[i][k] = sum_j dL/dY[i][j] * W[k][j]
//!   dL/dW[k][j] = sum_i dL/dY[i][j] * x[i][k]
//!   dL/db[j]    = sum_i dL/dY[i][j]
//!
//! Params flattened as [x (6, row-major), W (6, row-major), b (2)].

use qilm_core::autodiff::{Shape, Tape};
use qilm_oracle::gradcheck::gradcheck;

const BATCH: usize = 2;
const IN: usize = 3;
const OUT: usize = 2;

fn split(theta: &[f64]) -> (&[f64], &[f64], &[f64]) {
    let (x, rest) = theta.split_at(BATCH * IN);
    let (w, b) = rest.split_at(IN * OUT);
    (x, w, b)
}

fn forward_y(x: &[f64], w: &[f64], b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; BATCH * OUT];
    for i in 0..BATCH {
        for j in 0..OUT {
            let mut s = b[j];
            for k in 0..IN {
                s += x[i * IN + k] * w[k * OUT + j];
            }
            y[i * OUT + j] = s;
        }
    }
    y
}

fn loss(theta: &[f64]) -> f64 {
    let (x, w, b) = split(theta);
    forward_y(x, w, b).iter().map(|v| v * v).sum()
}

fn analytic_grad(theta: &[f64]) -> Vec<f64> {
    let (x, w, b) = split(theta);
    let y = forward_y(x, w, b);
    let dy: Vec<f64> = y.iter().map(|v| 2.0 * v).collect();

    let mut gx = vec![0.0; BATCH * IN];
    for i in 0..BATCH {
        for k in 0..IN {
            let mut s = 0.0;
            for j in 0..OUT {
                s += dy[i * OUT + j] * w[k * OUT + j];
            }
            gx[i * IN + k] = s;
        }
    }
    let mut gw = vec![0.0; IN * OUT];
    for k in 0..IN {
        for j in 0..OUT {
            let mut s = 0.0;
            for i in 0..BATCH {
                s += dy[i * OUT + j] * x[i * IN + k];
            }
            gw[k * OUT + j] = s;
        }
    }
    let mut gb = vec![0.0; OUT];
    for j in 0..OUT {
        let mut s = 0.0;
        for i in 0..BATCH {
            s += dy[i * OUT + j];
        }
        gb[j] = s;
    }

    let mut grad = gx;
    grad.extend(gw);
    grad.extend(gb);
    grad
}

fn theta_cases() -> Vec<Vec<f64>> {
    vec![
        vec![
            0.4, -0.3, 0.7, -0.6, 0.2, 0.9, // x (2x3)
            0.5, -0.2, 0.1, 0.8, -0.4, 0.3, // W (3x2)
            0.2, -0.1, // b
        ],
        vec![
            -1.1, 0.6, 0.05, 0.3, -0.7, 1.2, //
            -0.3, 0.9, 0.4, -0.6, 0.15, -0.25, //
            -0.5, 0.35,
        ],
    ]
}

#[test]
fn gradcheck_autodiff_linear() {
    for theta in theta_cases() {
        let max_rel = gradcheck(loss, analytic_grad, &theta, 1e-4);
        assert!(
            max_rel < 1e-4,
            "gradcheck_autodiff_linear: max relative error {max_rel} >= 1e-4 for theta {theta:?}"
        );
    }
}

/// Ties the hand-derived oracle to the real tape's `linear` (matmul + add
/// broadcast bias): same forward value, same backward-computed gradient.
#[test]
fn gradcheck_autodiff_linear_oracle_matches_tape() {
    let theta = &theta_cases()[0];
    let (x_vals, w_vals, b_vals) = split(theta);
    let oracle_loss = loss(theta);
    let oracle_grad = analytic_grad(theta);

    let mut tape = Tape::new();
    let x = tape.leaf(x_vals.to_vec(), Shape::mat(BATCH, IN));
    let w = tape.leaf(w_vals.to_vec(), Shape::mat(IN, OUT));
    let b = tape.leaf(b_vals.to_vec(), Shape::row(OUT));
    let y = tape.linear(x, w, b);
    let sq = tape.sum_squares(y);
    tape.backward(sq);

    assert!(
        (tape.value(sq)[0] - oracle_loss).abs() < 1e-9,
        "tape loss {} != oracle loss {}",
        tape.value(sq)[0],
        oracle_loss
    );

    let tape_grad: Vec<f64> = tape
        .grad(x)
        .iter()
        .chain(tape.grad(w).iter())
        .chain(tape.grad(b).iter())
        .cloned()
        .collect();
    for (i, (t, o)) in tape_grad.iter().zip(&oracle_grad).enumerate() {
        assert!((t - o).abs() < 1e-9, "grad[{i}]: tape={t} oracle={o}");
    }
}

/// Anti-vacuity: matmul with transposed dimensions (a common off-by-one bug
/// — reading W as (out,in) instead of (in,out)) must NOT match the correct
/// oracle forward value, proving the equality check above can say "no".
#[test]
fn gradcheck_autodiff_linear_can_say_no_on_transposed_weight() {
    let theta = &theta_cases()[0];
    let (x_vals, w_vals, b_vals) = split(theta);
    let correct_loss = loss(theta);

    // Deliberately WRONG: index W as if it were (OUT, IN) row-major (a
    // transpose bug) instead of the correct (IN, OUT).
    let mut broken_y = [0.0; BATCH * OUT];
    for i in 0..BATCH {
        for j in 0..OUT {
            let mut s = b_vals[j];
            for k in 0..IN {
                s += x_vals[i * IN + k] * w_vals[j * IN + k]; // wrong index order
            }
            broken_y[i * OUT + j] = s;
        }
    }
    let broken_loss: f64 = broken_y.iter().map(|v| v * v).sum();

    assert!(
        (broken_loss - correct_loss).abs() > 1e-3,
        "transposed-weight forward should visibly disagree with the correct oracle value"
    );
}
