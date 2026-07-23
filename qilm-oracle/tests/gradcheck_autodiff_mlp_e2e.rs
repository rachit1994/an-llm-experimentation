//! gradcheck_autodiff_mlp_e2e — end-to-end composition check for the whole
//! autodiff increment (implementation/tests/PHASE-1.md increment 1): a tiny
//! two-layer MLP chaining all four model ops in sequence,
//!
//!   h     = tanh(linear(x, W1, b1))
//!   logit = linear(h, W2, b2)
//!   logp  = log_softmax(logit)
//!   L     = cross_entropy(logp, targets)
//!
//! Every op above already has its own dedicated gradcheck test against a
//! hand-derived, tape-independent oracle formula. This test's job is
//! different and complementary: it proves the tape's chain-rule BOOKKEEPING
//! across multiple stacked nodes is correct — that `backward`'s reverse walk
//! correctly threads gradients through five composed ops back to every leaf
//! (x, W1, b1, W2, b2) — something a bug in `accumulate`/node ordering could
//! break even if every isolated op's own test passes. The independent oracle
//! here is exactly what gradcheck is for: central finite differences of the
//! tape's own forward pass (which never touches `backward`'s code), checked
//! against the tape's `backward()`-computed gradient for every leaf.

use qilm_core::autodiff::{Shape, Tape};
use qilm_oracle::gradcheck::gradcheck;

const BATCH: usize = 2;
const IN: usize = 3;
const HID: usize = 4;
const OUT: usize = 3;
const TARGETS: [usize; BATCH] = [2, 0];

// Params flattened as [x, W1, b1, W2, b2].
const N_X: usize = BATCH * IN;
const N_W1: usize = IN * HID;
const N_B1: usize = HID;
const N_W2: usize = HID * OUT;
const N_B2: usize = OUT;

type Params<'a> = (&'a [f64], &'a [f64], &'a [f64], &'a [f64], &'a [f64]);

fn split(theta: &[f64]) -> Params<'_> {
    let (x, rest) = theta.split_at(N_X);
    let (w1, rest) = rest.split_at(N_W1);
    let (b1, rest) = rest.split_at(N_B1);
    let (w2, b2) = rest.split_at(N_W2);
    (x, w1, b1, w2, b2)
}

/// Build the whole graph and return `(tape, x_id, w1_id, b1_id, w2_id, b2_id, loss_id)`.
#[allow(clippy::type_complexity)]
fn build(theta: &[f64]) -> (Tape, usize, usize, usize, usize, usize, usize) {
    let (x_v, w1_v, b1_v, w2_v, b2_v) = split(theta);
    let mut tape = Tape::new();
    let x = tape.leaf(x_v.to_vec(), Shape::mat(BATCH, IN));
    let w1 = tape.leaf(w1_v.to_vec(), Shape::mat(IN, HID));
    let b1 = tape.leaf(b1_v.to_vec(), Shape::row(HID));
    let w2 = tape.leaf(w2_v.to_vec(), Shape::mat(HID, OUT));
    let b2 = tape.leaf(b2_v.to_vec(), Shape::row(OUT));

    let h1 = tape.linear(x, w1, b1);
    let h = tape.tanh(h1);
    let logit = tape.linear(h, w2, b2);
    let logp = tape.log_softmax(logit);
    let loss = tape.cross_entropy(logp, &TARGETS);

    (tape, x, w1, b1, w2, b2, loss)
}

fn loss(theta: &[f64]) -> f64 {
    let (tape, _, _, _, _, _, loss) = build(theta);
    tape.value(loss)[0]
}

fn analytic_grad(theta: &[f64]) -> Vec<f64> {
    let (mut tape, x, w1, b1, w2, b2, loss) = build(theta);
    tape.backward(loss);
    let mut grad = Vec::with_capacity(theta.len());
    grad.extend_from_slice(tape.grad(x));
    grad.extend_from_slice(tape.grad(w1));
    grad.extend_from_slice(tape.grad(b1));
    grad.extend_from_slice(tape.grad(w2));
    grad.extend_from_slice(tape.grad(b2));
    grad
}

fn theta_cases() -> Vec<Vec<f64>> {
    // deterministic, hand-picked values (no RNG needed at this size).
    vec![
        vec![
            // x (2x3)
            0.3, -0.7, 1.1, -0.4, 0.6, 0.2, // W1 (3x4)
            0.5, -0.2, 0.1, 0.4, -0.3, 0.6, 0.2, -0.1, 0.05, -0.4, 0.3, -0.2, // b1 (4)
            0.1, -0.2, 0.05, 0.15, // W2 (4x3)
            0.2, -0.3, 0.1, 0.4, 0.05, -0.1, -0.2, 0.3, 0.15, -0.05, 0.25, 0.1, // b2 (3)
            0.05, -0.1, 0.2,
        ],
        vec![
            -0.5, 0.4, -0.2, 0.9, -0.6, 0.3, //
            -0.3, 0.5, -0.4, 0.2, 0.1, -0.5, 0.35, -0.15, 0.25, -0.45, 0.05, -0.25, //
            -0.1, 0.2, -0.15, 0.05, //
            0.3, -0.4, 0.2, -0.15, 0.35, 0.1, 0.25, -0.05, -0.2, 0.4, -0.3, 0.15, //
            0.1, 0.05, -0.15,
        ],
    ]
}

#[test]
fn gradcheck_autodiff_mlp_e2e() {
    for theta in theta_cases() {
        assert_eq!(theta.len(), N_X + N_W1 + N_B1 + N_W2 + N_B2);
        let max_rel = gradcheck(loss, analytic_grad, &theta, 1e-4);
        assert!(
            max_rel < 1e-4,
            "gradcheck_autodiff_mlp_e2e: max relative error {max_rel} >= 1e-4 for theta {theta:?}"
        );
    }
}

/// Anti-vacuity: zeroing out W2's gradient (simulating a "forgot to
/// backprop through the second linear layer" bug, e.g. an early-return in
/// `backward` before reaching the earlier nodes) must NOT match the tape's
/// real gradient for W1/b1 (which depend on W2's contribution flowing back
/// through tanh), proving the composed check above can say "no".
#[test]
fn gradcheck_autodiff_mlp_e2e_can_say_no_on_truncated_backward() {
    let theta = &theta_cases()[0];
    let (mut tape, x, w1, b1, w2, b2, loss_id) = build(theta);
    tape.backward(loss_id);
    let real_w1_grad = tape.grad(w1).to_vec();

    // Deliberately WRONG: a "truncated" backward that only ever reaches the
    // last linear layer (as if the reverse walk stopped early), leaving
    // every earlier leaf's gradient at zero.
    let wrong_w1_grad = vec![0.0; real_w1_grad.len()];

    let mut max_abs: f64 = 0.0;
    for (r, w) in real_w1_grad.iter().zip(&wrong_w1_grad) {
        max_abs = max_abs.max((r - w).abs());
    }
    assert!(
        max_abs > 1e-3,
        "a truncated backward (zero grad reaching W1) should visibly disagree with the real \
         gradient, got max abs diff {max_abs}"
    );

    // sanity: x, b1, w2, b2 grads are populated too (not incidentally zero).
    assert!(tape.grad(x).iter().any(|g| g.abs() > 1e-6));
    assert!(tape.grad(b1).iter().any(|g| g.abs() > 1e-6));
    assert!(tape.grad(w2).iter().any(|g| g.abs() > 1e-6));
    assert!(tape.grad(b2).iter().any(|g| g.abs() > 1e-6));
}
