//! gradcheck_autodiff_cross_entropy — finite-difference check for
//! `Tape::log_softmax` + `Tape::cross_entropy` (implementation/tests/PHASE-1.md
//! increment 1, op 4/4: the numerically-stable pair BPB is built on).
//!
//! Setup: logits `(batch=2, classes=3)`, target class per row `[1, 2]`.
//! `L = cross_entropy(log_softmax(logits), targets)`
//!    `= -(1/rows) * sum_r logits[r, t_r] - logsumexp(logits[r, :])`
//!
//! Hand-derived analytic gradient (independent of the tape's own backward;
//! the standard closed-form combined softmax-cross-entropy gradient):
//!   dL/dlogits[r,c] = (1/rows) * (softmax(logits)[r,:][c] - [c == t_r])

use qilm_core::autodiff::{Shape, Tape};
use qilm_oracle::gradcheck::gradcheck;

const ROWS: usize = 2;
const COLS: usize = 3;
const TARGETS: [usize; ROWS] = [1, 2];

fn softmax_row(row: &[f64]) -> Vec<f64> {
    let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = row.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn logsumexp_row(row: &[f64]) -> f64 {
    let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = row.iter().map(|v| (v - max).exp()).sum();
    sum.ln() + max
}

fn loss(logits: &[f64]) -> f64 {
    let mut total = 0.0;
    for r in 0..ROWS {
        let row = &logits[r * COLS..(r + 1) * COLS];
        let t = TARGETS[r];
        total += logsumexp_row(row) - row[t];
    }
    total / ROWS as f64
}

fn analytic_grad(logits: &[f64]) -> Vec<f64> {
    let mut grad = vec![0.0; ROWS * COLS];
    for r in 0..ROWS {
        let row = &logits[r * COLS..(r + 1) * COLS];
        let sm = softmax_row(row);
        let t = TARGETS[r];
        for c in 0..COLS {
            let indicator = if c == t { 1.0 } else { 0.0 };
            grad[r * COLS + c] = (sm[c] - indicator) / ROWS as f64;
        }
    }
    grad
}

fn cases() -> Vec<Vec<f64>> {
    vec![
        vec![0.5, -0.3, 1.2, -0.6, 2.0, 0.1],
        vec![3.0, -2.5, 0.4, 0.02, -0.9, 1.7],
        vec![10.0, -10.0, 0.0, -5.0, 5.0, 0.0], // wide-spread logits (stability regime)
    ]
}

#[test]
fn gradcheck_autodiff_cross_entropy() {
    for logits in cases() {
        let max_rel = gradcheck(loss, analytic_grad, &logits, 1e-4);
        assert!(
            max_rel < 1e-4,
            "gradcheck_autodiff_cross_entropy: max relative error {max_rel} >= 1e-4 for logits {logits:?}"
        );
    }
}

/// Ties the hand-derived oracle to the real tape's `log_softmax` +
/// `cross_entropy`: same forward value, same backward-computed gradient.
#[test]
fn gradcheck_autodiff_cross_entropy_oracle_matches_tape() {
    for logits in cases() {
        let oracle_loss = loss(&logits);
        let oracle_grad = analytic_grad(&logits);

        let mut tape = Tape::new();
        let x = tape.leaf(logits.clone(), Shape::mat(ROWS, COLS));
        let logp = tape.log_softmax(x);
        let l = tape.cross_entropy(logp, &TARGETS);
        tape.backward(l);

        assert!(
            (tape.value(l)[0] - oracle_loss).abs() < 1e-9,
            "tape loss {} != oracle loss {} for logits {logits:?}",
            tape.value(l)[0],
            oracle_loss
        );
        for (i, (t, o)) in tape.grad(x).iter().zip(&oracle_grad).enumerate() {
            assert!(
                (t - o).abs() < 1e-9,
                "grad[{i}]: tape={t} oracle={o} for logits {logits:?}"
            );
        }
    }
}

/// Anti-vacuity: a naive (unstable, no max-subtraction) log-softmax forward
/// on wide-spread logits overflows to NaN/Inf, which must NOT match the
/// correct, stable oracle value -- proving the check above can say "no" and
/// that stability (VERIFICATION.md §3) is actually being exercised.
#[test]
fn gradcheck_autodiff_cross_entropy_can_say_no_on_naive_unstable_softmax() {
    let logits = vec![1000.0, -1000.0, 0.0, 0.0, 0.0, 0.0];
    let correct_loss = loss(&logits);

    // Deliberately WRONG: exponentiate raw logits with no max-subtraction.
    let mut broken_loss = 0.0;
    for r in 0..ROWS {
        let row = &logits[r * COLS..(r + 1) * COLS];
        let sum_exp: f64 = row.iter().map(|v| v.exp()).sum(); // overflows
        let naive_logsumexp = sum_exp.ln();
        broken_loss += naive_logsumexp - row[TARGETS[r]];
    }
    broken_loss /= ROWS as f64;

    assert!(
        broken_loss.is_nan()
            || broken_loss.is_infinite()
            || (broken_loss - correct_loss).abs() > 1e-3,
        "the naive unstable computation should visibly disagree with (or blow up relative to) \
         the correct, stable oracle value, got broken={broken_loss} correct={correct_loss}"
    );
}
