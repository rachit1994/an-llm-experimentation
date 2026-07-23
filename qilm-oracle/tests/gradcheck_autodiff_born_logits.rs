//! gradcheck_autodiff_born_logits — finite-difference check for
//! `Tape::born_logits` (implementation/tests/PHASE-1.md T1.1: the Born-rule
//! byte head).
//!
//! `born_logits(a)_i = ln(a_i² + ε)` (ε = BORN_EPS). Two hand-derived oracles,
//! both independent of the tape's own backward:
//!
//!  (1) OP-LEVEL, via the *linear* scalarizer `L = Σ_i born_logits(a)_i` (a
//!      plain sum, so `L` is linear in the op's output and finite differences
//!      isolate `dy/da` with no amplification — a squared scalarizer would make
//!      `ln(a²+ε)²` stiff near small `a` and blow up the FD truncation error):
//!        y_i        = ln(a_i² + ε)
//!        dL/da_i    = dy_i/da_i = 2 a_i / (a_i² + ε)
//!      The tape has only `sum_squares` as a built-in scalarizer, so the
//!      exact oracle-matches-tape check below uses the squared form instead;
//!      that comparison is exact (no FD), so the stiffness is irrelevant there.
//!
//!  (2) END-TO-END BORN NLL (the model-facing head): for one row `a` and target
//!      `t`, `L = cross_entropy(log_softmax(born_logits(a)), [t])`. Because
//!      `softmax(ln(a_i²+ε)) = (a_i²+ε)/Σ_j(a_j²+ε)` is the Born distribution,
//!        L         = ln s − ln(a_t² + ε),   s = Σ_j (a_j² + ε)
//!        dL/da_i   = 2 a_i / s − [2 a_i / (a_i² + ε)] · 1{i = t}
//!      This is the check that the pattern model's BPB path is differentiated
//!      correctly, not just the isolated op.

use qilm_core::autodiff::{Shape, Tape, BORN_EPS};
use qilm_oracle::gradcheck::gradcheck;

// ---------- (1) op-level ----------

/// Linear scalarizer L = Σ ln(a²+ε) and its gradient dy/da = 2a/(a²+ε).
/// Used for the finite-difference check (well-conditioned near small a).
fn lin_loss(a: &[f64]) -> f64 {
    a.iter().map(|v| (v * v + BORN_EPS).ln()).sum()
}

fn lin_grad(a: &[f64]) -> Vec<f64> {
    a.iter().map(|v| 2.0 * v / (v * v + BORN_EPS)).collect()
}

/// Squared scalarizer L = Σ (ln(a²+ε))² and its gradient. Used only for the
/// EXACT (non-FD) oracle-matches-tape check, because `sum_squares` is the tape's
/// only built-in scalarizer.
fn sq_loss(a: &[f64]) -> f64 {
    a.iter().map(|v| (v * v + BORN_EPS).ln().powi(2)).sum()
}

fn sq_grad(a: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|v| {
            let y = (v * v + BORN_EPS).ln();
            2.0 * y * (2.0 * v / (v * v + BORN_EPS))
        })
        .collect()
}

fn cases() -> Vec<Vec<f64>> {
    vec![
        vec![0.7, -0.3, 1.5, 0.9],
        vec![-2.0, 0.05, 3.1, -1.1], // includes a small amplitude to exercise the ε floor
        vec![1.0, -1.0, 0.2, 4.0],
    ]
}

#[test]
fn gradcheck_autodiff_born_logits() {
    for a in cases() {
        let max_rel = gradcheck(lin_loss, lin_grad, &a, 1e-4);
        assert!(
            max_rel < 1e-4,
            "gradcheck_autodiff_born_logits: max relative error {max_rel} >= 1e-4 for a {a:?}"
        );
    }
}

/// Ties the hand-derived oracle to the real tape: same forward value, same
/// backward-computed gradient, via the squared scalarizer (`sum_squares`). This
/// is an EXACT comparison (no finite differences), so it covers the small-`a`
/// regime the FD check keeps out of the stiff squared loss.
#[test]
fn gradcheck_autodiff_born_logits_oracle_matches_tape() {
    for a in cases() {
        let oracle_loss = sq_loss(&a);
        let oracle_grad = sq_grad(&a);

        let mut tape = Tape::new();
        let leaf = tape.leaf(a.clone(), Shape::row(a.len()));
        let y = tape.born_logits(leaf);
        let sq = tape.sum_squares(y);
        tape.backward(sq);

        assert!(
            (tape.value(sq)[0] - oracle_loss).abs() < 1e-9,
            "tape loss {} != oracle loss {} for a {a:?}",
            tape.value(sq)[0],
            oracle_loss
        );
        for (i, (t, o)) in tape.grad(leaf).iter().zip(&oracle_grad).enumerate() {
            assert!(
                (t - o).abs() < 1e-9,
                "grad[{i}]: tape={t} oracle={o} for a {a:?}"
            );
        }
    }
}

/// Anti-vacuity: the "forgot the chain rule" bug — using `dy/da = 1` (i.e.
/// treating `born_logits` as the identity) instead of `2a/(a²+ε)` — must
/// visibly disagree with the tape's real gradient, proving the check can say
/// "no". (AGENTS.md Rule 2.)
#[test]
fn gradcheck_autodiff_born_logits_can_say_no_on_identity_derivative() {
    let a = vec![0.7, -1.3, 2.0];

    let mut tape = Tape::new();
    let leaf = tape.leaf(a.clone(), Shape::row(a.len()));
    let y = tape.born_logits(leaf);
    let sq = tape.sum_squares(y);
    tape.backward(sq);
    let real_grad = tape.grad(leaf).to_vec();

    // WRONG: dL/da_i = 2*y_i * 1 (missing the 2a/(a²+ε) inner derivative).
    let wrong_grad: Vec<f64> = a.iter().map(|v| 2.0 * (v * v + BORN_EPS).ln()).collect();

    let mut max_abs_diff: f64 = 0.0;
    for (r, w) in real_grad.iter().zip(&wrong_grad) {
        max_abs_diff = max_abs_diff.max((r - w).abs());
    }
    assert!(
        max_abs_diff > 0.1,
        "the identity-derivative gradient should visibly disagree with the tape's real gradient, \
         got max abs diff {max_abs_diff}"
    );
}

// ---------- (2) end-to-end Born NLL head ----------

fn born_nll_loss(a: &[f64], t: usize) -> f64 {
    let s: f64 = a.iter().map(|v| v * v + BORN_EPS).sum();
    s.ln() - (a[t] * a[t] + BORN_EPS).ln()
}

fn born_nll_grad(a: &[f64], t: usize) -> Vec<f64> {
    let s: f64 = a.iter().map(|v| v * v + BORN_EPS).sum();
    a.iter()
        .enumerate()
        .map(|(i, &ai)| {
            let mut g = 2.0 * ai / s;
            if i == t {
                g -= 2.0 * ai / (ai * ai + BORN_EPS);
            }
            g
        })
        .collect()
}

#[test]
fn gradcheck_autodiff_born_nll_head() {
    // amplitude vector (one row) + target byte
    let head_cases: Vec<(Vec<f64>, usize)> = vec![
        (vec![0.7, -0.3, 1.5, 0.9], 2),
        (vec![-2.0, 0.05, 3.1, -1.1, 0.4], 0),
        (vec![1.0, -1.0, 0.2, 4.0], 3),
    ];
    for (a, t) in head_cases {
        let f = |p: &[f64]| born_nll_loss(p, t);
        let g = |p: &[f64]| born_nll_grad(p, t);
        let max_rel = gradcheck(f, g, &a, 1e-4);
        assert!(
            max_rel < 1e-4,
            "gradcheck_autodiff_born_nll_head: max rel err {max_rel} >= 1e-4 for a {a:?}, t {t}"
        );

        // And the hand oracle must match the real tape's forward + backward.
        let mut tape = Tape::new();
        let leaf = tape.leaf(a.clone(), Shape::row(a.len()));
        let logits = tape.born_logits(leaf);
        let logp = tape.log_softmax(logits);
        let loss = tape.cross_entropy(logp, &[t]);
        tape.backward(loss);

        assert!(
            (tape.value(loss)[0] - born_nll_loss(&a, t)).abs() < 1e-9,
            "tape Born-NLL {} != oracle {} for a {a:?}, t {t}",
            tape.value(loss)[0],
            born_nll_loss(&a, t)
        );
        for (i, (tg, og)) in tape
            .grad(leaf)
            .iter()
            .zip(&born_nll_grad(&a, t))
            .enumerate()
        {
            assert!(
                (tg - og).abs() < 1e-9,
                "born-nll grad[{i}]: tape={tg} oracle={og} for a {a:?}, t {t}"
            );
        }
    }
}
