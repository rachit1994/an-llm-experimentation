//! gradcheck_pattern — T1.1 done-when: finite-difference the pattern model's
//! FULL byte cross-entropy loss (context → encode → predict → Born head → CE)
//! against the tape's backprop, max rel err < 1e-4. The numeric oracle (central
//! finite differences in qilm_oracle::gradcheck) is fully independent of the
//! tape's backward pass, so a wrong local derivative anywhere in the model's
//! graph fails this. A shape test alone is NOT the DoD (threat T1); this is.

use qilm_oracle::gradcheck::gradcheck;
use qilm_train::model_pattern::PatternModel;

fn model() -> PatternModel {
    // Small dims keep the finite-difference sweep fast; the graph structure is
    // identical to the 256-vocab model.
    PatternModel::new(5, 3, 4, 3)
}

/// Deterministic, smooth, nonzero parameter init (no RNG dependency): values in
/// [-0.4, 0.4], varied, so Born-head amplitudes are O(1) (away from 0 where the
/// squared-log head would be stiff for finite differences).
fn init_params(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.4 * ((i as f64) * 0.7 + 0.3).sin())
        .collect()
}

fn batch() -> (Vec<f64>, usize, Vec<usize>) {
    let m = model();
    let contexts: Vec<&[u8]> = vec![&[0, 1, 2], &[3, 4, 0], &[1, 1, 3], &[2, 4, 4]];
    let targets = vec![3usize, 1, 4, 0];
    let bag = m.bag_from_contexts(&contexts);
    (bag, contexts.len(), targets)
}

#[test]
fn gradcheck_pattern() {
    let m = model();
    let (bag, b, targets) = batch();
    let params = init_params(m.num_params());

    let loss = |p: &[f64]| m.byte_ce_loss_and_grad(p, &bag, b, &targets).0;
    let analytic = |p: &[f64]| m.byte_ce_loss_and_grad(p, &bag, b, &targets).1;

    let max_rel = gradcheck(loss, analytic, &params, 1e-4);
    assert!(
        max_rel < 1e-4,
        "gradcheck_pattern: full-loss max relative error {max_rel} >= 1e-4"
    );
}

/// Anti-vacuity: a corrupted gradient (drop the head's contribution by zeroing
/// the last block) must NOT match the finite-difference gradient — proving the
/// gradcheck above can say "no", not just rubber-stamp any vector.
#[test]
fn gradcheck_pattern_can_say_no() {
    let m = model();
    let (bag, b, targets) = batch();
    let params = init_params(m.num_params());

    let loss = |p: &[f64]| m.byte_ce_loss_and_grad(p, &bag, b, &targets).0;
    let wrong = |p: &[f64]| {
        let mut g = m.byte_ce_loss_and_grad(p, &bag, b, &targets).1;
        // Corrupt: zero the b_head block (the last `vocab` entries).
        let n = g.len();
        for gi in g.iter_mut().skip(n - m.vocab) {
            *gi = 0.0;
        }
        g
    };
    let max_rel = gradcheck(loss, wrong, &params, 1e-4);
    assert!(
        max_rel > 1e-4,
        "a corrupted gradient must fail the check (got max rel {max_rel})"
    );
}

#[test]
fn pattern_num_params_matches_formula() {
    let m = model();
    // vocab·d_emb + d_emb·d_z + d_z + d_z·d_z + d_z + d_z·vocab + vocab
    let expected = 5 * 3 + 3 * 4 + 4 + 4 * 4 + 4 + 4 * 5 + 5;
    assert_eq!(m.num_params(), expected);
}

#[test]
fn pattern_forward_shapes() {
    use qilm_core::autodiff::Tape;
    let m = model();
    let (bag, b, targets) = batch();
    let params = init_params(m.num_params());
    let mut tape = Tape::new();
    let (fwd, _p) = m.forward(&mut tape, &params, &bag, b, &targets);
    assert_eq!(tape.shape(fwd.z).rows, b);
    assert_eq!(tape.shape(fwd.z).cols, m.d_z);
    assert_eq!(tape.shape(fwd.z_hat).cols, m.d_z);
    assert_eq!(tape.value(fwd.byte_ce).len(), 1, "loss is scalar");
}
