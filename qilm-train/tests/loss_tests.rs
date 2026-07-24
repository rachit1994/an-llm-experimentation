//! loss_tests — T1.2 done-when: (1) the FULL loss gradchecks end-to-end
//! (finite-diff vs backprop < 1e-4), and (2) `test_loss_switches` — every C6
//! switch changes the computed loss (or, for stop-gradient, the gradient)
//! deterministically. The gradcheck runs with `no_stopgrad = true` so every
//! path is differentiable (the frozen target of the default config would make a
//! naive finite difference disagree with the correct stop-gradient analytic).

use qilm_oracle::gradcheck::gradcheck;
use qilm_train::loss::{total_loss, total_loss_and_grad, Batch, LossConfig, Regularizer};
use qilm_train::model_pattern::PatternModel;

fn model() -> PatternModel {
    PatternModel::new(5, 3, 4, 3)
}

fn init_params(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.4 * ((i as f64) * 0.7 + 0.3).sin())
        .collect()
}

// Three bags (context / next / view2) + targets, built once.
fn make_bags(m: &PatternModel) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>, usize) {
    let ctx: Vec<&[u8]> = vec![&[0, 1, 2], &[3, 4, 0], &[1, 1, 3], &[2, 4, 4]];
    let next: Vec<&[u8]> = vec![&[1, 2, 3], &[4, 0, 1], &[1, 3, 2], &[4, 4, 0]];
    let view2: Vec<&[u8]> = vec![&[0, 2, 1], &[3, 0, 4], &[1, 3, 1], &[2, 2, 4]];
    let targets = vec![3usize, 1, 4, 0];
    (
        m.bag_from_contexts(&ctx),
        m.bag_from_contexts(&next),
        m.bag_from_contexts(&view2),
        targets,
        ctx.len(),
    )
}

#[test]
fn gradcheck_full_loss() {
    let m = model();
    let (bc, bn, bv, targets, batch) = make_bags(&m);
    let params = init_params(m.num_params());
    // Fully connected (no stop-grad) so finite differences and backprop agree.
    let cfg = LossConfig {
        no_stopgrad: true,
        ..LossConfig::default()
    };
    let b = Batch {
        batch,
        bag_ctx: &bc,
        bag_next: &bn,
        bag_view2: &bv,
        targets: &targets,
    };

    let loss = |p: &[f64]| total_loss(&m, &cfg, p, &b);
    let grad = |p: &[f64]| total_loss_and_grad(&m, &cfg, p, &b).1;

    let max_rel = gradcheck(loss, grad, &params, 1e-4);
    assert!(
        max_rel < 1e-4,
        "gradcheck_full_loss: max relative error {max_rel} >= 1e-4"
    );
}

/// A corrupted gradient must fail the check — proving the gradcheck above can
/// say "no" (Rule 2).
#[test]
fn gradcheck_full_loss_can_say_no() {
    let m = model();
    let (bc, bn, bv, targets, batch) = make_bags(&m);
    let params = init_params(m.num_params());
    let cfg = LossConfig {
        no_stopgrad: true,
        ..LossConfig::default()
    };
    let b = Batch {
        batch,
        bag_ctx: &bc,
        bag_next: &bn,
        bag_view2: &bv,
        targets: &targets,
    };
    let loss = |p: &[f64]| total_loss(&m, &cfg, p, &b);
    let wrong = |p: &[f64]| {
        let mut g = total_loss_and_grad(&m, &cfg, p, &b).1;
        g.iter_mut().for_each(|x| *x *= 0.5); // scale every grad — must disagree
        g
    };
    assert!(gradcheck(loss, wrong, &params, 1e-4) > 1e-4);
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn test_loss_switches() {
    let m = model();
    let (bc, bn, bv, targets, batch) = make_bags(&m);
    let params = init_params(m.num_params());
    let b = Batch {
        batch,
        bag_ctx: &bc,
        bag_next: &bn,
        bag_view2: &bv,
        targets: &targets,
    };

    let base = LossConfig::default(); // stop-grad ON, invariance ON
    let (l_base, g_base) = total_loss_and_grad(&m, &base, &params, &b);

    // no_invariance drops L_inv → the loss VALUE changes.
    let cfg_noinv = LossConfig {
        no_invariance: true,
        ..base
    };
    let (l_noinv, _) = total_loss_and_grad(&m, &cfg_noinv, &params, &b);
    assert!(
        (l_noinv - l_base).abs() > 1e-9,
        "no_invariance must change the loss value ({l_base} vs {l_noinv})"
    );

    // no_stopgrad leaves the loss VALUE identical (same forward) but changes the
    // GRADIENT (the target branch now backpropagates).
    let cfg_nosg = LossConfig {
        no_stopgrad: true,
        ..base
    };
    let (l_nosg, g_nosg) = total_loss_and_grad(&m, &cfg_nosg, &params, &b);
    assert!(
        (l_nosg - l_base).abs() < 1e-9,
        "no_stopgrad must NOT change the loss value ({l_base} vs {l_nosg})"
    );
    assert!(
        max_abs_diff(&g_base, &g_nosg) > 1e-9,
        "no_stopgrad must change the gradient (stop-gradient path)"
    );

    // A weight change scales its term → the loss VALUE changes.
    let cfg_lam = LossConfig {
        lambda_byte: 2.0,
        ..base
    };
    let (l_lam, _) = total_loss_and_grad(&m, &cfg_lam, &params, &b);
    assert!(
        (l_lam - l_base).abs() > 1e-9,
        "changing lambda_byte must change the loss value"
    );

    // Sanity: the default regularizer is the wired one.
    assert_eq!(base.regularizer, Regularizer::JepaVicreg);
}
