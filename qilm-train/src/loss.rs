//! loss — the Phase-1 training objective (T1.2):
//!   L = λ_byte·L_byte_ce + λ_pat·L_pattern + λ_inv·L_inv + λ_var·anti_collapse
//! composed as ONE tape graph so the whole thing gradchecks end-to-end (T1.1),
//! with the C6 switches that the negative controls flip.
//!
//! Terms (all on the encoder representation z / predicted next pattern ẑ):
//!
//! - `L_byte_ce`: Born-head next-byte cross-entropy (drives BPB).
//! - `L_pattern`: JEPA prediction ‖ẑ − target‖². `target` is the encoder's z on
//!   the NEXT context; with stop-gradient (default) it is frozen to a constant so
//!   the predictor can't win by collapsing the target branch. `no_stopgrad`
//!   connects it instead — the collapse hole `nc_no_stopgrad_collapses` exploits.
//! - `L_inv`: invariance ‖z(view₁) − z(view₂)‖² between two augmented views.
//!   Dropped by `no_invariance`.
//! - `anti_collapse`: the regularizer. `jepa_vicreg` = VICReg variance hinge
//!   (Bardes et al. 2022), per-dim `Σ_j relu(1 − Var(z_j))`, which pushes every
//!   dimension to unit variance and is what keeps erank/meanstd up (H3).
//!   (`infonce`/`vq` are the other C6 ablation arms — not yet wired.)
//!
//! Stop-gradient note for gradcheck: the frozen target makes the analytic
//! gradient (which ignores the target branch) disagree with a naive finite
//! difference (which would recompute the target as params move). So the
//! end-to-end gradcheck uses `no_stopgrad = true` (fully connected, every path
//! differentiable); the stop-gradient path is exercised behaviorally by
//! `test_loss_switches` (it must change the GRADIENT while leaving the loss
//! value unchanged).

use crate::model_pattern::PatternModel;
use qilm_core::autodiff::{NodeId, Shape, Tape};

/// Anti-collapse mechanism (C6). Only `JepaVicreg` is wired for the Phase-1 G0
/// feasibility run; `InfoNce`/`Vq` are the planned ablation arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regularizer {
    JepaVicreg,
}

/// Loss weights + switches. Weights default to a byte-CE-led objective with a
/// VICReg-strength variance term; the negative controls flip the booleans.
#[derive(Debug, Clone, Copy)]
pub struct LossConfig {
    pub lambda_byte: f64,
    pub lambda_pattern: f64,
    pub lambda_inv: f64,
    pub lambda_var: f64,
    pub no_invariance: bool,
    pub no_stopgrad: bool,
    pub regularizer: Regularizer,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            lambda_byte: 1.0,
            lambda_pattern: 1.0,
            lambda_inv: 1.0,
            lambda_var: 1.0,
            no_invariance: false,
            no_stopgrad: false,
            regularizer: Regularizer::JepaVicreg,
        }
    }
}

/// One training batch: the context bag, the next-context bag (JEPA target), a
/// second augmented-view bag (invariance), and the true next bytes. Each bag is
/// row-major `(batch × vocab)` (see `PatternModel::bag_from_contexts`).
pub struct Batch<'a> {
    pub batch: usize,
    pub bag_ctx: &'a [f64],
    pub bag_next: &'a [f64],
    pub bag_view2: &'a [f64],
    pub targets: &'a [usize],
}

/// `sum_squares(a − b)` as a scalar tape node (a,b same shape).
fn sq_diff(tape: &mut Tape, a: NodeId, b: NodeId) -> NodeId {
    let neg_b = tape.scale(b, -1.0);
    let diff = tape.add(a, neg_b);
    tape.sum_squares(diff)
}

/// VICReg variance hinge on `z` (batch × d): `Σ_j relu(1 − Var(z_j))`, a scalar.
fn variance_hinge(tape: &mut Tape, z: NodeId, d: usize) -> NodeId {
    let mean = tape.row_mean(z); // (1 × d)
    let neg_mean = tape.scale(mean, -1.0);
    let centered = tape.add(z, neg_mean); // (batch × d), mean broadcast over rows
    let sq = tape.hadamard(centered, centered);
    let var = tape.row_mean(sq); // (1 × d) population variance per dim
    let neg_var = tape.scale(var, -1.0);
    let ones = tape.leaf(vec![1.0; d], Shape::row(d)); // target variance γ² = 1
    let gap = tape.add(ones, neg_var); // 1 − var_j
    let hinge = tape.relu(gap);
    let ones_col = tape.leaf(vec![1.0; d], Shape::mat(d, 1));
    tape.matmul(hinge, ones_col) // Σ_j hinge_j → (1 × 1)
}

/// Build the full loss graph on `tape`, returning the scalar loss node and the
/// parameter leaves (for grad readout).
fn build(
    tape: &mut Tape,
    model: &PatternModel,
    cfg: &LossConfig,
    params: &[f64],
    b: &Batch,
) -> (NodeId, crate::model_pattern::ParamLeaves) {
    let l = model.leaves(tape, params);

    // Context path → byte-CE.
    let bag_ctx = model.bag_leaf(tape, b.bag_ctx, b.batch);
    let z_ctx = model.encode_z(tape, &l, bag_ctx);
    let z_hat = model.predict(tape, &l, z_ctx);
    let byte_ce = model.born_ce(tape, &l, z_hat, b.targets);

    // JEPA prediction: target = z(next context), frozen unless no_stopgrad.
    let bag_next = model.bag_leaf(tape, b.bag_next, b.batch);
    let z_next = model.encode_z(tape, &l, bag_next);
    let target = if cfg.no_stopgrad {
        z_next
    } else {
        let v = tape.value(z_next).to_vec();
        let shape = tape.shape(z_next);
        tape.leaf(v, shape) // detached constant
    };
    let l_pattern = sq_diff(tape, z_hat, target);

    // Weighted sum, starting from byte-CE + λ_pat·L_pattern.
    let mut total = tape.scale(byte_ce, cfg.lambda_byte);
    let wp = tape.scale(l_pattern, cfg.lambda_pattern);
    total = tape.add(total, wp);

    // Invariance between two views (optional).
    if !cfg.no_invariance {
        let bag_v2 = model.bag_leaf(tape, b.bag_view2, b.batch);
        let z_v2 = model.encode_z(tape, &l, bag_v2);
        let l_inv = sq_diff(tape, z_ctx, z_v2);
        let wi = tape.scale(l_inv, cfg.lambda_inv);
        total = tape.add(total, wi);
    }

    // Anti-collapse regularizer.
    let anti = match cfg.regularizer {
        Regularizer::JepaVicreg => variance_hinge(tape, z_ctx, model.d_z),
    };
    let wv = tape.scale(anti, cfg.lambda_var);
    total = tape.add(total, wv);

    (total, l)
}

/// Forward + backward of the full loss: returns `(loss, grads)` where `grads`
/// is one entry per model parameter in layout order.
pub fn total_loss_and_grad(
    model: &PatternModel,
    cfg: &LossConfig,
    params: &[f64],
    b: &Batch,
) -> (f64, Vec<f64>) {
    let mut tape = Tape::new();
    let (total, l) = build(&mut tape, model, cfg, params, b);
    let loss = tape.value(total)[0];
    tape.backward(total);
    (loss, model.grads_in_order(&tape, &l))
}

/// Just the loss value (no backward) — for finite differences.
pub fn total_loss(model: &PatternModel, cfg: &LossConfig, params: &[f64], b: &Batch) -> f64 {
    let mut tape = Tape::new();
    let (total, _l) = build(&mut tape, model, cfg, params, b);
    tape.value(total)[0]
}
