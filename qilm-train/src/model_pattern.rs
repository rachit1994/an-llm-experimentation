//! model_pattern — the Born-head next-byte pattern model (T1.1), the arm whose
//! feasibility G0 tests against the byte-softmax baseline. Forward is built in
//! the autodiff tape (qilm_core), so the SAME code path trains (backprop) and
//! evaluates. The byte head is the Born rule expressed via `born_logits`
//! (p_i = a_i²/Σa_j² = softmax(ln a_i²)); this file wires
//!   bag context → encode → predict next pattern → Born head → byte cross-entropy.
//!
//! Parameters live in one flat `&[f64]` in this order:
//!   [ E: vocab·d_emb ][ W_enc: d_emb·d_z ][ b_enc: d_z ]
//!   [ W_pred: d_z·d_z ][ b_pred: d_z ][ W_head: d_z·vocab ][ b_head: vocab ]
//!
//! This increment implements the model forward + the byte cross-entropy loss
//! and gradchecks it end-to-end. The invariance / anti-collapse loss terms
//! (T1.2) layer on top of the `z`/`ẑ` this exposes.

use qilm_core::autodiff::{NodeId, Shape, Tape};
use qilm_core::encoder::encode;

/// Shape of the pattern model. `vocab` is 256 for raw bytes.
#[derive(Debug, Clone, Copy)]
pub struct PatternModel {
    pub vocab: usize,
    pub d_emb: usize,
    pub d_z: usize,
    pub context_len: usize,
}

/// The tape nodes a forward pass exposes, so loss terms beyond byte-CE (T1.2)
/// can attach to the representations without rebuilding the graph.
pub struct Forward {
    /// encoder output z, shape (batch × d_z)
    pub z: NodeId,
    /// predicted next pattern ẑ, shape (batch × d_z)
    pub z_hat: NodeId,
    /// byte cross-entropy loss node (scalar)
    pub byte_ce: NodeId,
}

impl PatternModel {
    pub fn new(vocab: usize, d_emb: usize, d_z: usize, context_len: usize) -> Self {
        assert!(
            vocab > 0 && d_emb > 0 && d_z > 0 && context_len > 0,
            "PatternModel: zero dim"
        );
        Self {
            vocab,
            d_emb,
            d_z,
            context_len,
        }
    }

    // ---- parameter layout ----
    fn n_e(&self) -> usize {
        self.vocab * self.d_emb
    }
    fn n_wenc(&self) -> usize {
        self.d_emb * self.d_z
    }
    fn n_benc(&self) -> usize {
        self.d_z
    }
    fn n_wpred(&self) -> usize {
        self.d_z * self.d_z
    }
    fn n_bpred(&self) -> usize {
        self.d_z
    }
    fn n_whead(&self) -> usize {
        self.d_z * self.vocab
    }
    fn n_bhead(&self) -> usize {
        self.vocab
    }

    /// Total learnable parameters (hand-derivable — pinned by a test).
    pub fn num_params(&self) -> usize {
        self.n_e()
            + self.n_wenc()
            + self.n_benc()
            + self.n_wpred()
            + self.n_bpred()
            + self.n_whead()
            + self.n_bhead()
    }

    /// Build the `(batch × vocab)` bag matrix from a batch of context windows,
    /// each of length `context_len`: `bag[i][b] = count(b in context_i)/L`.
    pub fn bag_from_contexts(&self, contexts: &[&[u8]]) -> Vec<f64> {
        let mut bag = vec![0.0; contexts.len() * self.vocab];
        let inv = 1.0 / self.context_len as f64;
        for (i, ctx) in contexts.iter().enumerate() {
            assert_eq!(
                ctx.len(),
                self.context_len,
                "bag_from_contexts: bad ctx len"
            );
            for &b in ctx.iter() {
                bag[i * self.vocab + b as usize] += inv;
            }
        }
        bag
    }

    /// Create all parameter leaves on `tape` from the flat `params`, in layout
    /// order, returning their NodeIds.
    fn param_leaves(&self, tape: &mut Tape, params: &[f64]) -> ParamLeaves {
        assert_eq!(
            params.len(),
            self.num_params(),
            "params length {} != num_params {}",
            params.len(),
            self.num_params()
        );
        let mut off = 0;
        let mut take = |tape: &mut Tape, n: usize, shape: Shape| {
            let leaf = tape.leaf(params[off..off + n].to_vec(), shape);
            off += n;
            leaf
        };
        let (v, de, dz) = (self.vocab, self.d_emb, self.d_z);
        ParamLeaves {
            e: take(tape, self.n_e(), Shape::mat(v, de)),
            w_enc: take(tape, self.n_wenc(), Shape::mat(de, dz)),
            b_enc: take(tape, self.n_benc(), Shape::row(dz)),
            w_pred: take(tape, self.n_wpred(), Shape::mat(dz, dz)),
            b_pred: take(tape, self.n_bpred(), Shape::row(dz)),
            w_head: take(tape, self.n_whead(), Shape::mat(dz, v)),
            b_head: take(tape, self.n_bhead(), Shape::row(v)),
        }
    }

    /// Build the full forward graph on `tape` for a batch: bag context →
    /// encode → predict → Born head → byte cross-entropy against `targets`.
    /// Returns the exposed nodes and the parameter leaves (for grad readout).
    pub fn forward(
        &self,
        tape: &mut Tape,
        params: &[f64],
        bag: &[f64],
        batch: usize,
        targets: &[usize],
    ) -> (Forward, ParamLeaves) {
        assert_eq!(bag.len(), batch * self.vocab, "bag shape mismatch");
        assert_eq!(targets.len(), batch, "one target per batch row");
        let p = self.param_leaves(tape, params);
        let bag_node = tape.leaf(bag.to_vec(), Shape::mat(batch, self.vocab));

        let z = encode(tape, bag_node, p.e, p.w_enc, p.b_enc);
        // predict next pattern ẑ = tanh(z·W_pred + b_pred)
        let pre = tape.linear(z, p.w_pred, p.b_pred);
        let z_hat = tape.tanh(pre);
        // Born head: amplitudes a = ẑ·W_head + b_head → born_logits → log_softmax → CE
        let a = tape.linear(z_hat, p.w_head, p.b_head);
        let logits = tape.born_logits(a);
        let logp = tape.log_softmax(logits);
        let byte_ce = tape.cross_entropy(logp, targets);

        (Forward { z, z_hat, byte_ce }, p)
    }

    /// Convenience for gradcheck / a plain byte-CE training step: forward, then
    /// backward from the byte-CE loss, returning `(loss, grads)` where `grads`
    /// is one entry per parameter in layout order.
    pub fn byte_ce_loss_and_grad(
        &self,
        params: &[f64],
        bag: &[f64],
        batch: usize,
        targets: &[usize],
    ) -> (f64, Vec<f64>) {
        let mut tape = Tape::new();
        let (fwd, p) = self.forward(&mut tape, params, bag, batch, targets);
        let loss = tape.value(fwd.byte_ce)[0];
        tape.backward(fwd.byte_ce);
        let mut grads = Vec::with_capacity(self.num_params());
        for leaf in [
            p.e, p.w_enc, p.b_enc, p.w_pred, p.b_pred, p.w_head, p.b_head,
        ] {
            grads.extend_from_slice(tape.grad(leaf));
        }
        (loss, grads)
    }
}

/// Parameter leaf NodeIds in layout order.
pub struct ParamLeaves {
    pub e: NodeId,
    pub w_enc: NodeId,
    pub b_enc: NodeId,
    pub w_pred: NodeId,
    pub b_pred: NodeId,
    pub w_head: NodeId,
    pub b_head: NodeId,
}
