//! model_token — the param-matched byte-softmax BASELINE for G0 (T1.4). A plain
//! next-byte predictor: embed the last `context_len` bytes, mean-pool them into
//! a context vector, project to `vocab` logits, softmax. It is the honest
//! yardstick the Born-head pattern model must match within 10% BPB — so this
//! file holds no clever tricks, only a transparent, param-countable baseline.
//!
//! Parameters live in a single flat `&[f64]` (so `testkit::random_init_model`
//! can fill an untrained one and the tape can own them for training):
//!   [ E: vocab×d ][ W: d×vocab ][ b: vocab ]
//! Forward here is plain arithmetic (evaluation only, no grad); the training
//! forward that threads these through the autodiff tape comes with the G0 run.

use qilm_core::autodiff::{NodeId, Shape, Tape};
use std::f64::consts::LN_2;

/// Shape of the byte baseline. `vocab` is 256 for raw bytes.
#[derive(Debug, Clone, Copy)]
pub struct TokenModel {
    pub vocab: usize,
    pub d: usize,
    pub context_len: usize,
}

impl TokenModel {
    pub fn new(vocab: usize, d: usize, context_len: usize) -> Self {
        assert!(
            vocab > 0 && d > 0 && context_len > 0,
            "TokenModel: zero dim"
        );
        Self {
            vocab,
            d,
            context_len,
        }
    }

    /// Total learnable parameters: `vocab*d` (embedding) + `d*vocab` (output
    /// projection) + `vocab` (bias). Hand-derivable — `nc_param_match` pins it.
    pub fn num_params(&self) -> usize {
        self.vocab * self.d + self.d * self.vocab + self.vocab
    }

    fn e_offset(&self) -> usize {
        0
    }
    fn w_offset(&self) -> usize {
        self.vocab * self.d
    }
    fn b_offset(&self) -> usize {
        self.vocab * self.d + self.d * self.vocab
    }

    /// Natural-log next-byte distribution given a `context` of exactly
    /// `context_len` bytes. Mean-pools the context embeddings, projects to
    /// logits, returns `log_softmax(logits)` (length `vocab`).
    pub fn forward_logp(&self, params: &[f64], context: &[u8]) -> Vec<f64> {
        assert_eq!(
            params.len(),
            self.num_params(),
            "forward_logp: params length {} != num_params {}",
            params.len(),
            self.num_params()
        );
        assert_eq!(
            context.len(),
            self.context_len,
            "forward_logp: context length {} != context_len {}",
            context.len(),
            self.context_len
        );
        let (d, vocab) = (self.d, self.vocab);

        // Mean-pooled context embedding c ∈ R^d.
        let mut c = vec![0.0; d];
        for &byte in context {
            let base = self.e_offset() + (byte as usize) * d;
            for k in 0..d {
                c[k] += params[base + k];
            }
        }
        let inv = 1.0 / context.len() as f64;
        for ck in c.iter_mut() {
            *ck *= inv;
        }

        // logits[j] = Σ_k c[k]·W[k][j] + b[j].
        let w0 = self.w_offset();
        let b0 = self.b_offset();
        let mut logits = vec![0.0; vocab];
        for (j, lj) in logits.iter_mut().enumerate() {
            let mut s = params[b0 + j];
            for k in 0..d {
                s += c[k] * params[w0 + k * vocab + j];
            }
            *lj = s;
        }

        // Numerically-stable log_softmax.
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = logits.iter().map(|&l| (l - max).exp()).sum();
        let log_sum_exp = sum_exp.ln() + max;
        logits.iter().map(|&l| l - log_sum_exp).collect()
    }

    /// Tape parameter leaves `(E, W, b)` from the flat `params`.
    fn leaves(&self, tape: &mut Tape, params: &[f64]) -> (NodeId, NodeId, NodeId) {
        let (v, d) = (self.vocab, self.d);
        let e = tape.leaf(params[..v * d].to_vec(), Shape::mat(v, d));
        let w = tape.leaf(params[v * d..v * d + d * v].to_vec(), Shape::mat(d, v));
        let b = tape.leaf(params[v * d + d * v..].to_vec(), Shape::row(v));
        (e, w, b)
    }

    /// Build the softmax forward on `tape`: bag `(batch × vocab)` → mean-pool
    /// embed `c = bag·E` → `logits = c·W + b` → `log_softmax`. Returns the
    /// log-prob node and the leaves.
    fn forward_tape(
        &self,
        tape: &mut Tape,
        params: &[f64],
        bag: &[f64],
        batch: usize,
    ) -> (NodeId, (NodeId, NodeId, NodeId)) {
        assert_eq!(bag.len(), batch * self.vocab, "bag shape mismatch");
        let (e, w, b) = self.leaves(tape, params);
        let bag_node = tape.leaf(bag.to_vec(), Shape::mat(batch, self.vocab));
        let c = tape.matmul(bag_node, e);
        let logits = tape.linear(c, w, b);
        let logp = tape.log_softmax(logits);
        (logp, (e, w, b))
    }

    /// Cross-entropy loss + gradient (one entry per param, layout order) for a
    /// batch of context bags and next-byte targets. The training forward.
    pub fn ce_loss_and_grad(
        &self,
        params: &[f64],
        bag: &[f64],
        batch: usize,
        targets: &[usize],
    ) -> (f64, Vec<f64>) {
        let mut tape = Tape::new();
        let (logp, (e, w, b)) = self.forward_tape(&mut tape, params, bag, batch);
        let ce = tape.cross_entropy(logp, targets);
        let loss = tape.value(ce)[0];
        tape.backward(ce);
        let mut g = Vec::with_capacity(self.num_params());
        for leaf in [e, w, b] {
            g.extend_from_slice(tape.grad(leaf));
        }
        (loss, g)
    }

    /// Natural-log next-byte distributions for a batch of context bags, row
    /// major `(batch × vocab)` — for BPB evaluation via `metrics::bpb::bpb`.
    pub fn logp_batch(&self, params: &[f64], bag: &[f64], batch: usize) -> Vec<f64> {
        let mut tape = Tape::new();
        let (logp, _l) = self.forward_tape(&mut tape, params, bag, batch);
        tape.value(logp).to_vec()
    }

    /// Convenience: score a byte stream's BPB in bits, sliding a `context_len`
    /// window. Positions `context_len..bytes.len()` are predicted from the
    /// preceding window (held-out: the target byte is never in its own
    /// context). Returns mean `-log2 p(byte_t)`.
    pub fn bpb_on(&self, params: &[f64], bytes: &[u8]) -> f64 {
        assert!(
            bytes.len() > self.context_len,
            "bpb_on: need more than context_len bytes"
        );
        let mut acc = 0.0;
        let mut n = 0usize;
        for t in self.context_len..bytes.len() {
            let context = &bytes[t - self.context_len..t];
            let logp = self.forward_logp(params, context);
            acc += -logp[bytes[t] as usize] / LN_2;
            n += 1;
        }
        acc / n as f64
    }
}
