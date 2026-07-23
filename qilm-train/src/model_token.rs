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
