//! train — plain SGD + corpus batching for the Phase-1 G0 run (T1.5/T1.6).
//! Deterministic (seeded `rand_chacha`, C4): same seed → same trained params.
//!
//! The synthetic Markov corpus is an alphabet of `K` symbols encoded as bytes;
//! `to_symbols` maps the raw byte stream to indices `0..K` by sorted distinct
//! value (encoding-agnostic). A model predicts symbol `t` from a `context_len`
//! window ending at `t-1`; each context is reduced to a bag (see
//! `PatternModel::bag_from_contexts`). For the JEPA term the "next" context is
//! the window ending at `t` (predicting `t+1`); for the invariance term a second
//! view is the context bag lightly mixed toward uniform (input-noise
//! augmentation) — both documented deviations kept simple for the feasibility
//! phase.

use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Map a raw byte stream to symbol indices `0..K` by sorted distinct value.
/// Returns `(symbols, K)`.
pub fn to_symbols(bytes: &[u8]) -> (Vec<usize>, usize) {
    let mut distinct: Vec<u8> = bytes.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let k = distinct.len();
    let symbols = bytes
        .iter()
        .map(|b| distinct.binary_search(b).unwrap())
        .collect();
    (symbols, k)
}

/// SGD hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct SgdConfig {
    pub steps: usize,
    pub lr: f64,
    pub batch: usize,
    pub seed: u64,
}

/// Plain SGD: `params -= lr · grad`, the gradient computed on a freshly-sampled
/// minibatch each step (the closure owns sampling, given the rng).
pub fn sgd<F>(mut params: Vec<f64>, cfg: &SgdConfig, mut grad_fn: F) -> Vec<f64>
where
    F: FnMut(&[f64], &mut ChaCha20Rng) -> Vec<f64>,
{
    let mut rng = ChaCha20Rng::seed_from_u64(cfg.seed);
    for _ in 0..cfg.steps {
        let g = grad_fn(&params, &mut rng);
        for (p, gi) in params.iter_mut().zip(&g) {
            *p -= cfg.lr * gi;
        }
    }
    params
}

/// Adam (Kingma & Ba 2015) with the standard defaults β1=0.9, β2=0.999, ε=1e-8.
/// Adaptive per-parameter step sizes — a much stronger optimizer than plain SGD
/// for the (harder) Born-head objective. Deterministic given `cfg.seed`.
pub fn adam<F>(mut params: Vec<f64>, cfg: &SgdConfig, mut grad_fn: F) -> Vec<f64>
where
    F: FnMut(&[f64], &mut ChaCha20Rng) -> Vec<f64>,
{
    const B1: f64 = 0.9;
    const B2: f64 = 0.999;
    const EPS: f64 = 1e-8;
    let mut rng = ChaCha20Rng::seed_from_u64(cfg.seed);
    let mut m = vec![0.0; params.len()];
    let mut v = vec![0.0; params.len()];
    for step in 1..=cfg.steps {
        let g = grad_fn(&params, &mut rng);
        let bc1 = 1.0 - B1.powi(step as i32);
        let bc2 = 1.0 - B2.powi(step as i32);
        for i in 0..params.len() {
            m[i] = B1 * m[i] + (1.0 - B1) * g[i];
            v[i] = B2 * v[i] + (1.0 - B2) * g[i] * g[i];
            let mhat = m[i] / bc1;
            let vhat = v[i] / bc2;
            params[i] -= cfg.lr * mhat / (vhat.sqrt() + EPS);
        }
    }
    params
}

/// One sampled minibatch, as bags + targets ready for the models.
pub struct BatchData {
    pub batch: usize,
    pub bag_ctx: Vec<f64>,
    pub bag_next: Vec<f64>,
    pub bag_view2: Vec<f64>,
    pub targets: Vec<usize>,
}

/// A symbol corpus over a `vocab`-symbol alphabet.
pub struct Corpus {
    pub symbols: Vec<usize>,
    pub vocab: usize,
}

impl Corpus {
    pub fn new(symbols: Vec<usize>, vocab: usize) -> Self {
        Self { symbols, vocab }
    }

    fn set_bag(&self, bag: &mut [f64], row: usize, window: &[usize]) {
        let inv = 1.0 / window.len() as f64;
        for &s in window {
            bag[row * self.vocab + s] += inv;
        }
    }

    /// Sample `batch` prediction positions and build the three bags + targets
    /// for a model with `context_len`. Positions `t` are drawn from
    /// `[context_len, n-2]` so both the target `t` and the JEPA-next window
    /// (ending at `t`, predicting `t+1`) exist.
    #[allow(clippy::needless_range_loop)] // parallel index into several bag arrays
    pub fn sample(&self, context_len: usize, batch: usize, rng: &mut ChaCha20Rng) -> BatchData {
        let n = self.symbols.len();
        assert!(n > context_len + 2, "corpus too short for context_len");
        let lo = context_len;
        let hi = n - 1; // need t and t (as next-window end) < n
        let span = hi - lo;
        let mut bag_ctx = vec![0.0; batch * self.vocab];
        let mut bag_next = vec![0.0; batch * self.vocab];
        let mut bag_view2 = vec![0.0; batch * self.vocab];
        let mut targets = vec![0usize; batch];
        for i in 0..batch {
            let t = lo + (rng.random::<f64>() * span as f64) as usize % span.max(1);
            let ctx = &self.symbols[t - context_len..t];
            let next = &self.symbols[t - context_len + 1..t + 1];
            self.set_bag(&mut bag_ctx, i, ctx);
            self.set_bag(&mut bag_next, i, next);
            targets[i] = self.symbols[t];
        }
        // view2 = context bag mixed 15% toward uniform (input-noise augmentation).
        let eps = 0.15;
        let unif = eps / self.vocab as f64;
        for i in 0..batch {
            for s in 0..self.vocab {
                bag_view2[i * self.vocab + s] = (1.0 - eps) * bag_ctx[i * self.vocab + s] + unif;
            }
        }
        BatchData {
            batch,
            bag_ctx,
            bag_next,
            bag_view2,
            targets,
        }
    }

    /// Build the full held-out evaluation bags + targets for BPB (every
    /// position `context_len..n`, in order — deterministic, no sampling).
    pub fn eval_bags(&self, context_len: usize) -> (Vec<f64>, usize, Vec<usize>) {
        let n = self.symbols.len();
        let batch = n - context_len;
        let mut bag = vec![0.0; batch * self.vocab];
        let mut targets = vec![0usize; batch];
        for (row, t) in (context_len..n).enumerate() {
            let ctx = &self.symbols[t - context_len..t];
            self.set_bag(&mut bag, row, ctx);
            targets[row] = self.symbols[t];
        }
        (bag, batch, targets)
    }
}

/// Random initial parameters ~ `scale · Uniform(-1, 1)`, seeded (C4).
pub fn init_params(n: usize, scale: f64, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| scale * (2.0 * rng.random::<f64>() - 1.0))
        .collect()
}
