//! bpb — bits per byte (T1.4), the common currency both the pattern arm and
//! the byte baseline are scored in for the G0 feasibility gate. BPB is the mean
//! `-log2 p(true next byte)` over HELD-OUT positions; a model that peeks at the
//! bytes it is scored on would report an impossibly low BPB, which `nc_bpb`'s
//! untrained-model control catches (threat T3).
//!
//! This function is deliberately model-agnostic: it takes the per-position
//! next-byte log-probabilities a model already produced and the true targets,
//! so the same metric scores the softmax baseline and the Born-head pattern
//! model identically (METRICS-AND-GATES.md §1).

use std::f64::consts::LN_2;

/// Bits per byte over `targets.len()` held-out positions. `logp` is row-major
/// `(n_positions × vocab)` NATURAL-log probabilities — each row a valid log
/// distribution (e.g. a `log_softmax` output) — and `targets[t]` is the true
/// byte at position `t`. Returns `-(1/N) Σ_t log2 p(target_t)`.
///
/// Panics on shape/target mismatch (a silent mismatch could hide a peeking
/// eval; better to fail loudly).
pub fn bpb(logp: &[f64], vocab: usize, targets: &[usize]) -> f64 {
    let n = targets.len();
    assert!(n > 0, "bpb: no positions to score");
    assert_eq!(
        logp.len(),
        n * vocab,
        "bpb: logp length {} != n*vocab {}",
        logp.len(),
        n * vocab
    );
    let mut acc = 0.0;
    for (t, &y) in targets.iter().enumerate() {
        assert!(y < vocab, "bpb: target byte {y} out of range 0..{vocab}");
        // -log2 p = -ln p / ln 2.
        acc += -logp[t * vocab + y] / LN_2;
    }
    acc / n as f64
}

/// True iff two parameter counts are within `tol_frac` of each other, measured
/// against the larger (the `nc_param_match` control uses `tol_frac = 0.05`).
/// The two model arms of G0 must be param-matched so a BPB gap reflects the
/// mechanism, not a size advantage (C6/C7).
pub fn params_within(a: usize, b: usize, tol_frac: f64) -> bool {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    if hi == 0 {
        return lo == 0;
    }
    (hi - lo) as f64 / hi as f64 <= tol_frac
}
