//! hopfield — modern-Hopfield (dense associative memory) retrieval (T0.6).
//!
//! retrieve(X, xi, beta) = X * softmax(beta * X^T xi)  (Ramsauer et al. 2020):
//! score each stored pattern by its dot product with the query, softmax the
//! scores (sharpness controlled by `beta`), and return the softmax-weighted
//! sum of stored patterns. Because addition is commutative and softmax
//! weights are computed per-pattern independently of storage order, the
//! output is exactly invariant to the order patterns are supplied in.

/// Retrieve from a set of stored patterns given a query `xi` and inverse
/// temperature `beta`. `patterns` is a slice of `N` equal-length vectors;
/// `xi` must have the same length as each pattern.
pub fn retrieve(patterns: &[Vec<f64>], xi: &[f64], beta: f64) -> Vec<f64> {
    assert!(
        !patterns.is_empty(),
        "retrieve: need at least one stored pattern"
    );
    let d = xi.len();
    for p in patterns {
        assert_eq!(
            p.len(),
            d,
            "retrieve: all patterns must match the query dimension"
        );
    }

    let scores: Vec<f64> = patterns.iter().map(|p| beta * dot(p, xi)).collect();
    let weights = softmax(&scores);

    let mut out = vec![0.0; d];
    for (p, w) in patterns.iter().zip(weights.iter()) {
        for j in 0..d {
            out[j] += w * p[j];
        }
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Numerically stable softmax (subtract the max before exponentiating).
fn softmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}
