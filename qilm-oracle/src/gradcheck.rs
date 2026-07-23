//! gradcheck — central finite differences vs. a hand-derived analytic gradient (T0.2).
//!
//! The single highest-value test in the project (VERIFICATION.md §3): every
//! kernel with trainable parameters must have its analytic backward pass
//! checked against `(L(theta+eps) - L(theta-eps)) / (2*eps)`. Disagreement
//! means the kernel's gradient is wrong, full stop — do not weaken the
//! `max_rel_err < 1e-4` bar to make a call site pass.

/// Central finite-difference gradient check.
///
/// `f` computes the scalar loss for a parameter vector; `analytic_grad` computes
/// the hand-derived gradient for the same parameter vector. Returns the maximum
/// relative error, over all parameters, between the analytic gradient and the
/// central finite difference `(f(theta+eps*e_i) - f(theta-eps*e_i)) / (2*eps)`.
///
/// Relative error uses `|analytic - numeric| / max(|analytic|, |numeric|, floor)`
/// with a small floor so that two near-zero gradients (both ~0) don't produce a
/// spurious 0/0 blowup.
pub fn gradcheck<F, G>(f: F, analytic_grad: G, params: &[f64], eps: f64) -> f64
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    const REL_FLOOR: f64 = 1e-8;

    let analytic = analytic_grad(params);
    assert_eq!(
        analytic.len(),
        params.len(),
        "analytic_grad must return one entry per parameter"
    );

    let mut max_rel: f64 = 0.0;
    for i in 0..params.len() {
        let mut plus = params.to_vec();
        let mut minus = params.to_vec();
        plus[i] += eps;
        minus[i] -= eps;

        let numeric = (f(&plus) - f(&minus)) / (2.0 * eps);
        let a = analytic[i];

        let denom = a.abs().max(numeric.abs()).max(REL_FLOOR);
        let rel = (a - numeric).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    max_rel
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L(x) = x^2, dL/dx = 2x. Sanity-checks the harness itself before it's
    /// trusted for real kernels: a correct analytic gradient must pass, and a
    /// deliberately wrong one (off by a constant factor) must fail.
    #[test]
    fn gradcheck_sanity_quadratic() {
        let f = |p: &[f64]| p[0] * p[0];
        let correct = |p: &[f64]| vec![2.0 * p[0]];
        let wrong = |p: &[f64]| vec![3.0 * p[0]]; // deliberately broken

        let ok = gradcheck(f, correct, &[1.7], 1e-4);
        assert!(ok < 1e-4, "correct analytic grad should pass: {ok}");

        let bad = gradcheck(f, wrong, &[1.7], 1e-4);
        assert!(bad > 1e-2, "wrong analytic grad should fail loudly: {bad}");
    }
}
