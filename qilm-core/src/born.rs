//! born — Born-rule readout (T0.4).
//!
//! p_k = |<m_k|psi>|^2 / sum_j |<m_j|psi>|^2  (squared-magnitude inner
//! product against each measurement vector, then normalized to a
//! probability distribution — no softmax involved).

use crate::complex::Complex;

/// Born readout: given a state `psi` and a set of measurement vectors,
/// returns the probability of each outcome. Panics if `measurements` is empty
/// or if the total (unnormalized) weight is zero (a degenerate/undefined
/// distribution, not a silently-wrong one).
pub fn born(psi: &Complex, measurements: &[Complex]) -> Vec<f64> {
    assert!(!measurements.is_empty(), "born: need at least one measurement vector");

    let raw: Vec<f64> = measurements
        .iter()
        .map(|m| {
            let (re, im) = m.inner(psi); // <m|psi>
            re * re + im * im // |<m|psi>|^2
        })
        .collect();

    let sum: f64 = raw.iter().sum();
    assert!(sum > 0.0, "born: total measurement weight is zero (degenerate state/measurements)");

    raw.iter().map(|p| p / sum).collect()
}
