//! bind — circular convolution + FFT bind/unbind, Proposition 0 (T0.3).
//!
//! `bind` (holographic binding) is circular convolution, computed efficiently
//! via the convolution theorem: `circular_conv(a,b) == ifft(fft(a) ⊙ fft(b))`.
//! `unbind` is the standard HRR approximate inverse: circular correlation with
//! `b`, i.e. `ifft(fft(c) ⊙ conj(fft(b)))`, which recovers `a` from `bind(a,b)`
//! to high cosine similarity when `b`'s entries are i.i.d. (so its spectrum is
//! approximately flat and `conj(fft(b)) * fft(b) ≈ |fft(b)|^2` acts like a
//! near-constant deconvolution kernel).
//!
//! `num_complex::Complex64` (rustfft's own type) is used only inside this
//! module at the FFT boundary; it never appears in this crate's public API —
//! everywhere else uses our own `complex::Complex` pair type.

use crate::complex::Complex;
use rustfft::{num_complex::Complex64, FftPlanner};

/// Direct O(d^2) circular convolution, the textbook definition:
/// (a ★ b)[n] = sum_{k=0}^{d-1} a[k] * b[(n-k) mod d].
/// Used only as the independent reference side of Proposition 0 — never as
/// the fast path (that's `bind`/`bind_freq`).
pub fn circular_conv(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "circular_conv requires equal-length inputs");
    let d = a.len();
    let mut out = vec![0.0; d];
    for n in 0..d {
        let mut s = 0.0;
        for k in 0..d {
            let idx = (n + d - k) % d;
            s += a[k] * b[idx];
        }
        out[n] = s;
    }
    out
}

/// Forward DFT of a real signal, returned as our own `Complex` pair type.
/// Unnormalized (X_k = sum_n x_n * exp(-2*pi*i*k*n/N)), matching the standard
/// forward-FFT convention.
pub fn fft(x: &[f64]) -> Complex {
    let n = x.len();
    let mut buf: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(n);
    plan.process(&mut buf);
    let re = buf.iter().map(|c| c.re).collect();
    let im = buf.iter().map(|c| c.im).collect();
    Complex { re, im }
}

/// Inverse DFT, normalized by 1/N, returning the real part. Circular
/// convolution/correlation of real signals always yields a real result up to
/// floating-point noise, so the imaginary part is discarded (checked in
/// debug builds).
pub fn ifft(x: &Complex) -> Vec<f64> {
    let n = x.len();
    let mut buf: Vec<Complex64> = x
        .re
        .iter()
        .zip(x.im.iter())
        .map(|(&r, &i)| Complex64::new(r, i))
        .collect();
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_inverse(n);
    plan.process(&mut buf);
    let scale = 1.0 / (n as f64);

    #[cfg(debug_assertions)]
    {
        let max_im: f64 = buf.iter().map(|c| (c.im * scale).abs()).fold(0.0, f64::max);
        debug_assert!(
            max_im < 1e-6,
            "ifft: non-negligible imaginary residue ({max_im}); caller passed a \
             spectrum that isn't the transform of a real circular conv/corr"
        );
    }

    buf.iter().map(|c| c.re * scale).collect()
}

/// Frequency-domain binding: ifft(fft(a) ⊙ fft(b)). Equal to `circular_conv`
/// (Proposition 0) but O(d log d) instead of O(d^2); this is the fast path
/// used everywhere else in the model.
pub fn bind_freq(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "bind_freq requires equal-length inputs");
    let spectrum = fft(a).mul(&fft(b));
    ifft(&spectrum)
}

/// Alias for the model-facing binding operation (holographic bind = circular
/// convolution via FFT).
pub fn bind(a: &[f64], b: &[f64]) -> Vec<f64> {
    bind_freq(a, b)
}

/// Approximate unbind (circular correlation with `b`): recovers `a` from
/// `bind(a, b)` up to high cosine similarity for random `b`. The conjugate on
/// `fft(b)` is what makes this a correlation (deconvolution) rather than a
/// second convolution — dropping it is the T0.3 catalog mutant.
pub fn unbind(c: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(c.len(), b.len(), "unbind requires equal-length inputs");
    let spectrum = fft(c).mul(&fft(b).conj());
    ifft(&spectrum)
}
