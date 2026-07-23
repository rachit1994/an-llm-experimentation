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
//! The DFT here is a hand-rolled `O(d^2)` implementation (no FFT crate — minimal
//! dependency policy, PRIOR-ART-AND-REUSE.md). At Phase-0 test sizes (`d ~ 256`)
//! this is instant, and `bind_freq` (DFT-multiply-inverse) remains an INDEPENDENT
//! computation from the direct `circular_conv`, so their equality (Proposition 0)
//! is a genuine cross-check, not a self-comparison.

use crate::complex::Complex;
use std::f64::consts::TAU;

/// Direct O(d^2) circular convolution, the textbook definition:
/// (a ★ b)[n] = sum_{k=0}^{d-1} a[k] * b[(n-k) mod d].
/// Used only as the independent reference side of Proposition 0 — never as
/// the fast path (that's `bind`/`bind_freq`).
///
/// Indices are modular (`(n - k) mod d`), so the double loop is written with
/// explicit indices rather than iterator adaptors.
#[allow(clippy::needless_range_loop)]
pub fn circular_conv(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(
        a.len(),
        b.len(),
        "circular_conv requires equal-length inputs"
    );
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
/// forward-FFT convention. Hand-rolled O(d^2) (no FFT crate).
#[allow(clippy::needless_range_loop)]
pub fn fft(x: &[f64]) -> Complex {
    let n = x.len();
    let mut re = vec![0.0; n];
    let mut im = vec![0.0; n];
    for k in 0..n {
        let mut sr = 0.0;
        let mut si = 0.0;
        for j in 0..n {
            let ang = -TAU * (k as f64) * (j as f64) / (n as f64);
            let (s, c) = ang.sin_cos();
            sr += x[j] * c;
            si += x[j] * s;
        }
        re[k] = sr;
        im[k] = si;
    }
    Complex { re, im }
}

/// Inverse DFT, normalized by 1/N, returning the real part. Circular
/// convolution/correlation of real signals always yields a real result up to
/// floating-point noise, so the imaginary part is discarded (checked in
/// debug builds). Hand-rolled O(d^2) (no FFT crate).
#[allow(clippy::needless_range_loop)]
pub fn ifft(x: &Complex) -> Vec<f64> {
    let n = x.re.len();
    let scale = 1.0 / (n as f64);
    let mut out = vec![0.0; n];
    #[cfg(debug_assertions)]
    let mut max_im = 0.0_f64;
    for j in 0..n {
        let mut sr = 0.0;
        let mut si = 0.0;
        for k in 0..n {
            let ang = TAU * (k as f64) * (j as f64) / (n as f64);
            let (s, c) = ang.sin_cos();
            // (Xr + i Xi)(c + i s) = (Xr c - Xi s) + i(Xr s + Xi c)
            sr += x.re[k] * c - x.im[k] * s;
            si += x.re[k] * s + x.im[k] * c;
        }
        out[j] = sr * scale;
        #[cfg(debug_assertions)]
        {
            max_im = max_im.max((si * scale).abs());
        }
    }
    #[cfg(debug_assertions)]
    debug_assert!(
        max_im < 1e-6,
        "ifft: non-negligible imaginary residue ({max_im}); caller passed a \
         spectrum that isn't the transform of a real circular conv/corr"
    );
    out
}

/// Frequency-domain binding: ifft(fft(a) ⊙ fft(b)). Equal to `circular_conv`
/// (Proposition 0). An independent DFT-domain computation from the direct
/// `circular_conv`, so their equality is a real cross-check.
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
