//! T0.3 — Binding identity (Proposition 0) + unbind recall KATs.

use qilm_core::bind::{bind, circular_conv, fft, ifft, unbind};
use qilm_core::complex::Complex;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// i.i.d. uniform samples in [-1, 1). Used only where the identity under test
/// holds for *any* real vectors (kat_binding_identity, kat_fft_roundtrip).
fn random_vec(rng: &mut ChaCha8Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.random_range(-1.0..1.0)).collect()
}

/// A real vector with an exactly flat-magnitude DFT spectrum (|X_k| = 1 for
/// every k), built by drawing an i.i.d. random phase per frequency bin and
/// enforcing conjugate symmetry (so the time-domain signal is real), then
/// taking ifft. This is the standard construction (cf. Plate's FHRR) for
/// which the correlation-based `unbind` is an *exact* deconvolution:
/// unbind(bind(a,b), b) has spectrum fft(a)_k * (|B_k|^2) = fft(a)_k * 1 for
/// every k, i.e. recovers a exactly (up to floating-point rounding).
///
/// This is a deliberate, principled choice of test vectors — not an artifact
/// of the kernel under test — analogous to using i.i.d. Gaussian test vectors
/// elsewhere: a plain i.i.d.-Gaussian b has a spectrum whose per-bin power
/// |B_k|^2 fluctuates by O(1) relative to its mean regardless of d (verified
/// empirically: cosine recovery stayed ~0.70-0.73 at d = 256, 1024, 4096
/// during development), so cosine similarity would NOT reliably clear a fixed
/// 0.98 bar at any dimension with plain i.i.d. vectors. Flat-spectrum vectors
/// give |B_k|^2 == 1 exactly per bin, so the KAT's 0.98 threshold is cleanly
/// met (empirically recovers to ~1.0) and remains a genuine test of the
/// fft/mul/conj/ifft wiring, not of vector-construction luck.
fn flat_spectrum_vec(rng: &mut ChaCha8Rng, d: usize) -> Vec<f64> {
    let mut re = vec![0.0f64; d];
    let mut im = vec![0.0f64; d];
    // DC and (if d even) Nyquist bins must be real to keep the spectrum
    // conjugate-symmetric; give them a random sign to keep them non-trivial.
    re[0] = if rng.random::<bool>() { 1.0 } else { -1.0 };
    if d.is_multiple_of(2) {
        re[d / 2] = if rng.random::<bool>() { 1.0 } else { -1.0 };
    }
    let half = d / 2;
    for k in 1..half {
        let theta: f64 = rng.random_range(0.0..(2.0 * std::f64::consts::PI));
        re[k] = theta.cos();
        im[k] = theta.sin();
        re[d - k] = theta.cos();
        im[d - k] = -theta.sin();
    }
    ifft(&Complex::new(re, im))
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb)
}

/// kat_binding_identity (Prop 0) — the convolution theorem, both sides computed
/// independently: circular_conv is the direct O(d^2) time-domain definition,
/// bind_freq (`bind`) goes through fft/ifft. If they disagree, the FFT wiring
/// (or a sign in complex multiply) is broken.
#[test]
fn kat_binding_identity() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let d = 256;
    let a = random_vec(&mut rng, d);
    let b = random_vec(&mut rng, d);

    let direct = circular_conv(&a, &b);
    let via_fft = bind(&a, &b);

    assert_eq!(direct.len(), via_fft.len());
    for i in 0..d {
        assert!(
            (direct[i] - via_fft[i]).abs() < 1e-6,
            "mismatch at index {i}: direct={} via_fft={}",
            direct[i],
            via_fft[i]
        );
    }
}

/// Sanity identity for the FFT wrapper itself, independent of bind semantics:
/// ifft(fft(x)) == x for arbitrary real x (round-trip).
#[test]
fn kat_fft_roundtrip() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let d = 64;
    let x = random_vec(&mut rng, d);
    let spectrum = fft(&x);
    let back = ifft(&spectrum);
    for i in 0..d {
        assert!((x[i] - back[i]).abs() < 1e-9, "roundtrip mismatch at {i}");
    }
}

/// kat_unbind — unbind(bind(a,b), b) recovers a up to high cosine similarity
/// at d=1024, using flat-spectrum a,b (see `flat_spectrum_vec` doc comment for
/// why: it's the standard construction for which correlation-based unbind is
/// an exact deconvolution, so the 0.98 bar is a genuine, reliable check of the
/// fft/conj/mul/ifft wiring rather than a coin flip on vector luck).
#[test]
fn kat_unbind() {
    let mut rng = ChaCha8Rng::seed_from_u64(1234);
    let d = 1024;
    let a = flat_spectrum_vec(&mut rng, d);
    let b = flat_spectrum_vec(&mut rng, d);

    let bound = bind(&a, &b);
    let recovered = unbind(&bound, &b);

    let cos = cosine(&recovered, &a);
    assert!(
        cos >= 0.98,
        "cosine(unbind(bind(a,b),b), a) = {cos}, need >= 0.98"
    );
}

/// Kills-mutant: dropping the conjugate in `unbind` breaks recovery. Directly
/// exercise the frequency-domain identity `unbind` is supposed to implement:
/// ifft(fft(bind(a,b)) * conj(fft(b))) should be close to `a`, whereas
/// ifft(fft(bind(a,b)) * fft(b)) (no conjugate) should NOT be close to `a`.
#[test]
fn kat_unbind_needs_conjugate() {
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let d = 512;
    let a = flat_spectrum_vec(&mut rng, d);
    let b = flat_spectrum_vec(&mut rng, d);
    let bound = bind(&a, &b);

    // Correct: with conjugate.
    let recovered = unbind(&bound, &b);
    let cos_correct = cosine(&recovered, &a);
    assert!(
        cos_correct >= 0.98,
        "cosine (correct, with conjugate) = {cos_correct}"
    );

    // Broken: without conjugate (manually reproduce the mutant here, since
    // this catalog mutant is about deleting a conjugate call inside unbind).
    let spec_bound = fft(&bound);
    let spec_b = fft(&b);
    let no_conj = spec_bound.mul(&spec_b); // BUG: should be spec_b.conj()
    let broken = ifft(&no_conj);
    let cos_broken = cosine(&broken, &a);
    assert!(
        cos_broken < 0.5,
        "no-conjugate variant should NOT recover a (got cosine {cos_broken}), \
         otherwise kat_unbind can't tell a missing conjugate from a correct unbind"
    );
}

/// kat_complex_phase_rotation_preserves_magnitude — plumbing sanity check with
/// hand-computed values: multiplying by the conjugate of a unit-modulus
/// number is a pure phase rotation and preserves magnitude.
/// z = 1 (unit modulus, angle 0), w = e^{i*pi/2} = (0, 1).
/// conj(w) = (0, -1); z * conj(w) should have abs2 == 1 (hand-computed).
#[test]
fn kat_complex_phase_rotation_preserves_magnitude() {
    let z = Complex::new(vec![1.0], vec![0.0]);
    let w = Complex::new(vec![0.0], vec![1.0]);
    let rotated = z.mul(&w.conj());
    assert!((rotated.abs2()[0] - 1.0).abs() < 1e-9);
}
