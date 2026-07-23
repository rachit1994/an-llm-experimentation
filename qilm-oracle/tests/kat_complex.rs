//! T0.1 — Complex (re, im) pair KATs.
//!
//! Every expected value here is hand-computed in the comment above the assertion,
//! independent of the kernel under test (VERIFICATION.md §3: never paste the
//! kernel's own output back as the "expected" value).

use qilm_core::complex::Complex;

/// kat_interference — the canonical two-wave interference identity.
///
/// psi1 = 1 * e^{i*0}         = (re=1,       im=0)
/// psi2 = 1 * e^{i*pi/3}      = (re=cos(60deg)=0.5, im=sin(60deg)=0.8660254...)
///
/// Hand-computed by the general interference law (independent of `add`/`abs2`):
///   |psi1 + psi2|^2 = r1^2 + r2^2 + 2*r1*r2*cos(delta_phi)
///                   = 1 + 1 + 2*1*1*cos(pi/3)
///                   = 2 + 2*0.5
///                   = 3.0
#[test]
fn kat_interference() {
    let psi1 = Complex::new(vec![1.0], vec![0.0]);
    let theta = std::f64::consts::PI / 3.0;
    let psi2 = Complex::new(vec![theta.cos()], vec![theta.sin()]);

    let sum = psi1.add(&psi2);
    let intensity = sum.abs2();

    let expected = 3.0_f64;
    assert!(
        (intensity[0] - expected).abs() < 1e-6,
        "abs2(psi1+psi2) = {}, expected {} (hand-computed r1^2+r2^2+2r1r2cos(pi/3))",
        intensity[0],
        expected
    );
}

/// kat_complex_mul — independent hand-computed complex multiply, chosen so both
/// cross terms (ad, bc) and both same-sign terms (ac, bd) are nonzero and
/// distinguishable, so a sign flip anywhere in `mul` (e.g. `ac - bd` -> `ac + bd`)
/// changes the result.
///
/// (2+3i) * (4+5i):
///   re = 2*4 - 3*5 = 8 - 15  = -7
///   im = 2*5 + 3*4 = 10 + 12 = 22
/// (computed by hand with the standard (a+bi)(c+di) = (ac-bd)+(ad+bc)i rule,
/// not by running the kernel).
#[test]
fn kat_complex_mul() {
    let a = Complex::new(vec![2.0], vec![3.0]);
    let b = Complex::new(vec![4.0], vec![5.0]);
    let c = a.mul(&b);

    assert!(
        (c.re[0] - (-7.0)).abs() < 1e-9,
        "re = {}, expected -7",
        c.re[0]
    );
    assert!(
        (c.im[0] - 22.0).abs() < 1e-9,
        "im = {}, expected 22",
        c.im[0]
    );
}

/// kat_complex_conj_abs2 — conj(z) has the same |z|^2 as z, and z * conj(z) has
/// zero imaginary part equal to |z|^2 in the real part.
/// z = 3 + 4i => |z|^2 = 9 + 16 = 25 (hand-computed).
#[test]
fn kat_complex_conj_abs2() {
    let z = Complex::new(vec![3.0], vec![4.0]);
    let expected_abs2 = 25.0_f64;

    assert!((z.abs2()[0] - expected_abs2).abs() < 1e-9);
    assert!((z.conj().abs2()[0] - expected_abs2).abs() < 1e-9);

    let z_times_conj = z.mul(&z.conj());
    assert!((z_times_conj.re[0] - expected_abs2).abs() < 1e-9);
    assert!(z_times_conj.im[0].abs() < 1e-9);
}
