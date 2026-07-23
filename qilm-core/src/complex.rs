//! complex — an explicit (re, im) pair type (T0.1, doc 00-stack-and-principles.md).
//!
//! DEVIATION NOTE (documented, see final report): doc 00's sketch is
//! `struct Complex<B> { re: Tensor<B>, im: Tensor<B> }`, generic over a Burn
//! backend `B`. Phase 0 has no training loop / autodiff engine yet (that
//! arrives with the model in Phase 1+), so pulling in Burn here would be
//! machinery with nothing to drive it. This concrete version stores `re`/`im`
//! as plain `Vec<f64>` (a CPU "tensor" of rank 1) — every method below is
//! written purely in terms of elementwise real arithmetic, so swapping in a
//! real `Tensor<B>` later is a mechanical substitution, not a redesign.
//! `rustfft`'s own `num_complex::Complex64` is used internally only at the FFT
//! boundary (bind.rs) and never leaks into this type's public API, per the
//! instruction not to use `num-complex` for *our* Complex type.

/// A complex-valued vector stored as separate real and imaginary parts.
#[derive(Clone, Debug, PartialEq)]
pub struct Complex {
    pub re: Vec<f64>,
    pub im: Vec<f64>,
}

impl Complex {
    /// Build a Complex from explicit re/im parts. Panics if lengths differ.
    pub fn new(re: Vec<f64>, im: Vec<f64>) -> Self {
        assert_eq!(re.len(), im.len(), "re/im length mismatch");
        Self { re, im }
    }

    /// All-zero complex vector of length `n`.
    pub fn zeros(n: usize) -> Self {
        Self {
            re: vec![0.0; n],
            im: vec![0.0; n],
        }
    }

    /// A real-valued vector lifted into Complex (im = 0).
    pub fn from_real(re: &[f64]) -> Self {
        Self {
            re: re.to_vec(),
            im: vec![0.0; re.len()],
        }
    }

    pub fn len(&self) -> usize {
        self.re.len()
    }

    pub fn is_empty(&self) -> bool {
        self.re.is_empty()
    }

    /// Elementwise complex addition: (a+bi) + (c+di) = (a+c) + (b+d)i.
    pub fn add(&self, o: &Self) -> Self {
        assert_eq!(self.len(), o.len());
        let re = self.re.iter().zip(&o.re).map(|(a, c)| a + c).collect();
        let im = self.im.iter().zip(&o.im).map(|(b, d)| b + d).collect();
        Self { re, im }
    }

    /// Elementwise complex subtraction.
    pub fn sub(&self, o: &Self) -> Self {
        assert_eq!(self.len(), o.len());
        let re = self.re.iter().zip(&o.re).map(|(a, c)| a - c).collect();
        let im = self.im.iter().zip(&o.im).map(|(b, d)| b - d).collect();
        Self { re, im }
    }

    /// Elementwise complex multiply (Hadamard product): (a+bi)(c+di) = (ac-bd) + (ad+bc)i.
    pub fn mul(&self, o: &Self) -> Self {
        assert_eq!(self.len(), o.len());
        let n = self.len();
        let mut re = Vec::with_capacity(n);
        let mut im = Vec::with_capacity(n);
        for i in 0..n {
            let (a, b, c, d) = (self.re[i], self.im[i], o.re[i], o.im[i]);
            re.push(a * c - b * d);
            im.push(a * d + b * c);
        }
        Self { re, im }
    }

    /// Complex conjugate: conj(a+bi) = a-bi.
    pub fn conj(&self) -> Self {
        Self {
            re: self.re.clone(),
            im: self.im.iter().map(|b| -b).collect(),
        }
    }

    /// Scale by a real scalar.
    pub fn scale(&self, s: f64) -> Self {
        Self {
            re: self.re.iter().map(|a| a * s).collect(),
            im: self.im.iter().map(|b| b * s).collect(),
        }
    }

    /// Elementwise squared magnitude: |a+bi|^2 = a^2 + b^2 (the Born weight).
    pub fn abs2(&self) -> Vec<f64> {
        self.re
            .iter()
            .zip(&self.im)
            .map(|(a, b)| a * a + b * b)
            .collect()
    }

    /// Elementwise magnitude.
    pub fn abs(&self) -> Vec<f64> {
        self.abs2().into_iter().map(|v| v.sqrt()).collect()
    }

    /// Squared L2 norm of the whole vector: sum_i |z_i|^2.
    pub fn norm2(&self) -> f64 {
        self.abs2().iter().sum()
    }

    /// L2 norm of the whole vector.
    pub fn norm(&self) -> f64 {
        self.norm2().sqrt()
    }

    /// Complex (Hermitian) inner product <self|o> = sum_i conj(self_i) * o_i.
    pub fn inner(&self, o: &Self) -> (f64, f64) {
        assert_eq!(self.len(), o.len());
        let mut re = 0.0;
        let mut im = 0.0;
        for i in 0..self.len() {
            // conj(self_i) = (self.re[i], -self.im[i])
            re += self.re[i] * o.re[i] + self.im[i] * o.im[i];
            im += self.re[i] * o.im[i] - self.im[i] * o.re[i];
        }
        (re, im)
    }
}
