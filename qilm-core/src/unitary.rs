//! unitary — Cayley + Givens unitary layers (T0.5).
//!
//! Cayley: given a general complex matrix `raw`, project it onto the
//! skew-Hermitian cone via `A = (raw - raw^†) / 2` (this projection IS the
//! "skew-Hermitian constraint" from the mutant catalog — dropping it is
//! exactly the mutant `kat_cayley_unitary` must catch), then
//! `U = (I - A)(I + A)^{-1}` is unitary for any skew-Hermitian A (a classical
//! fact: eigenvalues of A are purely imaginary, so I+A is always invertible
//! and the Cayley transform of a skew-Hermitian generator is unitary).
//!
//! Givens: a single sweep of plane rotations G(i,i+1,angle_i) for
//! i in 0..d-1, composed into U = G_{d-2} * ... * G_0. Each G is real
//! orthogonal (embedded with im=0 in the complex representation), and a
//! product of orthogonal matrices is orthogonal, hence trivially unitary
//! (U^T = U^{-1} = U^†  since U is real).

use crate::complex::Complex;

/// A dense d x d complex matrix, row-major.
#[derive(Clone, Debug)]
pub struct ComplexMat {
    pub re: Vec<f64>,
    pub im: Vec<f64>,
    pub d: usize,
}

impl ComplexMat {
    pub fn zeros(d: usize) -> Self {
        Self { re: vec![0.0; d * d], im: vec![0.0; d * d], d }
    }

    pub fn identity(d: usize) -> Self {
        let mut m = Self::zeros(d);
        for i in 0..d {
            m.re[i * d + i] = 1.0;
        }
        m
    }

    pub fn get(&self, i: usize, j: usize) -> (f64, f64) {
        (self.re[i * self.d + j], self.im[i * self.d + j])
    }

    pub fn set(&mut self, i: usize, j: usize, re: f64, im: f64) {
        self.re[i * self.d + j] = re;
        self.im[i * self.d + j] = im;
    }

    pub fn add(&self, o: &Self) -> Self {
        assert_eq!(self.d, o.d);
        let re = self.re.iter().zip(&o.re).map(|(a, b)| a + b).collect();
        let im = self.im.iter().zip(&o.im).map(|(a, b)| a + b).collect();
        Self { re, im, d: self.d }
    }

    pub fn sub(&self, o: &Self) -> Self {
        assert_eq!(self.d, o.d);
        let re = self.re.iter().zip(&o.re).map(|(a, b)| a - b).collect();
        let im = self.im.iter().zip(&o.im).map(|(a, b)| a - b).collect();
        Self { re, im, d: self.d }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self { re: self.re.iter().map(|a| a * s).collect(), im: self.im.iter().map(|a| a * s).collect(), d: self.d }
    }

    /// Conjugate transpose (Hermitian adjoint) M^†.
    pub fn conj_transpose(&self) -> Self {
        let d = self.d;
        let mut out = Self::zeros(d);
        for i in 0..d {
            for j in 0..d {
                let (re, im) = self.get(i, j);
                out.set(j, i, re, -im);
            }
        }
        out
    }

    /// Complex matrix multiply.
    pub fn matmul(&self, o: &Self) -> Self {
        assert_eq!(self.d, o.d);
        let d = self.d;
        let mut out = Self::zeros(d);
        for i in 0..d {
            for k in 0..d {
                let (a_re, a_im) = self.get(i, k);
                if a_re == 0.0 && a_im == 0.0 {
                    continue;
                }
                for j in 0..d {
                    let (b_re, b_im) = o.get(k, j);
                    let (o_re, o_im) = out.get(i, j);
                    out.set(i, j, o_re + a_re * b_re - a_im * b_im, o_im + a_re * b_im + a_im * b_re);
                }
            }
        }
        out
    }

    /// Complex matrix-vector product Mψ.
    pub fn apply(&self, psi: &Complex) -> Complex {
        let d = self.d;
        assert_eq!(psi.len(), d, "ComplexMat::apply: dimension mismatch");
        let mut re = vec![0.0; d];
        let mut im = vec![0.0; d];
        for i in 0..d {
            let mut sre = 0.0;
            let mut sim = 0.0;
            for j in 0..d {
                let (a_re, a_im) = self.get(i, j);
                let (x_re, x_im) = (psi.re[j], psi.im[j]);
                sre += a_re * x_re - a_im * x_im;
                sim += a_re * x_im + a_im * x_re;
            }
            re[i] = sre;
            im[i] = sim;
        }
        Complex { re, im }
    }

    fn swap_rows(&mut self, r1: usize, r2: usize) {
        let d = self.d;
        for j in 0..d {
            self.re.swap(r1 * d + j, r2 * d + j);
            self.im.swap(r1 * d + j, r2 * d + j);
        }
    }

    /// Complex matrix inverse via Gauss-Jordan elimination with partial
    /// (magnitude) pivoting. Panics if the matrix is (numerically) singular.
    pub fn inverse(&self) -> Self {
        let d = self.d;
        let mut a = self.clone();
        let mut inv = Self::identity(d);

        for col in 0..d {
            let mut pivot_row = col;
            let (pre0, pim0) = a.get(col, col);
            let mut best = pre0.hypot(pim0);
            for r in (col + 1)..d {
                let (re, im) = a.get(r, col);
                let mag = re.hypot(im);
                if mag > best {
                    best = mag;
                    pivot_row = r;
                }
            }
            assert!(best > 1e-12, "ComplexMat::inverse: matrix is singular or near-singular");
            if pivot_row != col {
                a.swap_rows(col, pivot_row);
                inv.swap_rows(col, pivot_row);
            }

            let (pre, pim) = a.get(col, col);
            let denom = pre * pre + pim * pim;
            for j in 0..d {
                let (are, aim) = a.get(col, j);
                a.set(col, j, (are * pre + aim * pim) / denom, (aim * pre - are * pim) / denom);
                let (ire, iim) = inv.get(col, j);
                inv.set(col, j, (ire * pre + iim * pim) / denom, (iim * pre - ire * pim) / denom);
            }

            for r in 0..d {
                if r == col {
                    continue;
                }
                let (fre, fim) = a.get(r, col);
                if fre == 0.0 && fim == 0.0 {
                    continue;
                }
                for j in 0..d {
                    let (are, aim) = a.get(col, j);
                    let (rre, rim) = a.get(r, j);
                    a.set(r, j, rre - (fre * are - fim * aim), rim - (fre * aim + fim * are));

                    let (aire, aiim) = inv.get(col, j);
                    let (rire, riim) = inv.get(r, j);
                    inv.set(r, j, rire - (fre * aire - fim * aiim), riim - (fre * aiim + fim * aire));
                }
            }
        }
        inv
    }
}

/// Project a general complex matrix onto the skew-Hermitian cone:
/// A = (raw - raw^†) / 2. This is the constraint the Cayley catalog mutant
/// removes; without it, `cayley` is not guaranteed unitary.
pub fn skew_hermitian_part(raw: &ComplexMat) -> ComplexMat {
    let dagger = raw.conj_transpose();
    raw.sub(&dagger).scale(0.5)
}

/// Cayley transform of a skew-Hermitian generator: U = (I - A)(I + A)^{-1}.
/// `a_skew` must already be skew-Hermitian (build it with
/// `skew_hermitian_part` from unconstrained raw parameters).
pub fn cayley(a_skew: &ComplexMat) -> ComplexMat {
    let d = a_skew.d;
    let ident = ComplexMat::identity(d);
    let i_minus_a = ident.sub(a_skew);
    let i_plus_a = ident.add(a_skew);
    let inv = i_plus_a.inverse();
    i_minus_a.matmul(&inv)
}

/// A single sweep of Givens (plane) rotations: U = G_{d-2} * ... * G_1 * G_0,
/// where G_i rotates the (i, i+1) coordinate plane by `angles[i]`. Requires
/// `angles.len() == d - 1`.
pub fn givens(angles: &[f64], d: usize) -> ComplexMat {
    assert_eq!(angles.len(), d.saturating_sub(1), "givens: need exactly d-1 angles for one sweep");
    let mut u = ComplexMat::identity(d);
    for (i, &theta) in angles.iter().enumerate() {
        let (c, s) = (theta.cos(), theta.sin());
        let mut g = ComplexMat::identity(d);
        g.set(i, i, c, 0.0);
        g.set(i, i + 1, -s, 0.0);
        g.set(i + 1, i, s, 0.0);
        g.set(i + 1, i + 1, c, 0.0);
        u = g.matmul(&u);
    }
    u
}
