//! collapse — the anti-collapse metric (T1.3), measured RELATIVE TO THE TARGET
//! PATTERNS. This is the exact locus of the repo's founding scar (a variance
//! metric that was green but vacuous), so read AGENTS.md Rules 1–3 before
//! touching it: the "expected" spread comes from an INDEPENDENT reference set
//! `Z*`, never from `Z` itself, and a constant encoder must drive the ratio to
//! ~0 (the `nc_collapse_canary`), not silently pass.
//!
//! Two collapse signals, both compared to the targets' own values:
//!   - meanstd(Z): mean over dims of the per-dim population std. Collapse (all
//!     rows equal) → every dim has std 0 → meanstd 0.
//!   - erank(Z):   effective rank (Roy & Vetterli 2007) = exp(H), H the Shannon
//!     entropy of the normalized singular values of Z. Collapse → few effective
//!     directions → low erank.
//!
//! The gate (H3) passes iff BOTH ratios `erank(Z)/erank(Z*)` and
//! `meanstd(Z)/meanstd(Z*)` clear their frozen mins (gates.toml [collapse]).
//!
//! Singular values are obtained WITHOUT a linear-algebra dependency (Rule 12):
//! σ_i = sqrt(eigenvalue_i of the d×d Gram matrix ZᵀZ), and the symmetric
//! eigenproblem is solved with a hand-rolled cyclic Jacobi rotation sweep
//! (Golub & Van Loan, *Matrix Computations* 4th ed., §8.5), validated by a
//! known-answer test (collapse_kat.rs) with hand-computed eigenvalues.

/// Population standard deviation per column, averaged over the `d` columns.
/// `z` is row-major `n × d`. Panics if `z.len() != n*d` or `n == 0`.
pub fn meanstd(z: &[f64], n: usize, d: usize) -> f64 {
    assert_eq!(
        z.len(),
        n * d,
        "meanstd: z length {} != n*d {}",
        z.len(),
        n * d
    );
    assert!(n > 0 && d > 0, "meanstd: empty matrix");
    let mut acc = 0.0;
    for j in 0..d {
        let mut mean = 0.0;
        for i in 0..n {
            mean += z[i * d + j];
        }
        mean /= n as f64;
        let mut var = 0.0;
        for i in 0..n {
            let e = z[i * d + j] - mean;
            var += e * e;
        }
        var /= n as f64; // population variance
        acc += var.sqrt();
    }
    acc / d as f64
}

/// Effective rank of `z` (row-major `n × d`): `exp(H)`, `H` the entropy of the
/// normalized singular values. Range `[1, d]`. A rank-1 matrix → ~1; `d`
/// equal-weight orthogonal directions → `d`. Returns `1.0` for an all-zero
/// matrix (a single degenerate direction), never `0` or `NaN`.
pub fn erank(z: &[f64], n: usize, d: usize) -> f64 {
    assert_eq!(
        z.len(),
        n * d,
        "erank: z length {} != n*d {}",
        z.len(),
        n * d
    );
    assert!(n > 0 && d > 0, "erank: empty matrix");

    // Gram = ZᵀZ (d × d, symmetric PSD); eigenvalues are σ².
    let gram = gram_ztz(z, n, d);
    let eig = jacobi_eigenvalues(&gram, d);

    // Drop eigenvalues that are numerically zero RELATIVE to the largest, then
    // take σ = sqrt of the survivors (standard effective-rank practice). The
    // threshold is applied to the EIGENVALUES, not to σ: a Jacobi sweep leaves
    // a true-zero eigenvalue at ~1e-12·λ_max, but sqrt would lift its σ/σ_max
    // ratio to ~1e-6, so filtering after sqrt keeps noise and inflates a
    // rank-1 matrix's erank above 1. `REL_TOL` kills only numerical zeros.
    const REL_TOL: f64 = 1e-9;
    let eig_max = eig.iter().cloned().fold(0.0_f64, f64::max);
    let sigma: Vec<f64> = eig
        .iter()
        .filter(|&&e| e > REL_TOL * eig_max)
        .map(|&e| e.sqrt())
        .collect();
    let sum: f64 = sigma.iter().sum();
    if sum <= 0.0 || sigma.len() <= 1 {
        return 1.0; // one (or zero) effective direction
    }
    // H = -Σ p ln p, p = σ / Σσ; erank = exp(H).
    let mut h = 0.0;
    for &s in &sigma {
        let p = s / sum;
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h.exp()
}

/// The two collapse ratios, measured relative to the INDEPENDENT target set
/// `z_star` (Rule 1: the expectation is the targets' spread, not `z`'s own).
/// Both `z` and `z_star` must share the representation dim `d`; the sample
/// counts (`nz`, `nstar`) may differ. Returns `(erank_ratio, meanstd_ratio)`.
pub fn collapse_ratios(z: &[f64], nz: usize, z_star: &[f64], nstar: usize, d: usize) -> (f64, f64) {
    let erank_star = erank(z_star, nstar, d);
    let meanstd_star = meanstd(z_star, nstar, d);
    assert!(
        erank_star > 0.0 && meanstd_star > 0.0,
        "collapse_ratios: degenerate target set Z* (erank* {erank_star}, meanstd* {meanstd_star}) \
         — the reference must have real spread or the ratio is meaningless"
    );
    let erank_ratio = erank(z, nz, d) / erank_star;
    let meanstd_ratio = meanstd(z, nz, d) / meanstd_star;
    (erank_ratio, meanstd_ratio)
}

/// Gram matrix `ZᵀZ` (`d × d`, row-major) for a row-major `n × d` `Z`.
fn gram_ztz(z: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut g = vec![0.0; d * d];
    for i in 0..n {
        let row = &z[i * d..(i + 1) * d];
        for a in 0..d {
            let za = row[a];
            if za == 0.0 {
                continue;
            }
            for b in 0..d {
                g[a * d + b] += za * row[b];
            }
        }
    }
    g
}

/// Eigenvalues of a symmetric `n × n` matrix (row-major) via the cyclic Jacobi
/// rotation method (Golub & Van Loan §8.5). Deterministic, no dependencies.
/// Returns the `n` diagonal entries after the off-diagonal mass is driven to
/// ~0 (the eigenvalues, unordered). Validated in collapse_kat.rs against
/// hand-computed eigenvalues.
fn jacobi_eigenvalues(mat: &[f64], n: usize) -> Vec<f64> {
    let idx = |i: usize, j: usize| i * n + j;
    let mut a = mat.to_vec();
    if n <= 1 {
        return a;
    }
    const MAX_SWEEPS: usize = 100;
    for _ in 0..MAX_SWEEPS {
        // Sum of squared off-diagonal entries; stop once negligible.
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[idx(p, q)] * a[idx(p, q)];
            }
        }
        if off <= 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[idx(p, q)];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let app = a[idx(p, p)];
                let aqq = a[idx(q, q)];
                // Rotation that zeroes a[p][q]:
                //   θ = (aqq-app)/(2 apq), t = sign(θ)/(|θ|+√(θ²+1)),
                //   c = 1/√(t²+1), s = t c.  (signum(0)=1 ⇒ θ=0 gives a 45° turn.)
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Two-sided rotation A ← JᵀAJ (only rows/cols p,q change).
                for k in 0..n {
                    let akp = a[idx(k, p)];
                    let akq = a[idx(k, q)];
                    a[idx(k, p)] = c * akp - s * akq;
                    a[idx(k, q)] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[idx(p, k)];
                    let aqk = a[idx(q, k)];
                    a[idx(p, k)] = c * apk - s * aqk;
                    a[idx(q, k)] = s * apk + c * aqk;
                }
            }
        }
    }
    (0..n).map(|i| a[idx(i, i)]).collect()
}
