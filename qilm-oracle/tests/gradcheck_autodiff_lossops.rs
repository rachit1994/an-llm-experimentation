//! gradcheck_autodiff_lossops — finite-difference checks for the Phase-1 loss
//! ops added to the tape: `scale`, `hadamard`, `row_mean`, `relu` (T1.2). Each
//! op has a hand-derived oracle (independent of the tape's backward, Rule 1),
//! an exact oracle-matches-tape check, and an anti-vacuity canary (Rule 2).
//! Scalarizer for every finite-difference check is `sum_squares(op(...))`.

use qilm_core::autodiff::{Shape, Tape};
use qilm_oracle::gradcheck::gradcheck;

// ------------------------------------------------------------- scale(x, s)
const S: f64 = 1.7;

// L = Σ (s·x_i)²  ⇒  dL/dx_i = 2 s² x_i
fn scale_loss(x: &[f64]) -> f64 {
    x.iter().map(|v| (S * v).powi(2)).sum()
}
fn scale_grad(x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| 2.0 * S * S * v).collect()
}

#[test]
fn gradcheck_autodiff_scale() {
    let x = vec![0.5, -1.3, 2.0, 0.7];
    assert!(gradcheck(scale_loss, scale_grad, &x, 1e-4) < 1e-4);

    // oracle matches tape
    let mut t = Tape::new();
    let l = t.leaf(x.clone(), Shape::row(x.len()));
    let y = t.scale(l, S);
    let sq = t.sum_squares(y);
    t.backward(sq);
    assert!((t.value(sq)[0] - scale_loss(&x)).abs() < 1e-9);
    for (g, o) in t.grad(l).iter().zip(scale_grad(&x)) {
        assert!((g - o).abs() < 1e-9);
    }
}

#[test]
fn gradcheck_autodiff_scale_can_say_no() {
    // WRONG: forget one factor of s (use 2·s·x instead of 2·s²·x).
    let x = vec![0.5, -1.3, 2.0];
    let wrong = |x: &[f64]| x.iter().map(|v| 2.0 * S * v).collect::<Vec<_>>();
    assert!(gradcheck(scale_loss, wrong, &x, 1e-4) > 1e-4);
}

// ---------------------------------------------------------- hadamard(a, b)
// params = [a.. , b..]; L = Σ (a_i b_i)²  ⇒  dL/da_i = 2 a_i b_i², dL/db_i = 2 a_i² b_i
fn split(p: &[f64]) -> (&[f64], &[f64]) {
    p.split_at(p.len() / 2)
}
fn had_loss(p: &[f64]) -> f64 {
    let (a, b) = split(p);
    a.iter().zip(b).map(|(x, y)| (x * y).powi(2)).sum()
}
fn had_grad(p: &[f64]) -> Vec<f64> {
    let (a, b) = split(p);
    let mut g: Vec<f64> = a.iter().zip(b).map(|(x, y)| 2.0 * x * y * y).collect();
    g.extend(a.iter().zip(b).map(|(x, y)| 2.0 * x * x * y));
    g
}

#[test]
fn gradcheck_autodiff_hadamard() {
    let p = vec![0.7, -0.4, 1.2, /*b*/ 1.1, 0.9, -0.6];
    assert!(gradcheck(had_loss, had_grad, &p, 1e-4) < 1e-4);

    let (a, b) = split(&p);
    let mut t = Tape::new();
    let la = t.leaf(a.to_vec(), Shape::row(a.len()));
    let lb = t.leaf(b.to_vec(), Shape::row(b.len()));
    let y = t.hadamard(la, lb);
    let sq = t.sum_squares(y);
    t.backward(sq);
    assert!((t.value(sq)[0] - had_loss(&p)).abs() < 1e-9);
    let oracle = had_grad(&p);
    for (g, o) in t.grad(la).iter().zip(&oracle[..a.len()]) {
        assert!((g - o).abs() < 1e-9);
    }
    for (g, o) in t.grad(lb).iter().zip(&oracle[a.len()..]) {
        assert!((g - o).abs() < 1e-9);
    }
}

#[test]
fn gradcheck_autodiff_hadamard_can_say_no() {
    // WRONG: swap the roles (da uses a², db uses b²) — disagrees unless a==b.
    let p = vec![0.7, -0.4, 1.2, 1.1, 0.9, -0.6];
    let wrong = |p: &[f64]| {
        let (a, b) = split(p);
        let mut g: Vec<f64> = a.iter().zip(b).map(|(x, y)| 2.0 * x * x * y).collect();
        g.extend(a.iter().zip(b).map(|(x, y)| 2.0 * x * y * y));
        g
    };
    assert!(gradcheck(had_loss, wrong, &p, 1e-4) > 1e-4);
}

// ------------------------------------------------------------- row_mean(x)
// x is (ROWS×COLS); y_j = mean_i x_ij; L = Σ_j y_j²  ⇒  dL/dx_ij = 2 y_j / ROWS
const ROWS: usize = 3;
const COLS: usize = 4;
fn colmeans(x: &[f64]) -> Vec<f64> {
    let mut m = [0.0; COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            m[c] += x[r * COLS + c];
        }
    }
    m.iter().map(|v| v / ROWS as f64).collect()
}
fn rm_loss(x: &[f64]) -> f64 {
    colmeans(x).iter().map(|m| m * m).sum()
}
fn rm_grad(x: &[f64]) -> Vec<f64> {
    let m = colmeans(x);
    let mut g = vec![0.0; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            g[r * COLS + c] = 2.0 * m[c] / ROWS as f64;
        }
    }
    g
}

#[test]
fn gradcheck_autodiff_row_mean() {
    let x: Vec<f64> = (0..ROWS * COLS).map(|i| 0.3 * (i as f64) - 1.0).collect();
    assert!(gradcheck(rm_loss, rm_grad, &x, 1e-4) < 1e-4);

    let mut t = Tape::new();
    let l = t.leaf(x.clone(), Shape::mat(ROWS, COLS));
    let y = t.row_mean(l);
    let sq = t.sum_squares(y);
    t.backward(sq);
    assert_eq!(t.shape(y), Shape::row(COLS));
    assert!((t.value(sq)[0] - rm_loss(&x)).abs() < 1e-9);
    for (g, o) in t.grad(l).iter().zip(rm_grad(&x)) {
        assert!((g - o).abs() < 1e-9);
    }
}

#[test]
fn gradcheck_autodiff_row_mean_can_say_no() {
    // WRONG: forget the 1/ROWS averaging factor in the backward.
    let x: Vec<f64> = (0..ROWS * COLS).map(|i| 0.3 * (i as f64) - 1.0).collect();
    let wrong = |x: &[f64]| {
        let m = colmeans(x);
        let mut g = vec![0.0; ROWS * COLS];
        for r in 0..ROWS {
            for c in 0..COLS {
                g[r * COLS + c] = 2.0 * m[c]; // missing / ROWS
            }
        }
        g
    };
    assert!(gradcheck(rm_loss, wrong, &x, 1e-4) > 1e-4);
}

// ------------------------------------------------------------------ relu(x)
// L = Σ max(0, x_i)²  ⇒  dL/dx_i = 2 x_i if x_i>0 else 0.  Inputs kept away
// from 0 so finite differences don't straddle the kink.
fn relu_loss(x: &[f64]) -> f64 {
    x.iter().map(|v| v.max(0.0).powi(2)).sum()
}
fn relu_grad(x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|v| if *v > 0.0 { 2.0 * v } else { 0.0 })
        .collect()
}

#[test]
fn gradcheck_autodiff_relu() {
    let x = vec![0.8, -1.5, 2.2, -0.6, 1.1];
    assert!(gradcheck(relu_loss, relu_grad, &x, 1e-4) < 1e-4);

    let mut t = Tape::new();
    let l = t.leaf(x.clone(), Shape::row(x.len()));
    let y = t.relu(l);
    let sq = t.sum_squares(y);
    t.backward(sq);
    assert!((t.value(sq)[0] - relu_loss(&x)).abs() < 1e-9);
    for (g, o) in t.grad(l).iter().zip(relu_grad(&x)) {
        assert!((g - o).abs() < 1e-9);
    }
}

#[test]
fn gradcheck_autodiff_relu_can_say_no() {
    // WRONG: treat relu as identity (2·x everywhere) — disagrees on x<0.
    let x = vec![0.8, -1.5, 2.2, -0.6];
    let wrong = |x: &[f64]| x.iter().map(|v| 2.0 * v).collect::<Vec<_>>();
    assert!(gradcheck(relu_loss, wrong, &x, 1e-4) > 1e-4);
}
