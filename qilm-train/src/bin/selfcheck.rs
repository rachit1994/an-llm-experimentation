//! selfcheck — Phase 0's harness self-check binary. Actually runs the
//! finite-difference gradcheck for the Complex-multiply kernel (the same
//! computation as qilm-oracle's `gradcheck_complex` test) and records the
//! REAL measured number via `provenance::write_metrics`, proving the
//! provenance -> gate -> report chain works end-to-end before any model
//! exists (VERIFICATION.md §8: "nothing downstream is believable until
//! Phase 0's DoD is green").
//!
//! CI's `verify-results` / `repro-smoke` jobs run this (regenerating
//! `runs/`, which is gitignored) before calling `report` / `report --check`,
//! so the committed `results/RESULTS.md` is always reproducible from a clean
//! checkout, never a stored constant.

use qilm_oracle::gradcheck::gradcheck;
use qilm_train::provenance::{discover_workspace_root, sha256_hex, write_metrics, MetricsInput};
use serde_json::json;
use std::time::Instant;

// Identical to qilm-oracle/tests/gradcheck_complex.rs's `loss` /
// `analytic_grad` (see that file for the hand-derivation) -- this binary
// exists to RUN that check as part of a provenance-stamped artifact, not to
// redefine what "correct" means.
fn loss(p: &[f64]) -> f64 {
    let (a_re, a_im, b_re, b_im) = (p[0], p[1], p[2], p[3]);
    let c_re = a_re * b_re - a_im * b_im;
    let c_im = a_re * b_im + a_im * b_re;
    c_re * c_re + c_im * c_im
}

fn analytic_grad(p: &[f64]) -> Vec<f64> {
    let (a_re, a_im, b_re, b_im) = (p[0], p[1], p[2], p[3]);
    let c_re = a_re * b_re - a_im * b_im;
    let c_im = a_re * b_im + a_im * b_re;
    vec![
        2.0 * c_re * b_re + 2.0 * c_im * b_im,
        -2.0 * c_re * b_im + 2.0 * c_im * b_re,
        2.0 * c_re * a_re + 2.0 * c_im * a_im,
        -2.0 * c_re * a_im + 2.0 * c_im * a_re,
    ]
}

fn main() {
    let start = Instant::now();

    let params = [0.7_f64, -0.3, 0.5, 0.9];
    let max_rel_err = gradcheck(loss, analytic_grad, &params, 1e-4);

    let wall_clock_s = start.elapsed().as_secs_f64();

    let config_bytes = format!("selfcheck:complex_mul_gradcheck:params={params:?}:eps=1e-4");
    let config_sha256 = sha256_hex(config_bytes.as_bytes());
    let run_id = &config_sha256[..16];

    let input = MetricsInput {
        config_sha256: config_sha256.clone(),
        dataset_sha256: "none".to_string(), // pure kernel self-check, no dataset involved
        seed: 0,
        backend: "cpu-scalar-oracle".to_string(),
        wall_clock_s,
        metrics: json!({ "gradcheck_max_rel_err": max_rel_err }),
    };

    let workspace_root = discover_workspace_root();
    let runs_dir = workspace_root.join("runs");
    match write_metrics(&runs_dir, run_id, input) {
        Ok(path) => {
            println!("selfcheck: wrote {}", path.display());
            println!("selfcheck: gradcheck_max_rel_err = {max_rel_err}");
        }
        Err(e) => {
            eprintln!("selfcheck: failed to write metrics: {e}");
            std::process::exit(1);
        }
    }
}
