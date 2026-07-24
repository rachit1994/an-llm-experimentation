//! qilm-train — training loops, metrics, gates, and the anti-fake report pipeline.
//!
//! provenance — write_metrics stamps git_sha, config_sha256, dataset_sha256,
//!     code_fingerprint, seed, backend, wall_clock_s, host. (VERIFICATION.md §4.1, T0.9)
//! gate(name, metrics.json) -> exit 0/1, reads gates.toml. Gates are code, not prose.
//! report — regenerates results/RESULTS.md from runs/ + gates.toml (T0.9).
//! testkit — deliberately-broken model/data constructors for the negative-
//!     control battery (T0.8); the nc_* tests live in tests/negative_controls.rs.
//! (Phase 1+) metrics::{collapse, bpb, invariance, ece}, model_pattern, model_token, loss.
#![allow(dead_code)]

pub mod gate;
pub mod loss;
pub mod metrics;
pub mod model_pattern;
pub mod model_token;
pub mod provenance;
pub mod report;
pub mod testkit;
pub mod train;
