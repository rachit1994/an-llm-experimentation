//! qilm-train — training loops, metrics, gates, and the anti-fake report pipeline.
//!
//! provenance — write_metrics stamps git_sha, config_sha256, dataset_sha256,
//!     code_fingerprint, seed, backend, wall_clock_s, host. (VERIFICATION.md §4.1, T0.9)
//! gate(name, metrics.json) -> exit 0/1, reads gates.toml. Gates are code, not prose.
//! (Phase 1+) metrics::{collapse, bpb, invariance, ece}, model_pattern, model_token, loss.
#![allow(dead_code)]

pub mod provenance;
pub mod gate;
