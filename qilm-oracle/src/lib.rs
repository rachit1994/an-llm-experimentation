//! qilm-oracle — scalar f64 reference implementations + finite-difference gradient
//! checks. The firewall against fake training numbers (VERIFICATION.md §3, T0.2).
//!
//! gradcheck(f, params, eps=1e-4) -> max_rel_err ; MUST be < 1e-4 for every kernel.
#![allow(dead_code)]

pub mod gradcheck;
