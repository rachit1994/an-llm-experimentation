//! qilm-core — model kernels for CCR.
//!
//! Phase 0 (implementation/tests/PHASE-0.md) implements, each with a KAT + gradcheck:
//!   complex   — (re, im) pair type + add/mul/conj/abs2/scale            (T0.1)
//!   bind      — circular convolution + FFT bind/unbind (Prop 0)         (T0.3)
//!   born      — squared-magnitude readout, normalized                   (T0.4)
//!   unitary   — Cayley + Givens; U†U = I, norm-preserving               (T0.5)
//!   hopfield  — X·softmax(β XᵀΞ) retrieval                              (T0.6)
//!
//! CONTRACT: every public fn here has a matching test in qilm-oracle. A fn without
//! a KAT + (if it has params) a gradcheck is not "done" (see ../VERIFICATION.md §8).
#![allow(dead_code)]

pub mod autodiff;
pub mod bind;
pub mod born;
pub mod complex;
pub mod encoder;
pub mod hopfield;
pub mod unitary;
