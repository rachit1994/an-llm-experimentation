//! qilm-data — deterministic, content-hashed data (VERIFICATION.md §7, tests/README H1).
//!
//! synth_markov(order, seed, n) -> (bytes, Truth)  with entropy_bits_per_byte (H) and
//!     order0_entropy (H0) computed ANALYTICALLY from the transition matrix — the target
//!     the model must reach, independent of any model. (T0.7)
//! synth_concepts(...) -> (items, Truth)           planted-concept invariance fixture.
//! load_split(name) -> bytes                        asserts sha256 ∈ data/SPLIT_HASHES.
//!
//! kat_synth_entropy: H from the matrix must match an independent empirical estimator
//! within 0.02 — proving the fixture's own target is correct.
#![allow(dead_code)]

pub mod synth;
pub mod splits;

// Re-export public types and functions
pub use synth::{synth_markov, synth_concepts, MarkovTruth, ConceptTruth, ConceptItem, write_truth_json};
pub use splits::load_split;
