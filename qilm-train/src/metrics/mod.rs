//! metrics — model-level measurements for the Phase-1+ gates. Each metric here
//! has (a) a known-answer test with a HAND-computed expectation independent of
//! the metric code (Rule 1) and (b) a paired anti-vacuity canary proving it can
//! say "no" (Rule 2). See implementation/tests/PHASE-1.md T1.3/T1.4.

pub mod bpb;
pub mod collapse;
