# BUILD-LOG — background build status (reviewer-maintained)

The lead (reviewer) updates this after each phase gate. Do not hand-write metric numbers
here; cite `results/RESULTS.md` (generated). Status ∈ {SKELETON, IN-PROGRESS, IN-REVIEW,
DONE(gate PASS), NARROWED, BLOCKED}.

| phase | scope | agent(s) | status | gate result (see RESULTS.md) |
|---|---|---|---|---|
| 0 | workspace + kernels + oracle + anti-fake harness + CI | Sonnet (core/oracle/train/CI), Haiku (qilm-data) | **DONE (gate PASS)** | gradcheck PASS (9.99e-13 < 1e-4); bind.rs mutation 73/73 viable caught, 0 survivors |
| 1 | next-pattern predictor + L_inv + G0 | — | not started | — |
| 2 | attractor memory + encoder invariance (H4) | — | not started | — |
| 3 | glow / calibration (H5) | — | not started | — |
| 4 | complex/phase ablation (H1) | — | not started | — |
| 5 | unitary dynamics (copy task) | — | not started | — |
| 6 | generation + readout mitigations | — | not started | — |

## Reviewer log
- SKELETON committed: virtual workspace (4 stub crates), `rust-toolchain.toml` (1.94.1),
  `gates.toml` + `gates.lock`. `cargo build` green on the empty workspace.
- **Haiku `qilm-data` v1 REJECTED** (review caught it): the entropy fixture generated uniform
  bytes (no Markov source) and compared the sample's entropy to itself (`|Δ|=0.000000`) — a
  vacuous known-answer test. Sent back with an exact redesign; v2 accepted (real transition
  matrix, analytic truth, independent estimator, anti-vacuity canary).
- **Sonnet `phase0-core` completed** (kernels, oracle KAT+gradcheck, provenance/gate/report,
  CI, anti-vacuity canaries) before hitting a session limit.
- **Phase 0 INTEGRATED to `main`** (fast-forward `ffe76e7`): merged both branches and fixed
  the integration defects two independent agents introduced — unified dependency versions via
  `[workspace.dependencies]` (workspace had rand 0.8 AND 0.10, mismatched sha2/serde), removed
  `rustfft` (hand-rolled O(d²) DFT) and `rand_distr` (Box-Muller), standardized on
  `rand_chacha` for determinism. Core dep budget now: rand, rand_chacha, serde, serde_json,
  sha2, toml (+ dev proptest, approx, tempfile).
  Reproduced from clean: **cargo test --workspace = 49 passed / 0 failed**; fmt clean; clippy 0
  warnings; anti-vacuity-lint passes; entropy KAT `|Δ|=0.0024` (nonzero) with canary at 5.65.
- **Phase 0 CLOSED.** Mutation testing (`cargo-mutants --test-workspace=true`) on the
  highest-risk / modified file `qilm-core/src/bind.rs`: **74 mutants → 73 caught, 0 missed, 1
  unviable** — the suite kills every viable mutation of the binding kernel (incl. the new
  hand-rolled DFT), proving the tests are non-vacuous. Full-workspace mutation is left to the
  nightly CI `mutants` job (the other kernels are finite-diff gradcheck-validated < 1e-4 +
  KAT-covered). Model-level negative controls (collapse/BPB/invariance) are correctly deferred
  to Phase 1 (their metrics don't exist yet).
- **Phase 1 STARTED:** first increment `build/phase1-autodiff` (Sonnet) — an index-tape
  reverse-mode autodiff ported from micrograd (zero new deps, every op finite-diff-gradchecked),
  pushing to origin incrementally for human review (AGENTS.md rule 13).
