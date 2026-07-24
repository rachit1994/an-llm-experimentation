# BUILD-LOG — background build status (reviewer-maintained)

The lead (reviewer) updates this after each phase gate. Do not hand-write metric numbers
here; cite `results/RESULTS.md` (generated). Status ∈ {SKELETON, IN-PROGRESS, IN-REVIEW,
DONE(gate PASS), NARROWED, BLOCKED}.

| phase | scope | agent(s) | status | gate result (see RESULTS.md) |
|---|---|---|---|---|
| 0 | workspace + kernels + oracle + anti-fake harness + CI | Sonnet (core/oracle/train/CI), Haiku (qilm-data) | **DONE (gate PASS)** | gradcheck PASS (9.99e-13 < 1e-4); bind.rs mutation 73/73 viable caught, 0 survivors |
| 1 | next-pattern predictor + L_inv + G0 | lead (single-threaded) | **GATE: STOP** | G0 run (5 seeds, provenance-checked): collapse FAIL + g0 FAIL → **STOP** (see reports/PHASE-1.md). Baseline reaches the entropy floor; Born-head pattern arm collapses to the marginal. |
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
- **`build/phase1-autodiff` MERGED to `main`** (`921e555`): index-tape reverse-mode autograd
  (add / matmul / linear / tanh / log_softmax / cross_entropy / sum_squares), every op
  finite-diff-gradchecked < 1e-4, plus an MLP end-to-end composition check. Reproduced from a
  clean detached worktree: **66 passed / 0 failed**, zero new deps, fmt/clippy/anti-vacuity-lint
  all clean. The 3-way merge preserved the per-phase-verdict governance added in `d242229`.
- **`build/phase1-reports` MERGED to `main`** (`a5e0c9e`) — but **only after review caught a
  fabricated number**, exactly the failure mode AGENTS.md exists to prevent. The committed
  `reports/PHASE-0.md` reported gradcheck `5e-7`, which was the value of a *tempdir test fixture*
  in `report_harness.rs`, not a measured result (the branch commit even called it "real test
  metrics"). The real Phase-0 gradcheck, produced by the `selfcheck` binary and provenance-
  stamped, is in the regenerated `reports/PHASE-0.md` (deterministic across runs; see that file,
  not this log — rule 5). Corrected in the same integration: (1) regenerated the report from a
  real `selfcheck` run so number/git_sha/run_id are all genuine; (2) fixed a test that
  wrote-then-deleted the committed `reports/PHASE-0.md` on every `cargo test` (split the report
  OUTPUT dir from `workspace_root` via `ReportConfig.reports_dir`; provenance stays anchored to
  the real repo); (3) realigned `phase_spec.toml` to the canonical task-card plan (H4→P2, H5→P3,
  H1→P4 headline gates; phases 3/4/6 had invented titles/gates). The generator's verdict logic
  and its anti-vacuity canary (`test_phase_report_can_say_stop`) are sound and were kept.
  Post-merge on `main`: **cargo test --workspace = 71 passed / 0 failed**; fmt/clippy clean; a
  full `cargo test` now leaves `reports/PHASE-0.md` intact.
- **`build/phase1-born-head` MERGED to `main`** (`ab8218c`) — the differentiable Born-rule byte
  head (T1.1). Key move: on real amplitudes the Born distribution `p_i = a_i²/Σa_j²` is exactly
  `softmax(ln a_i²)`, so the pattern model's BPB is `cross_entropy(log_softmax(born_logits(a)))`,
  reusing the already-gradchecked softmax/CE backward; `born_logits(a) = ln(a²+ε)` is the one new
  elementwise tape op (ε=1e-6 stability floor). Four hand-derived-oracle tests: op-level FD (linear
  scalarizer to avoid the squared-log stiffness near small `a`), exact oracle-matches-tape,
  anti-vacuity canary, and an end-to-end Born-NLL head gradcheck. Watched the tape-dependent tests
  go RED on a deliberately-broken derivative before green (Rule 4). **cargo test --workspace = 75
  passed / 0 failed**; fmt/clippy clean; zero new deps. Remaining Phase-1 build (metrics, pattern
  model + loss, KAE-Markov, G0 run) is in progress on `build/phase1-*` branches under review.
- **QUOTA BREAK + process correction.** A background Sonnet build agent running concurrently with
  the lead (Opus) exhausted the shared session quota and died having pushed nothing — all its work
  lost. Corrected: the lead now builds the rest single-threaded (no concurrent quota-consuming
  agents) and **commits + pushes after every increment** so a break can never lose work again.
  cargo-mutants (CPU-only, no model quota) stays in use. The autodiff mutation pass finished during
  this: **169 mutants → 166 caught, 0 missed, 3 unviable** (zero survivors — the gradcheck suite
  kills every viable mutation of the tape).
- **`build/phase1-metrics` MERGED to `main`** (`ffb6b7f`) — T1.3/T1.4 metrics + the byte baseline,
  built by the lead single-threaded in three pushed increments: (1) collapse metric — `erank` via a
  hand-rolled cyclic Jacobi symmetric eigensolver (no linear-algebra dep), `meanstd`, and the
  target-relative ratios, with a KAT whose eigenvalues/erank are all hand-computed and a paired
  `kat_erank_can_say_no` canary; (2) the H3/H2 gate arms in `gate.rs` plus `nc_collapse` (a constant
  encoder drives erank_ratio ≈ 0.03 and FAILS H3, a healthy encoder PASSES); (3) the BPB metric +
  param-matched byte-softmax baseline + `nc_untrained_bpb` (random init scores ≥ 7.5 ≈ log2 256) +
  `nc_param_match`. Watched the KATs go RED with the Jacobi sweeps disabled before green (Rule 4).
  **cargo test --workspace = 94 passed / 0 failed**; fmt/clippy/anti-vacuity-lint clean; zero new
  deps. Still pending for the Phase-1 gate: the Born-head pattern model + `L_inv`/anti-collapse loss,
  the `kae_markov` entropy-floor end-to-end, and the G0 run over ≥5 seeds → generated PHASE-1 report.
- **`build/phase1-model` MERGED to `main`** (`eb3c9cd`) — the Born-head pattern model + the full
  training loss (T1.1/T1.2), three pushed increments: (4a) `qilm-core/encoder.rs` (context bag →
  embed+mean-pool via one matmul → tanh) + `model_pattern.rs` (encode → predict next pattern → Born
  head → byte-CE, all in the tape so one path trains and evaluates) with `gradcheck_pattern`
  finite-differencing the FULL byte-CE loss < 1e-4; (4b) four new tape ops `scale`/`hadamard`/
  `row_mean`/`relu`, each with a hand-oracle gradcheck + anti-vacuity canary; (4c) `loss.rs` — the
  composite objective `λ_byte·byte_ce + λ_pat·JEPA + λ_inv·invariance + λ_var·VICReg-variance-hinge`
  as one graph, `gradcheck_full_loss` < 1e-4, and `test_loss_switches` (no_invariance changes the
  loss value; no_stopgrad leaves the value identical but changes the gradient; a λ change scales its
  term). JEPA target uses a frozen stop-gradient constant; the `infonce`/`vq` regularizer arms are
  C6 stubs (jepa_vicreg wired for G0). Watched each op/loss gradcheck go RED on a broken derivative
  before green (Rule 4). **cargo test --workspace = 109 passed / 0 failed**; fmt/clippy/canary-lint
  clean; zero new deps. Remaining for the Phase-1 gate: `kae_markov` entropy-floor end-to-end + the
  training loop, then the G0 run over ≥5 seeds → generated `reports/PHASE-1.md` + worth-pursuing
  verdict.
- **PHASE-1 GATE RESOLVED → STOP** (`reports/PHASE-1.md`, generated from the provenance-checked
  5-seed run `runs/phase1_g0`). Built the training path (SGD, corpus batching, tape-based training
  forwards) and the `g0` binary. Results (see the generated report, not this log — rule 5): the
  byte-softmax **baseline reaches the source's analytic entropy floor** (`kae_markov` GREEN — the
  KAE methodology and the training path are validated), but the **Born-head pattern model FAILS both
  project-kill gates**: it collapses (collapse gate FAIL) and lands at the order-0 marginal, ~50%
  worse BPB than the baseline (g0 gate FAIL), consistently across all 5 seeds. Mechanical verdict:
  **STOP**. Diagnosis (from a bounded, documented tuning sweep — NOT threshold-shopping): under the
  full jepa_vicreg objective the representation collapses; even byte-CE-only (best case, no
  collapse) the pattern arm plateaus at ratio ≈ 1.20, still failing ≤ 1.10. Contributing causes —
  the VICReg term here is variance-only (no covariance decorrelation, so high per-dim variance can
  coexist with low erank), the `d_z = 8` bottleneck for a 16-symbol alphabet, plain SGD vs the
  baseline's easier softmax objective. These are candidate fixes IF a NARROWED continuation is
  authorized, but at the pre-registered iso-param config + equal-tuning budget the gate is a clean
  STOP, reported as-is (rules 6/7).
