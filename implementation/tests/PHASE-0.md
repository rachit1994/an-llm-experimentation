# Phase 0 test suite — kernels + the anti-fake harness

Goal of this phase: build the kernels **and** the machinery that makes every later number trustworthy
(fixtures, negative-control harness, provenance/report, frozen gates, CI). Nothing downstream is
believable until Phase 0's DoD is green. Read [`../VERIFICATION.md`](../VERIFICATION.md) and
[`README.md`](README.md) first.

**Precondition:** empty workspace. **DoD (this phase):** VERIFICATION §8 items 1–4 for `qilm-core` +
`qilm-oracle`, plus the harness (H1–H4) and CI (all six jobs) exist and are green on the fixtures.

---

## Kernel cards

### T0.1 — Complex `(re, im)` type + ops
- Files: `qilm-core/src/complex.rs`, `qilm-oracle/tests/kat_complex.rs`
- Signature: `struct Complex<B>{re:Tensor<B>,im:Tensor<B>}` with `add,mul,conj,abs2,scale`.
- Done-when: `kat_interference` GREEN — for `ψ1=1·e^{i0}`, `ψ2=1·e^{iπ/3}`, assert `abs2(add(ψ1,ψ2))
  == 1+1+2·cos(π/3) = 3.0` within `1e-6` (expected value hand-computed in the test comment).
- Stays-red: n/a (L0).
- Kills-mutant: sign flip in `mul` → `kat_interference` red.

### T0.2 — Finite-difference gradient check (the highest-value test)
- Files: `qilm-oracle/src/gradcheck.rs`, `qilm-oracle/tests/gradcheck_complex.rs`
- Signature: `fn gradcheck(f, params, eps=1e-4) -> max_rel_err`.
- Done-when: `gradcheck_complex` GREEN — analytic grad of `L=abs2(mul(a,b))` vs central difference,
  **max relative error < 1e-4**.
- Kills-mutant: any wrong backward in `mul` → red. *This test is the firewall against fake training
  numbers; do not weaken its tolerance.*

### T0.3 — Binding identity (Proposition 0)
- Files: `qilm-core/src/bind.rs`, `qilm-oracle/tests/kat_bind.rs`
- Signature: `fn circular_conv(a,b)`, `fn fft(x)`, `fn bind_freq(a,b)=ifft(fft(a)⊙fft(b))`.
- Done-when: `kat_binding_identity` GREEN — `circular_conv(a,b) == bind_freq(a,b)` for random `a,b`,
  `d=256`, tol `1e-6`.
- Done-when (2): `kat_unbind` GREEN — `cos(unbind(bind(a,b),b), a) ≥ 0.98` at `d=1024`.
- Kills-mutant: conjugate dropped in unbind → `kat_unbind` red.

### T0.4 — Born readout
- Files: `qilm-core/src/born.rs`, `qilm-oracle/tests/kat_born.rs`
- Signature: `fn born(psi, measurements) -> probs` (softmax-free: `|⟨m_k|ψ⟩|²` then normalize).
- Done-when: `kat_born_sum` GREEN — `Σ probs == 1` within `1e-6`, for random state + 256 measurements.
- Kills-mutant: remove normalization → `kat_born_sum` red.

### T0.5 — Unitary layers (Cayley + Givens)
- Files: `qilm-core/src/unitary.rs`, `qilm-oracle/tests/kat_unitary.rs`
- Signature: `fn cayley(A_skew)->U`, `fn givens(angles)->U`.
- Done-when: `kat_cayley_unitary` GREEN (`U†U==I`, tol `1e-6`) **and** `kat_unitary_norm` GREEN
  (`‖Uψ‖==‖ψ‖`, tol `1e-6`) for both parameterizations.
- Property (L1): `prop_unitary_norm` over 100 random inputs (proptest), same tolerance.
- Kills-mutant: Cayley without skew-Hermitian constraint → `kat_cayley_unitary` red.

### T0.6 — Modern-Hopfield retrieval
- Files: `qilm-core/src/hopfield.rs`, `qilm-oracle/tests/kat_hopfield.rs`
- Signature: `fn retrieve(X, xi, beta=1/sqrt(d)) -> pattern` (`X·softmax(β Xᵀξ)`).
- Done-when: `kat_hopfield_recall` GREEN — store 8 random patterns, query with a stored pattern +
  small noise, retrieved cosine to the original `≥ 0.99`.
- Metamorphic (L1): `prop_hopfield_perm` — permuting the stored set does not change which pattern a
  clean query retrieves.

---

## Harness cards (build these — they enforce the anti-fake guarantees)

### T0.7 — Synthetic Markov fixture with computed truth
- Files: `qilm-data/src/synth.rs`, `qilm-data/tests/kat_synth_entropy.rs`
- Signature: `fn synth_markov(order,seed,n)->(Vec<u8>, Truth)` writing `entropy_bits_per_byte` (H),
  `order0_entropy` (H0).
- Done-when: `kat_synth_entropy` GREEN — H computed from the transition matrix matches a second,
  independent estimator (empirical plug-in entropy on 2M bytes) within `0.02`. *This proves the target
  the model must reach is itself correct.*
- Kills-mutant: wrong H formula → red (the two estimators diverge).

### T0.8 — Negative-control harness
- Files: `qilm-train/tests/negative_controls.rs`, `qilm-train/src/testkit.rs`
- Signature: `constant_encoder()`, `random_init_model()`, `shuffle_labels()`, `strip_term(name)`.
- Done-when: the helpers exist and `nc_born_sum`, `nc_unitary_norm` (re-exported L0 identities) pass.
- Note: the *model-level* controls (`nc_collapse_canary`, etc.) are authored here but first exercised in
  Phase 1 when a real model exists; their metric side (does the metric fire?) is testable now with the
  `constant_encoder()` stub returning a fixed vector.

### T0.9 — Provenance + report + gate
- Files: `qilm-train/src/bin/report.rs`, `qilm-train/src/gate.rs`, `qilm-train/src/provenance.rs`,
  `gates.toml`, `gates.lock`
- Signature: `write_metrics(run_id, metrics)`; `report`; `gate(name, metrics_path)->exit_code`.
- Done-when: `test_report_refuses_fabrication` GREEN — calling `report` on a `metrics.json` whose
  `git_sha` ≠ HEAD returns non-zero and writes nothing; `test_report_roundtrip` GREEN — a valid run
  produces a `RESULTS.md` whose numbers equal the `metrics.json` (no transcription).
- Done-when (2): `test_frozen_gates` GREEN — `sha256(gates.toml)==gates.lock`.
- Kills-mutant: report reads a number from anywhere but `metrics.json` → `test_report_roundtrip` red.

### T0.10 — CI workflow
- Files: `.github/workflows/verify.yml`
- Done-when: the six jobs of [`README.md`](README.md#ci-wiring) are defined; `verify-results` and
  `frozen-gates` pass on the current tree; `repro-smoke` is wired (it becomes meaningful in Phase 1).

---

## Phase-0 DoD checklist (all must be true)

- [ ] T0.1–T0.6: every KAT + `gradcheck_*` GREEN; `prop_unitary_norm`, `prop_hopfield_perm` GREEN.
- [ ] T0.7: `kat_synth_entropy` GREEN (the fixture's target is itself verified).
- [ ] T0.8–T0.9: provenance/report/gate built; `test_report_refuses_fabrication`,
      `test_report_roundtrip`, `test_frozen_gates` GREEN.
- [ ] T0.10: CI green on `test`, `frozen-gates`, `verify-results`, `coverage-ratchet`.
- [ ] `cargo mutants -p qilm-core` reports **zero survivors** in `complex.rs`, `bind.rs`, `born.rs`,
      `unitary.rs` (each catalog mutant killed by a named test).
- [ ] `results/RESULTS.md` exists, is generator-stamped, and contains **no** model numbers yet (only
      the harness self-checks) — proving the pipeline runs before any claim is made.
