# tests/ — the executable test suite, one file per phase

This directory turns [`../VERIFICATION.md`](../VERIFICATION.md) into **task cards** an implementing
agent executes in order. Read `../VERIFICATION.md` first (the architecture and the anti-fake rules),
then open the file for your current phase and work its cards top to bottom.

Index (dependency order — do not skip ahead; a card's precondition is the prior phase's DoD):

| file | builds | headline gate it must make PASS |
|---|---|---|
| [`PHASE-0.md`](PHASE-0.md) | the kernels + the anti-fake harness itself | G-stack: all KATs + gradchecks green |
| [`PHASE-1.md`](PHASE-1.md) | next-pattern predictor + invariance term + G0 | G0: no-collapse ∧ BPB ≤ 1.10× baseline |
| [`PHASE-2.md`](PHASE-2.md) | attractor memory + encoder-invariance margin | H4: within≥0.90 ∧ between≤0.30 ∧ margin≥0.50 |
| [`PHASE-3.md`](PHASE-3.md) | glow / calibrated confidence | H5: ΔECE≤−0.02 ∧ acc-guard ∧ p<0.05 |
| [`PHASE-4.md`](PHASE-4.md) | complex/phase + the decisive H1 ablation | H1: Δ≥δ_phase ∧ significant, or drop the claim |
| [`PHASE-5.md`](PHASE-5.md) | unitary dynamics + copy task | copy recall ≥0.99 where free recurrence fails |
| [`PHASE-6.md`](PHASE-6.md) | Born generation + readout mitigations | a mitigation reaches parity sub-`O(\|V\|d)` |

---

## The shared harness (build this in Phase 0, reuse everywhere)

Four pieces of infrastructure are built once, in Phase 0, and every later phase depends on them. They
are what make the anti-fake guarantees real, so they come **before** any model code.

### H1. Fixtures (`qilm-data`, deterministic, hashed)

- `synth_markov(order, seed, n) -> (bytes, truth.json)` — a known `k`-th-order Markov source; `truth.json`
  carries `entropy_bits_per_byte` (H) and `order0_entropy` (H0) computed **analytically from the
  transition matrix** (not from any model). Used by `kae_markov_bpb` (VERIFICATION §7.1).
- `synth_concepts(n_concepts, aug_family, seed) -> (items, truth.json)` — items with a **planted concept
  id** + algorithmic augmentations; `truth.json` carries the max achievable `within/between`. Used by
  `kae_invariance` (§7.2).
- `load_split(name) -> bytes` — loads a fixed split and asserts its sha256 is in `data/SPLIT_HASHES`;
  refuses to load otherwise (defeats silent data drift).
- Every fixture is content-addressed: the loader records `dataset_sha256` into the run artifact.

### H2. Negative-control harness (`qilm-train/tests/negative_controls.rs`)

Helpers to construct the deliberately-broken models the controls need: `constant_encoder()`,
`shuffle_labels()`, `random_init_model()`, `strip_term("L_inv")`, `uniform_confidence_glow()`. The
`nc_*` tests (VERIFICATION §5) use these. **A negative control passing on broken code is a bug in the
metric — fix the metric, not the control.**

### H3. Provenance + report generator (`qilm-train/src/bin/report.rs`, `gate.rs`)

- `write_metrics(run_id, metrics)` stamps `git_sha`, `config_sha256`, `dataset_sha256`,
  `code_fingerprint`, `seed`, `backend`, `wall_clock_s`, `host` (VERIFICATION §4.1).
- `report` regenerates `results/RESULTS.md` / `.json` from `runs/` + `gates.toml`, checking provenance
  and refusing to emit fabricated/stale numbers (§4.2). `report --check` is what CI diffs.
- `gate <name> <metrics.json>` reads `gates.toml`, returns exit 0 (PASS) / 1 (FAIL). Gates are code.

### H4. Frozen thresholds (`gates.toml` + `gates.lock`)

`gates.toml` holds every constant from [`../METRICS-AND-GATES.md`](../METRICS-AND-GATES.md) §7.
`gates.lock = sha256(gates.toml)`, committed. CI job `frozen-gates` asserts equality. Never edit a
threshold to pass; a threshold change is a reviewable, hash-visible commit.

---

## Task-card format (every card looks like this)

```
### T<phase>.<n> — <goal>
- Files:        <exact paths to create/edit>
- Signature:    <the fn/struct signature to implement>
- Done-when:    <named test> is GREEN  (write it first; it must be RED on the empty stub)
- Stays-red:    <named negative control> stays RED on the stub / on the broken variant
- Kills-mutant: <mutant from VERIFICATION §6> (run `cargo mutants -f <file>`; must be killed)
- DoD:          the four lines above are all satisfied; no number typed into any .md
```

Rules for the executing agent (from VERIFICATION §9, repeated because they matter):
1. **Write the test first and watch it fail on the stub.** A test that is green before you implement is
   vacuous — rewrite it.
2. **Never proceed on a green negative control.** If `nc_collapse_canary` passes, your metric is blind;
   stop and fix the metric.
3. **Never write a measured number into a Markdown file.** Run `report`; cite "see RESULTS.md".
4. **A failing gate is a finding, not a bug to hide.** Report `measured=…, threshold=…`. Do not touch
   `gates.toml`.
5. **If a number feels like a judgment call, the card is under-specified — escalate, do not invent.**

---

## CI wiring (`.github/workflows/verify.yml` — built in Phase 0)

Jobs, all required to merge:

| job | runs | fails when |
|---|---|---|
| `test` | `cargo test --workspace` | any KAT / property / `nc_*` red |
| `anti-vacuity-lint` | `bash scripts/lint_canaries.sh` | a `kat_*`/`kae_*` known-answer test has no paired canary (AGENTS.md rule 2) |
| `frozen-gates` | check `sha256(gates.toml) == gates.lock` | a threshold was changed without updating the lock |
| `verify-results` | `report --check` then `git diff --exit-code results/` | a results number was hand-edited or is stale |
| `repro-smoke` | clean checkout → run `kae_markov_bpb` on CPU, fixed seed | produced BPB ≠ ledger BPB ± 0.02 |
| `mutants` (nightly) | `cargo mutants` on changed crates | a catalog mutant survived |
| `coverage-ratchet` | `cargo llvm-cov` + count `nc_*` | coverage drops or a negative control was deleted |

Green CI = the numbers are real, the metrics can say "no", and the suite would catch a real bug. That
is the bar. Nothing merges below it.
