# Verification architecture — how "green" is made to mean "actually works"

Read this before writing any test. It exists to defeat one specific, expensive failure: **all tests
pass and the science is still broken.** Unit tests alone cannot prevent that — a suite can be entirely
green while the model has collapsed, the metric is measuring the wrong thing, the eval is peeking at
the test set, or a number in the report was typed by hand and never computed. This document specifies
the layers, the anti-fake machinery, and the negative controls that together make a green build mean
*the mechanism works and the reported numbers were earned*.

Audience: an implementing agent (Sonnet/Haiku) who will **follow** this. Every requirement here is
mechanical — a named test, an exact assertion, an exact file. Where a number appears (e.g. `1e-4`,
`7.5 bits/byte`), it is a pre-registered constant, not a suggestion. Do not invent numbers; do not
edit generated results; do not lower a threshold to get green. If a gate fails, the finding is that
the gate failed — report it, do not "fix" it by weakening it.

---

## 1. Threat model — the seven ways a green suite lies

| # | Lie | Example | Defeated by |
|---|---|---|---|
| T1 | **Vacuous assertion** | test checks tensor *shape*, not *value* | known-answer tests (§3), mutation testing (§6) |
| T2 | **Metric measures nothing** | "collapse" metric returns OK on a collapsed model | negative controls (§5): collapse canary |
| T3 | **Eval peeks** | val/test bytes leak into train; ECE on train split | leakage controls (§5): untrained-BPB, shuffled-label |
| T4 | **Trivially satisfiable** | invariance "passes" because the encoder collapsed | discrimination margin + `no_invariance` control (§5) |
| T5 | **Number is fabricated** | report says `BPB = 1.2`, no run produced it | provenance chain + generated reports (§4) |
| T6 | **Number is stale/mismatched** | report from an old commit or a different dataset | provenance hashes checked at report time (§4) |
| T7 | **Threshold slid to pass** | someone changed `1.10` to `1.40` to get PASS | frozen `gates.toml` + `gates.lock` in CI (§4) |

Every layer below maps to one or more of these. A test that does not defeat a named threat is
decoration.

---

## 2. The five-layer pyramid (bottom carries the weight)

```
        ┌─────────────────────────────────────────────┐
   L4   │  Headline runs → gate numbers (provenance)   │   the claims
        ├─────────────────────────────────────────────┤
   L3   │  Known-answer END-TO-END on synthetic data   │   "achieves the number", not "loss went down"
        ├─────────────────────────────────────────────┤
   L2   │  NEGATIVE CONTROLS (must fail on broken code) │   defeats T1–T4  ← the layer people skip
        ├─────────────────────────────────────────────┤
   L1   │  Property / metamorphic tests (all inputs)   │   invariants: norm, Σp=1, bind∘unbind
        ├─────────────────────────────────────────────┤
   L0   │  Known-answer unit tests (oracle + grad)     │   exact values, finite-diff gradients
        └─────────────────────────────────────────────┘
```

Unit tests (L0) are necessary and **not sufficient** — they live at the bottom because a model can pass
every L0 test and still not learn. L2 (negative controls) and L3 (known-answer end-to-end) are the
layers that catch "green but broken", and they are **mandatory**, not optional.

---

## 3. L0 — Known-answer tests (KATs), not shape checks

Every kernel ships with a test whose **expected value is computed independently** (by a one-line NumPy
snippet recorded in the test as a comment, or by a closed-form identity), never by running the kernel
and pasting its output back in (that would test nothing — T1). Required KATs:

- `kat_interference`: `|ψ1+ψ2|²` for two fixed unit amplitudes equals the hand-computed
  `r1²+r2²+2r1r2·cos(Δφ)`, tol `1e-6`.
- `kat_binding_identity` (Prop 0): `circular_conv(a,b) == ifft(fft(a) ⊙ fft(b))`, random `a,b`, tol `1e-6`.
- `kat_unbind`: `unbind(bind(a,b), b) ≈ a` in cosine, tol `≥ 0.98` at `d=1024`.
- `kat_born_sum`: Born readout probabilities sum to `1`, tol `1e-6`.
- `kat_unitary_norm`: `‖Uψ‖ == ‖ψ‖` for Cayley and Givens `U`, tol `1e-6`.
- `kat_cayley_unitary`: `U†U == I`, tol `1e-6`.
- `gradcheck_<kernel>`: analytic grad vs central finite difference `(L(θ+ε)−L(θ−ε))/2ε`,
  `ε=1e-4`, **max relative error `< 1e-4`**, for every kernel with parameters.

The finite-difference gradient check is the single highest-value L0 test: it catches the class of bug
(a wrong backward pass) that produces plausible-looking-but-wrong training and therefore fake numbers.

---

## 4. Anti-fake machinery — numbers can only enter a report by being computed

This is the defense against T5/T6/T7 and is **non-negotiable**. The rule: *no number is ever written
into a Markdown file by a human or an agent.* Numbers live in run artifacts; documents display them
only via a generator.

### 4.1 Every run emits a signed artifact

A training/eval run writes `runs/<run_id>/metrics.json` where `run_id = sha256(config.json)[:16]` and
the file contains, at minimum:

```json
{
  "git_sha": "…",             // HEAD at run time
  "config_sha256": "…",       // hash of the exact config
  "dataset_sha256": "…",      // hash of the exact train/val/test bytes actually loaded
  "code_fingerprint": "…",    // hash of qilm-core + qilm-train source (or the built binary)
  "seed": 0,
  "backend": "ndarray-cpu",
  "wall_clock_s": 812.4,
  "host": "…",
  "metrics": { "bpb_deploy": 1.83, "erank_ratio": 0.71, "within": 0.94, "between": 0.18, "ece": 0.031, "…": 0 }
}
```

### 4.2 Reports are generated, never edited

`cargo run -p qilm-train --bin report` reads `runs/`, `data/SPLIT_HASHES`, and `gates.toml`,
**recomputes** every derived number, evaluates every gate, and writes `results/RESULTS.md` +
`results/RESULTS.json`. The first line of `RESULTS.md` is:

```
<!-- GENERATED by `qilm-train report` — DO NOT EDIT. Edits are reverted by CI (verify-results). -->
```

The generator **refuses to emit** (non-zero exit) if, for any run it is asked to report:
- `git_sha` ≠ current `HEAD` (or a committed release tag) — defeats T6 (stale commit).
- `dataset_sha256` ∉ the committed `data/SPLIT_HASHES` — defeats T6 (wrong/leaked data).
- a gate is marked PASS but no `metrics.json` with matching provenance backs it — defeats T5 (fabrication).

### 4.3 Thresholds are frozen

`gates.toml` holds every pre-registered constant (from `METRICS-AND-GATES.md` §7). A committed
`gates.lock` = `sha256(gates.toml)`. CI job `frozen-gates` asserts they match. To change a threshold
you must update `gates.lock` in the same commit — a **visible, reviewable** act, never a silent slide
(defeats T7). An agent that fails a gate must report the failure, not edit `gates.toml`.

### 4.4 CI enforces "reproduce or it didn't happen"

- `verify-results`: re-runs `report --check`, which regenerates `RESULTS.*` and diffs against the
  committed copy. **Any** drift — including a single hand-edited digit — fails CI (defeats T5).
- `repro-smoke`: from a clean checkout, on CPU with a fixed seed, runs the L3 synthetic end-to-end
  (§7 fixture) and asserts the produced BPB matches the ledger value within `±0.02 bits/byte`,
  **deterministically**. This proves the pipeline actually runs and produces that *kind* of number
  from raw data — a stored constant cannot pass because the run recomputes it.

---

## 5. L2 — The negative-control battery (must be RED on broken code)

These are the tests that make "green" trustworthy. Each is written so that **a correct implementation
passes and a specifically-broken one fails.** They live in `qilm-train/tests/negative_controls.rs` and
are prefixed `nc_`. The whole battery runs in CI on every push. Required controls:

| control | procedure | assertion (correct code) | the lie it catches |
|---|---|---|---|
| `nc_untrained_bpb` | eval a **random-init** model | `bpb ≥ 7.5` (≈ log2 256 = 8) | T3 — eval peeking / BPB miscomputed |
| `nc_shuffled_labels` | train AG News with labels permuted | test acc ∈ `[0.23, 0.27]` (chance 0.25) | T3 — label leakage |
| `nc_collapse_canary` | force a **constant** encoder | `erank_ratio ≤ 0.05` **and** collapse gate = FAIL | T2 — collapse metric blind |
| `nc_collapsed_invariance` | constant encoder | `within ≈ between`, `margin < 0.10`, H4 gate = FAIL | T4 — invariance won by collapse |
| `nc_invariance_needs_Linv` | train with `no_invariance` | H4 gate = FAIL (`margin < 0.50`) | T4 — invariance is trivial/incidental |
| `nc_glow_accuracy_guard` | glow variant that lowers all confidences | H5 gate = FAIL (accuracy guard trips) | T4 — calibration bought by dumbing down |
| `nc_determinism` | run twice, same seed, CPU | `metrics.json` identical (excl. `wall_clock_s`, `host`) | reproducibility claim false |
| `nc_split_disjoint` | hash every doc in each split | train ∩ val ∩ test = ∅; no split reused | T3 — split leakage |
| `nc_param_match` | count params of both arms | within `±5%` | C7 — unfair baseline |
| `nc_born_sum`, `nc_unitary_norm` | (L0 identities, also run here) | as §3 | T1 |

**How to read a negative control:** it is a *good sign when the "broken" version fails*. If
`nc_collapse_canary` ever passes (a constant encoder scores well), the **metric** is broken, not the
model — stop and fix the metric before trusting any real number. The negative controls test the
*tests*.

---

## 6. Mutation testing — proof the suite is not vacuous

A green suite is only meaningful if it would have gone **red** on a real bug. We prove this with
`cargo-mutants` (CI job `mutants`, allowed to run nightly given cost) and with a **hand-curated mutant
catalog** that must be killed by name. For each mutant, the named test that must fail is listed; if the
mutant survives (tests still green), the suite has a hole and a test must be added.

| mutant (deliberate bug) | test that MUST go red |
|---|---|
| flip a sign in complex multiply (`−bd` → `+bd`) | `kat_binding_identity`, `kat_interference` |
| drop the `stopgrad` in `L_pattern` | `nc_collapse_canary` on a *real* run (collapse fires) |
| remove Born normalization | `kat_born_sum` |
| use the **train** split for eval | `nc_untrained_bpb` (too low) + `nc_shuffled_labels` (too high) |
| skip the `L_inv` term | `nc_invariance_needs_Linv` |
| compute ECE with `M=1` bin | `test_ece_bins` (bin-count guard) |
| off-by-one in the param counter | `nc_param_match` |
| return a constant from the encoder | `nc_collapse_canary`, `nc_collapsed_invariance` |
| Cayley without the skew-Hermitian constraint | `kat_cayley_unitary`, `kat_unitary_norm` |

**Definition-of-done for a phase's test suite includes killing every mutant in its row-set** (§8).
A suite that cannot kill "return a constant from the encoder" is worthless for this project.

---

## 7. L3 — Known-answer end-to-end on synthetic data (the anti-self-deception layer)

> **MANDATORY, learned from a real defect (see AGENTS.md scar).** Two rules govern every known-answer
> test (`kat_*`, `kae_*`), enforced by the `anti-vacuity-lint` CI job (§9 wiring) and the DoD (§8):
>
> 1. **The expected value must be computed *independently* of the artifact under test.** The truth for
>    a data fixture comes from the parameters that *define* the source (e.g. the transition matrix `P`
>    and its stationary distribution), **never** from counting the sample you generated. If "expected"
>    and "actual" share a computation, the difference is structurally `0` and the test proves nothing.
> 2. **Every such test ships a paired anti-vacuity canary** — a sibling test (name contains
>    `can_say_no` / `vacuit` / `canary`) that feeds a *wrong* input and asserts the check *fails* by a
>    wide margin. A known-answer test with no canary is not evidence and does not satisfy the DoD.
>
> Tell to watch for: an analytic-vs-empirical comparison returning **exactly** `0.000000`. That is a
> self-comparison bug, not a success — investigate before believing it.

"Loss went down" is not evidence the model works; the loss can go down while the model memorizes noise
or collapses. The fix is data whose **correct answer is known before the model runs**, computed by the
data generator, not the model.

### 7.1 The synthetic Markov corpus (Phases 1, 3, 4, 5)

`qilm-data` includes a generator `synth_markov(order=k, seed=S)` that emits bytes from a **known**
`k`-th-order Markov source and writes `fixtures/synth/truth.json`:

```json
{ "entropy_bits_per_byte": 1.83, "order0_entropy": 4.91, "order": 2, "seed": 7, "n_bytes": 2000000 }
```

`entropy_bits_per_byte` (`H`) and `order0_entropy` (`H0`) are computed **analytically from the
transition matrix**, independent of any model. The end-to-end test `kae_markov_bpb`:

```
PASS ⟺  H ≤ BPB_model ≤ H + 0.15          (model approaches the true source entropy)
        AND BPB_model ≤ H0 − 0.30          (model beats the order-0 baseline by a known margin)
```

This is the single most important test in the project: it replaces "the loss looks good" with "the
model reached the information-theoretic floor of a source whose entropy we computed by hand." A model
that trains but plateaus above `H0` is broken, and this test says so with an absolute number no one
typed in.

### 7.2 The planted-concept invariance fixture (Phases 1, 2)

`synth_concepts(n_concepts, aug_family, seed)` emits items each carrying a **planted concept id** and a
set of algorithmic augmentations of it. Because the construction is known, the generator writes the
**maximum achievable** `within`/`between` for a perfect encoder to `truth.json`. `kae_invariance`
asserts the trained encoder reaches `within ≥ 0.90`, `between ≤ 0.30`, `margin ≥ 0.50` (the H4 gate) on
**held-out** concept ids — and, as a control, that shuffling the planted ids destroys the margin
(proving the encoder is using real structure, not an artifact).

### 7.3 The copy task (Phase 5)

The Arjovsky copy/recall task has a known-solvable structure: `kae_copy` asserts the unitary layer
reaches `≥ 0.99` recall at a delay `T` where the `no_unitary` control (a free recurrence) stays at
chance — a differential result, not an absolute one.

---

## 8. Definition-of-Done (DoD) — the per-phase gate an agent must clear

A phase is **not done** — regardless of how green `cargo test` looks — until **all** of the following
are true and shown in `results/RESULTS.md` (generated, §4.2):

1. **L0 green**: every KAT and every `gradcheck_*` for the phase's kernels passes (§3).
2. **L1 green**: every property/metamorphic test passes.
3. **L2 correct-signed**: every `nc_*` in the phase's set passes (i.e. the broken versions fail as
   designed) (§5).
4. **Mutants killed**: every mutant in the phase's row-set (§6) is killed by a named test; `cargo
   mutants` reports zero *survivors* in the phase's crates (or each survivor is triaged with an added
   test).
5. **L3 known-answer met**: the phase's synthetic end-to-end (§7) reaches its **computed** target
   (e.g. `kae_markov_bpb` within `[H, H+0.15]`).
6. **Gate PASS from provenance**: the phase's headline gate (from `METRICS-AND-GATES.md`) evaluates
   PASS in the **generated** `RESULTS.md`, backed by a `metrics.json` whose provenance checks pass
   (§4.2), over the required `≥ 5` seeds.
7. **Coverage ratchet**: line coverage of the phase's crate does not decrease, and the count of `nc_*`
   controls does not decrease (deleting a canary fails CI).

If 1–4 are green but 5 or 6 fail, **the code is not working** even though the unit tests pass — this is
exactly the situation the user warned about, and the DoD makes it a visible FAIL, not a silent pass.

---

## 9. The working loop for an implementing agent (Sonnet/Haiku)

Follow this loop per task card (task cards live in `tests/PHASE-N.md`). Do not deviate.

1. **Read** the task card: it names the file, the function signature, the test to make green, and the
   negative control that must stay red on a stub.
2. **Write the test first** (or confirm it exists) and run it — it must **fail** on the empty stub
   (a test that passes on a stub is vacuous; delete and rewrite it).
3. **Implement** the function until the KAT/property test is green.
4. **Run the negative controls** for the module. If a control that should be red is green, your metric
   or your test is wrong — fix that before proceeding. Never proceed on a green negative control.
5. **Run the mutant(s)** listed for the task (`cargo mutants -f <file>`); if any survives, add a test.
6. **Never** write a number into a Markdown file. To see a result, run `qilm-train report` and read
   the generated `RESULTS.md`. If you need a number in prose, cite it as "see RESULTS.md" — do not
   transcribe it (transcription drift fails `verify-results`).
7. **If a gate fails**, stop and report "Gate X FAILED: measured=…, threshold=…". Do not change the
   threshold, do not delete the test, do not narrow the metric. A failed gate is a finding, which is
   the whole point of the gate.

Rule of thumb for the division of labor: **the thresholds, the metric definitions, the negative
controls, and the fixtures are fixed here (thinking); implementing the kernels and making the named
tests green is the follow-on work (following).** An agent should never need to *decide* a number — if
it feels like it must, that is a signal the task card is under-specified; escalate rather than invent.

---

## 10. What this does and does not guarantee (honesty)

It guarantees: a reported number was computed by the pipeline from hashed data at a known commit; the
metric that produced it was itself validated by a negative control; the test suite would have caught a
real bug (mutants); and the model reached an *independently-known* target on synthetic data. It does
**not** guarantee the model will hit the headline numbers on real data — that is what the runs are
for, and the honest outcome may be a documented FAIL (project stop or claim-narrow). Verification makes
the *outcome trustworthy*; it does not make the outcome *positive*. Confusing the two is the last way a
green suite lies, and we refuse it: **we ship the true number, whatever it is.**

---

### Interview questions this doc answers

- *"Your tests are green — how do I know the model actually works?"* Because green requires the L2
  negative controls (a collapsed model *fails*), the L3 known-answer end-to-end (BPB reaches a source
  entropy we computed by hand), and a mutant-killed suite — unit tests are only the bottom layer.
- *"How do I know a reported number wasn't made up?"* Numbers exist only in `runs/*/metrics.json` with
  a git/dataset/code hash; reports are *generated*, and CI regenerates-and-diffs, so a hand-typed digit
  fails `verify-results`. Thresholds are hash-frozen so none can be slid to pass.
- *"How is the plan followable by a weaker agent?"* Every task card names the file, the signature, the
  test that defines done, and the negative control that must stay red — no number is ever a judgment
  call; if it feels like one, the card is under-specified and the agent escalates.

### Operator's scar

We once had 240 green tests and a model that had silently collapsed in week two; the suite checked
shapes, that loss decreased, and that the API didn't panic — all true, all useless. The collapse metric
existed but was only asserted to "run without error," never asserted to *fire on a collapsed model*.
The scar is L2 and the mutant catalog: **a test that has never been shown to fail on broken code is not
evidence.** Every metric in this project now has a negative control that proves it can say "no", and
every phase must kill "return a constant from the encoder" before its numbers are believed.
