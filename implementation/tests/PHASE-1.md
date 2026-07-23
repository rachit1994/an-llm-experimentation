# Phase 1 test suite — pattern feasibility, G0, and the anti-collapse controls

Goal: the next-pattern predictor (with the `L_inv` invariance term) trains without collapsing and
reaches BPB within 10% of a param-matched byte baseline — the **project go/no-go**. This phase is where
the negative-control battery earns its keep, because a pattern objective is exactly the kind of thing
that goes green-but-collapsed.

**Precondition:** Phase 0 DoD green. **Headline gate:** G0 (see
[`../METRICS-AND-GATES.md`](../METRICS-AND-GATES.md) §2–§3).

---

## Model cards

### T1.1 — Encoder + next-pattern predictor + byte Born head
- Files: `qilm-core/src/encoder.rs`, `qilm-train/src/model_pattern.rs`
- Signature: `enc(bytes)->z`; `predict(z_ctx)->ẑ`; a 256-way Born head (from T0.4) for BPB.
- Done-when: `test_pattern_forward_shapes` GREEN **and** `gradcheck_pattern` GREEN (finite-diff on the
  full loss, max rel err `< 1e-4`).
- Note: shape tests are necessary but *not* a DoD by themselves (threat T1).

### T1.2 — The loss with the invariance term
- Files: `qilm-train/src/loss.rs`
- Signature: `L = L_pattern + L_inv + anti_collapse + λ·L_byte_ce`, each behind a switch (C6):
  `no_invariance`, `no_stopgrad`, regularizer ∈ {jepa_vicreg, infonce, vq}.
- Done-when: `test_loss_switches` GREEN — each switch changes the computed loss deterministically.

### T1.3 — Collapse metric, measured **relative to targets**
- Files: `qilm-train/src/metrics/collapse.rs`, `qilm-train/tests/nc_collapse.rs`
- Signature: `erank(Z)`, `meanstd(Z)`, and the ratios `erank(Z)/erank(Z*)`, `meanstd(Z)/meanstd(Z*)`.
- Done-when: `nc_collapse_canary` GREEN — with `constant_encoder()`, `erank_ratio ≤ 0.05` **and**
  `gate("H3", …)` returns FAIL. *(If this ever passes on the constant encoder, the metric is blind —
  stop and fix the metric, per VERIFICATION §5.)*
- Kills-mutant: implement the old vacuous "≥10× a constant" (which is `≥0` and always true) →
  `nc_collapse_canary` must go red (a collapsed model would wrongly PASS).

### T1.4 — BPB metric + the two baselines
- Files: `qilm-train/src/metrics/bpb.rs`, `qilm-train/src/model_token.rs`,
  `qilm-train/tests/nc_bpb.rs`
- Signature: `bpb(model, held_out_bytes) -> f64`; a param-matched byte-softmax baseline.
- Done-when: `nc_untrained_bpb` GREEN — a `random_init_model()` scores `bpb ≥ 7.5` (near `log2 256=8`).
  *A low BPB from an untrained model means the eval peeks — this control catches it (threat T3).*
- Done-when (2): `nc_param_match` GREEN — pattern and token arms within `±5%` params.
- Kills-mutant: eval on the train split → `nc_untrained_bpb` red (untrained scores too low from
  memorized bias) — cross-checked by `nc_shuffled_labels` in Phase 4 for classification.

### T1.5 — Known-answer end-to-end: reach the source entropy
- Files: `qilm-train/tests/kae_markov.rs`
- Procedure: train the byte baseline **and** the pattern model on `synth_markov(order=2, seed=7)`.
- Done-when: `kae_markov_bpb` GREEN for **both** models — `H ≤ BPB ≤ H+0.15` **and** `BPB ≤ H0−0.30`,
  where `H, H0` come from the fixture's `truth.json` (VERIFICATION §7.1). *This is the test that
  replaces "loss went down" with "reached the information floor we computed by hand." A model that
  trains but stalls above `H0` fails here even if every unit test is green.*
- This test is also the `repro-smoke` CI target: clean checkout → deterministic BPB within `±0.02`.

---

## The G0 gate card

### T1.6 — Run G0 and emit the verdict (never by hand)
- Files: `runs/…` (produced), `gates.toml` (already frozen), `results/RESULTS.md` (generated)
- Procedure: train the pattern model (each regularizer) and the baseline on the real byte corpus,
  `≥5` seeds; `write_metrics` each run; `report`.
- Done-when: `results/RESULTS.md` shows, for the best regularizer, **from provenance-checked runs**:
  ```
  G0: erank_ratio ≥ 0.50  AND  meanstd_ratio ≥ 0.50  AND  bpb_ratio ≤ 1.10   →  PASS
  ```
  If PASS, proceed to Phase 2. **If FAIL, stop the project and report the measured numbers** — do not
  retune to force PASS beyond the pre-registered equal-tuning budget (C6/C7).
- Anti-fake: the PASS/FAIL is written by `report`, backed by `metrics.json` with matching `git_sha` and
  `dataset_sha256`; a hand-typed "PASS" fails `verify-results` in CI.

---

## Phase-1 negative-control set (must all be correct-signed)

| control | expected on correct code |
|---|---|
| `nc_collapse_canary` | constant encoder → FAIL the H3 gate |
| `nc_untrained_bpb` | random init → `bpb ≥ 7.5` |
| `nc_no_stopgrad_collapses` | training with `no_stopgrad` → the H3 gate FAILS on the real run (collapse) |
| `nc_determinism` | two CPU runs, same seed → identical `metrics.json` (excl. wall-clock/host) |
| `nc_param_match` | arms within ±5% params |

## Phase-1 DoD checklist

- [ ] T1.1–T1.2 forward + `gradcheck_pattern` GREEN; loss switches verified.
- [ ] T1.3 collapse metric fires on the constant encoder (`nc_collapse_canary`); the vacuous mutant is
      killed.
- [ ] T1.4 `nc_untrained_bpb`, `nc_param_match` GREEN.
- [ ] T1.5 `kae_markov_bpb` GREEN for both models (reaches the computed entropy floor); `repro-smoke`
      green.
- [ ] T1.6 G0 PASS/FAIL emitted by `report` from provenance-checked runs over ≥5 seeds; the verdict in
      `RESULTS.md` was generated, not typed.
- [ ] `cargo mutants -p qilm-train -f loss.rs -f metrics/collapse.rs -f metrics/bpb.rs`: zero survivors.
