# 09 — Hardware requirements (everything to *decide* fits a Mac Mini M1)

The claim is narrow and computed, not hand-waved: **everything needed to decide the hypotheses fits
a Mac Mini M1** (8–16 GB unified memory). Scaling *beyond* the decision runs on standard consumer/
prosumer hardware — never a special chip. All numbers below are order-of-magnitude budgets with
their arithmetic shown (C4: numbers carry their derivation).

---

## The pattern model vs the token baseline (the headline budget)

Configuration: `d = 512`, `4` layers, `|V| = 32000` (for the token baseline's vocab).

**Token baseline (has a vocabulary table):**

```
vocab table:        |V|·d = 32000 × 512        = 16.4M params
body (4 layers):    ~ (attention + MLP)        ≈ 12.6M params
total                                          ≈ 29.0M params
```

**Pattern model (predicts in shared latent → drops the vocab table):**

```
body (4 layers, shared-latent, tied I/O)       ≈ 12.6M params
vocab table                                    = 0 (reclaimed — doc 07)
total                                          ≈ 12.6M params   →  2.3× smaller than 29M
```

**Training footprint (pattern model, fp32, Adam):**

```
params        12.6M × 4 bytes                     ≈ 50 MB
Adam m,v      2 × 12.6M × 4 bytes                  ≈ 101 MB
             ─────────────────────────────────────────────
optimizer+params                                  ≈ 151 MB   (+ activations, batch-dependent)
```

**~151 MB** for params+optimizer state sits comfortably in M1 unified memory with room for
activations and data. The pattern model is **2.3× smaller** than the token baseline *precisely
because* it drops the 16.4M vocab table (doc 07, Reason 1).

---

## Associative memory (the capacity engine, and it's tiny)

The modern-Hopfield associative store (doc 08) is a **`d × d` Hebbian matrix**:

```
d = 512:   512 × 512 × 4 bytes ≈ 1.0 MB
```

Crucially, this ~**1 MB** is **independent of the number of concepts stored**, because patterns
**superpose** in the same matrix (you don't get a new row per concept — you add into the existing
matrix). Naming 3e20 patterns (doc 08) does not cost 3e20 anything; it costs one `d×d` matrix.

**Explicit codebooks** (if a VQ variant materializes discrete codes) scale with codebook size:

```
codebook of K codes × d dims × 4 bytes:
   K = 1e4:   1e4 × 512 × 4  ≈ 0.02 GB
   K = 1e6:   1e6 × 512 × 4  ≈ 2.0  GB
```

**0.02–2 GB** for explicit codebooks — the upper end is the M1's ceiling, which is *why* we prefer
the superposed matrix and small codebooks. **We never materialize 36 billion of anything** (doc 08,
Part 4): the whole point of superposition is that capacity is not a table you allocate.

---

## Training time (measured-order estimates on M1 CPU/Metal)

| Run | Task | Order-of-magnitude time |
|---|---|---|
| **Cut A** | AG News classification (small, `|C|=4`) | **~6–15 min** |
| **Cut P/B** | char/byte-level LM (from-scratch) | **~50–125 min** |

These are *decision-grade* budgets: fast enough to sweep ≥5 seeds and several ablations in a day on
one machine, which is exactly what the fairness rules (doc 06) require. The AG News run being ~10
minutes is what makes the phase=0 ablation (H1) cheap to run many times — a prerequisite for
distinguishing a real effect from seed noise.

---

## Why the M1 is enough (and where you'd graduate)

The M1 is enough to **decide** every hypothesis because deciding requires *small, many, reproducible*
runs, not one huge run. The decision questions — does phase beat phase=0? does the pattern objective
collapse? does glow calibrate? — are all answerable at 12.6M params in minutes-to-hours.

The **scaling ladder** (for *after* the decision, if the claims survive) is deliberately boring
hardware, never a special chip:

| Rung | Hardware | Purpose |
|---|---|---|
| Decide | **Mac Mini M1** (8–16 GB) | all H1–H5 decisions, ablations, Pareto curve at small scale |
| Confirm | consumer GPU (e.g. one 12–24 GB card) / **Mac Studio** | mid-scale scaling-trend points |
| Extrapolate | **1× A100** (rented) | a few larger points to fit the scaling curve (doc 06) |

No rung needs a QPU, a cluster, or exotic silicon. If a claim needs more than one A100 to *decide*,
that is a signal the claim isn't decidable at research scale and should be re-scoped — not a signal
to buy hardware.

---

## The Metal caveat (restated, because it's the one real gotcha)

Apple **Metal has only partial complex-number support**. Consequences and mitigations:

- **Run CPU (NdArray) first** for correctness and bit-for-bit determinism (C4), then Metal for speed.
- Represent all complex state as **`(re, im)` real pairs** (doc 03) so the *same code* runs on both
  backends; a unitary complex op becomes a structured **orthogonal real** op that Metal handles fine.
- Budget for Metal being *faster but not the source of truth*: headline numbers are the CPU numbers
  (C3); Metal timings are for the Pareto/latency story, separately verified.

---

### Interview questions this doc answers

- *"Does this really fit on a Mac Mini?"* Yes: pattern model 12.6M params / ~151 MB train footprint
  (2.3× smaller than the 29M token baseline), associative memory ~1 MB, AG News ~6–15 min, char-LM
  ~50–125 min.
- *"How can capacity be huge if memory is tiny?"* The associative store is a fixed `d×d` (~1 MB)
  matrix; patterns superpose, so storage cost is independent of concept count — you never
  materialize 36B.
- *"When would you need bigger hardware?"* Only to *confirm/extrapolate* scaling after the M1
  decides; the ladder tops out at a single rented A100, never a special chip.

### Operator's scar

An early plan budgeted an explicit `1e6`-code codebook "to be safe," quietly committing **2 GB** and
turning an M1 job into an OOM lottery. Nobody had multiplied `1e6 × 512 × 4`. The scar: *every memory
line item ships with its multiplication written out*, and the default is the **superposed matrix**,
not an enumerated table — because the failure isn't "we ran out of RAM," it's "we didn't notice we'd
allocated a table the whole architecture was designed to avoid."
