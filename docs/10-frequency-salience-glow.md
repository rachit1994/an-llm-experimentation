# 10 — Frequency salience: the "glow" layer

The glow layer is the **product** thesis: a pattern **brightens** the more it is encountered, and
that brightness becomes a **calibrated, inspectable confidence** signal — an anti-hallucination
mechanism a local model can *show you*. This doc gives the mechanism, the payoff, and the
**mandatory** runaway fix without which the brightest concept is just the word "the."

---

## The mechanism (one encounter, three updates)

On each encounter of concept `c` (with sparse pattern `ξ_c`):

**1. Deepen the attractor (Hebbian well-deepening):**

```
W  += η · ξ_c · ξ_cᵀ          (co-activation strengthens the associative weights)
n_c += 1                      (increment the encounter count)
```

Repeated encounters make `c`'s basin of attraction **deeper and wider** — the classic Hebbian
"cells that fire together wire together," which in energy terms lowers the energy at `ξ_c`.

**2. Compute brightness (Weber–Fechner + homeostasis):**

```
π_c = normalize( log(1 + n_c) )
```

Brightness is the **log** of the count (Weber–Fechner: perception scales with the logarithm of
stimulus, so the 1000th encounter matters less than the 10th), then **normalized** across concepts
(homeostasis: total brightness is bounded, so one concept can't run away — see the fix below).

**3. Set the prior / well-depth from brightness:**

```
well-depth(c) ∝ π_c                     (brighter ⇒ deeper attractor ⇒ easier to fall into)
Born prior:   p(c) ∝ π_c   ⇔   |α_c| ∝ √π_c      (since Born rule: p = |α|²)
```

Brightness is simultaneously a **dynamical** property (deeper well) and a **Bayesian** one (higher
prior). Note the Born-consistent coupling: to get prior probability `p(c) ∝ π_c`, the *amplitude*
must scale as `√π_c`, because probability is amplitude-squared (doc 03). This keeps the glow layer
consistent with the quantum formalism rather than bolted on.

---

## Confidence = retrieval margin

Brightness alone is a prior; **confidence** is how decisively the model retrieved *this* concept
over the runner-up:

```
confidence = π_top1 − π_top2        (the retrieval margin: top-1 minus top-2 brightness/score)
```

A large margin means "one attractor clearly won"; a small margin means "two concepts are competing"
— exactly the situation where a model *should* hedge. This margin is **directly inspectable** (C8):
you can print, for any output, how close the second-best concept was. That is the anti-hallucination
signal — the model can say "I'm at margin 0.03 here, treat this as a guess."

---

## The payoff (why this is the product, not a feature)

- **Calibrated confidence.** If brightness tracks correctness (H5), the margin is a *calibrated*
  probability you can threshold: emit if margin > τ, else abstain/hedge. This is what turns a small
  local model into a *trustworthy* one.
- **Anti-hallucination for a local model.** Mainstream LLMs don't natively expose "how sure am I,
  and why." A glow model exposes both the **prior** (how often have I seen this?) and the **margin**
  (how close was the alternative?), inspectably (C8). For an **on-device** model — where you can't
  fall back to a bigger cloud model — a built-in "I don't know" is a differentiator.
- **Inspectable by construction.** Every confidence number decomposes into counts `n_c` and margins
  you can read out. No post-hoc calibration black box; the signal *is* the mechanism.

**Test (H5):** robustness rises with exposure, and glow **reduces ECE** (Expected Calibration Error)
vs a no-glow model on a reliability diagram. If brightness ⟂ correctness, we **narrow** the claim to
"salience weighting" and drop "confidence/anti-hallucination" (doc 05, Phase 3).

---

## The MANDATORY runaway fix (without this, glow learns "the")

Naive brightness `π_c ∝ n_c` has a fatal failure: the most **frequent** token dominates. In English
that is "the" — it would become the brightest, deepest, highest-prior concept, and the model would
"confidently" retrieve "the" everywhere. Frequency is **not** salience. The fix has **three
mandatory components**, all on by default:

1. **Log-compression** (already in the law): `log(1+n_c)` flattens the head so "the" (n≈10⁶) is only
   ~2× the brightness of a word seen 1000× (`log(1e6)/log(1e3) ≈ 2`), not 1000×.
2. **Divisive / homeostatic normalization:** `π_c = π̃_c / Σ_j π̃_j` (or a running homeostatic
   target) caps *total* brightness, so raising one concept lowers others — no unbounded runaway.
3. **IDF-like salience weighting:** down-weight concepts by how *broadly* they co-occur (inverse
   document/context frequency). A concept that fires in *every* context (like "the") is
   *uninformative* and gets salience ≈ 0; a concept that fires in *specific* contexts is salient.
   ```
   salience(c) ∝ log( total_contexts / contexts_containing(c) )      (IDF)
   brightness_final(c) = normalize( log(1+n_c) · salience(c) )
   ```

**Together:** log-compression flattens the head, normalization bounds the total, and IDF kills the
"fires-everywhere" concepts. Skip any one and the runaway returns. This is not optional tuning; it is
the difference between a calibrated confidence signal and a model that is loudly certain about
"the."

---

## Failure modes and guards

| Failure | Symptom | Guard |
|---|---|---|
| Frequency runaway | "the"/space is brightest | log + normalize + IDF (all three) |
| Saturation | brightness stops discriminating | Weber–Fechner log already compresses; monitor dynamic range |
| Staleness | early-seen concepts stay bright forever | decay term `n_c ← λ·n_c` (homeostasis over time) |
| Brightness ⟂ correctness | ECE not improved | H5 kill → narrow to salience-only |

---

### Interview questions this doc answers

- *"How does the model know it's confident?"* Two readouts: the **prior** `π_c ∝ log(1+n_c)`
  (how often seen) and the **margin** `π_top1 − π_top2` (how decisively retrieved) — both inspectable.
- *"Won't the most frequent word just dominate?"* Yes, unless you apply all three fixes:
  log-compression, divisive/homeostatic normalization, and IDF-like salience. Frequency ≠ salience.
- *"Why is 'inspectable confidence' valuable?"* It's an anti-hallucination / abstain signal for an
  **on-device** model that can't defer to a bigger cloud model — and it's calibrated (H5: beats
  no-glow on ECE) or we drop the claim.

### Operator's scar

The first glow prototype worked beautifully in a demo and then, on real text, became supremely
confident about the space character and "the." We'd shipped brightness ∝ raw count, no IDF. The
"most salient concept in English," per our model, was a blank space. The scar is the bold word
**MANDATORY**: the three-part runaway fix is not a tuning knob you add later — it is load-bearing,
because the whole product claim ("calibrated confidence") inverts into a punchline the moment
frequency is allowed to masquerade as salience.
