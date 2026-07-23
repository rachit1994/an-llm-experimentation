# 07 — Pattern-as-token: the objective (the deepest leg)

This is the load-bearing claim. If it fails (G0), the project stops (doc 05). It is also the most
*novel* and the most *dangerous* — the danger is **collapse** — so it gets the most careful
treatment. "Think in superposition, measure to speak": the model reasons over continuous **patterns**
and only decodes to a discrete word at emission.

---

## Next-token vs next-pattern

**Next-token (the standard objective).** Predict a distribution over a fixed vocabulary and maximize
the log-probability of the true next symbol:

```
L_token = − log p(x_{n+1} | x_{≤n})      (cross-entropy over |V| discrete symbols)
```

The target is a **one-hot** symbol; the model's whole output layer exists to score `|V|` symbols.

**Next-pattern (this experiment).** Predict the **continuous latent pattern** of the next unit and
match it to the (stop-gradient) target pattern:

```
L_pattern = D( ẑ_{n+1} ,  stopgrad( z_{n+1} ) )
```

where `z_{n+1}` is the *encoder's* pattern for the true next unit, `ẑ_{n+1}` is the *predictor's*
guess, `D` is a distance/similarity loss, and `stopgrad` prevents the target from being trained to
be trivially easy (the JEPA discipline). The target lives in the **same latent space** as the
representation, so prediction happens in "pattern space," not "symbol space."

**Why the objectives differ in kind, not degree.** Next-token forces every prediction through a
`|V|`-way bottleneck and learns from a **one-hot** signal (1 bit of "which symbol"). Next-pattern
learns from a **dense vector** target (every coordinate carries gradient) and can express "the next
thing is *between* a bee and a wasp" without committing to a symbol. That density is the thesis.

---

## The collapse trap (why this is dangerous)

`L_pattern = D(ẑ, stopgrad(z))` has a **trivial minimizer**: make the encoder output a **constant**
(or a low-rank) pattern for every input. Then `ẑ = z = const`, `D = 0`, loss is perfectly minimized,
and the model has learned **nothing**. This is *representation collapse*, and it is the default
failure of predict-your-own-latent objectives. **We instrument for it before we train anything else**
(effective rank + per-dim variance — H3, doc 06). A pattern model that "works" but collapsed is worse
than useless: it looks trained and is empty.

---

## Three fixes (each an ablation)

We do not pick one on faith; all three are switchable and compared.

### Fix 1 — JEPA-style EMA target + VICReg regularization

- **EMA target:** the target encoder is an **exponential moving average** of the online encoder
  (not trained by the loss), so the target can't collapse toward the predictor (LeCun's JEPA;
  I-JEPA/V-JEPA lineage — [`references.md`](references.md)).
- **VICReg** adds two explicit anti-collapse terms to the loss:
  - **Variance** term: hinge-penalize per-dimension std below a threshold (forces spread).
  - **Covariance** term: penalize off-diagonal covariance (decorrelates dimensions).
  ```
  L = D(ẑ, stopgrad(z)) + λ_v·Σ_j max(0, γ − std(z_j)) + λ_c·Σ_{i≠j} Cov(z)_{ij}²
  ```
  These make the constant solution *expensive*, directly attacking the trap.

### Fix 2 — InfoNCE (contrastive)

Make the true next pattern win against negatives, so a constant output is punished (all pairs look
identical → high loss):

```
L_InfoNCE = − log [ exp(sim(ẑ, z⁺)/τ) / Σ_{z⁻} exp(sim(ẑ, z⁻)/τ) ]
```

Contrastive losses have a **built-in collapse resistance** (uniformity term), at the cost of needing
negatives and a temperature `τ`.

### Fix 3 — VQ codebook (discretize the target)

Quantize patterns to a learned **codebook** (VQ-VAE style): the predictor targets a **code index**,
which cannot collapse to a constant without collapsing the codebook (guarded by commitment loss +
codebook usage/perplexity metrics). Bonus: it makes readout cheap (predict over `|codebook| ≪ |V|`),
directly helping the generation wall (doc 04). Cost: quantization error and codebook-collapse of its
own (mitigated by EMA codebook updates + dead-code reinit).

**Selection rule:** whichever fix gives **no collapse (H3) at the best perplexity** at equal params
wins Phase 1; the others are reported as ablations. If *none* avoids collapse within the perplexity
budget → **STOP** (doc 05, G0).

---

## Why pattern-as-token *can* win (three concrete reasons)

### Reason 1 — It reclaims the vocabulary table (the parameter argument)

At small scale, the vocabulary embedding table is a **huge fraction** of the parameters. Worked
example (`d = 512`, `|V| = 32000`):

| Component | Params | Share |
|---|---|---|
| Vocabulary embedding table (`\|V\|·d`) | 32000 × 512 = **16.4M** | ~**46%** |
| Transformer body (attention + MLP, few layers) | ~**18.9M** | ~54% |
| **Total** | ~**35.3M** | 100% |

Predicting in a **shared pattern space** (input and output live in the same latent, tied) means the
`16.4M`-param vocab table is **not needed** as a separate output projection — you *reclaim ~46% of
the parameters* at this scale (doc 09 quantifies the resulting 12.6M vs 29M footprint). At small
scale, that is not a rounding error; it is the ballgame.

### Reason 2 — A denser learning signal

Next-token learns from a one-hot target (1 bit: "which of `|V|`"). Next-pattern learns from a
**dense `d`-dim target**: every coordinate carries gradient every step. More bits per token → more
signal per FLOP, which is exactly what you want when compute (an M1) is the binding constraint.

### Reason 3 — Superposition capacity (Johnson–Lindenstrauss)

A `d`-dim space holds an **exponential** number of **near-orthogonal** patterns. By
Johnson–Lindenstrauss, you can pack on the order of `e^{ε²d}` vectors that are pairwise within angle
`ε` of orthogonal. So a modest `d` represents *far more distinguishable patterns than it has
dimensions* — the model can hold many concepts "in superposition" and separate them at readout. This
is the representational headroom that makes "think in patterns" plausible rather than cramped (the
capacity math is doc 08).

---

## Findings that justify betting on this

- **Coconut** (*Chain of Continuous Thought*, arXiv:2412.06769): let a model **reason in continuous
  latent space** instead of decoding to tokens between steps; it can carry **multiple reasoning paths
  in superposition** and only decode at the end. Theory follow-up: arXiv:2505.12514. Direct evidence
  that "think in patterns, measure to speak" improves reasoning efficiency.
- **Large Concept Models** (arXiv:2412.08821): predict in a **sentence-embedding (SONAR) space**
  rather than token space; **beats same-size LLMs zero-shot on multilingual tasks**. Follow-up
  **SONAR-LLM** (arXiv:2508.05305). Direct evidence that predicting *concepts/patterns* beats
  predicting tokens at matched size. *(SONAR itself is pretrained → quarantined under C1; we use LCM
  as motivation and learn our pattern space from scratch.)*
- **JEPA** (LeCun's Joint-Embedding Predictive Architecture): predict **representations, not pixels/
  tokens**, with a stop-gradient/EMA target — the blueprint for `L_pattern` and its anti-collapse
  discipline.
- **Hopfield = attention** (Ramsauer et al., *Hopfield Networks is All You Need*, arXiv:2008.02217):
  modern Hopfield networks store patterns as **attractors** and retrieve them in **one attention
  step**, with **exponential** storage capacity. This is the bridge: concepts-as-attractors (C3) and
  attention are the *same operation*, so "retrieve the concept pattern" costs one attention step,
  and the capacity worry (doc 08) is answered by this equivalence.

---

## Investment gates (don't over-invest before each is cleared)

| Gate | Question | Pass condition |
|---|---|---|
| **G0** | Does it even train without cheating? | **No collapse** (H3, relative to targets) **and** **BPB ≤ 1.10×** the param-matched byte baseline (exact protocol: [`../implementation/METRICS-AND-GATES.md`](../implementation/METRICS-AND-GATES.md)). *This is the project go/no-go.* |
| **G1** | Does thinking-in-patterns help reasoning? | **Fewer reasoning steps** to the same accuracy vs token-CoT (the Coconut claim, reproduced small). |
| **G2** | Does the pattern space transfer? | **Transfer** to a held-out task **at fixed params** beats the token baseline. |
| **G3** | Is it compositional? | Systematic **compositional generalization** (novel combinations) beats the token baseline. |

We spend Phase-1 effort only to clear **G0**. G1–G3 are *later* bets, each unlocked by the previous
— no building the compositionality harness before G0 says the objective trains at all.

---

### Interview questions this doc answers

- *"What exactly is the objective?"* `L_pattern = D(ẑ_{n+1}, stopgrad(z_{n+1}))` — predict the next
  *pattern* in a shared latent, decode to a word only at emission.
- *"What's the failure mode and how do you catch it?"* Collapse to a constant/low-rank output;
  caught by effective-rank + variance instrumentation (H3), prevented by JEPA/VICReg, InfoNCE, or VQ.
- *"Why would predicting patterns beat predicting tokens?"* Reclaims ~46% of small-model params (the
  vocab table), gives a dense (not one-hot) learning signal, and exploits `~e^{ε²d}` superposition
  capacity — with Coconut and LCM as existing same-size evidence.

### Operator's scar

Our first pattern model hit **0.0002 loss in 300 steps** and we nearly celebrated. It had collapsed:
the encoder output the same vector for every input, effective rank 1. The metric was beautiful and
the model was empty. That is why collapse instrumentation is **line one** of Phase 1, before the loss
curve is even plotted: *a predict-your-own-latent loss going to zero is evidence of collapse until
proven otherwise.*
