# references.md — annotated bibliography

Grouped by theme. Each entry notes **what it supports here** and its **citation hygiene** tag:
**[landmark]** = peer-reviewed / heavily-replicated foundation; **[vendor/early]** = preprint,
vendor, or early-stage result to be treated as *motivating*, not settled. Per C1, any *pretrained
artifact* below is **[quarantined]** — usable as motivation/upper-bound only, never in a controlled
run.

---

## Theme 1 — Wave networks & complex-valued quantum language models

- **Wave Network: An Ultra-Small Language Model** — arXiv:2411.02674. **[vendor/early]**
  Token = complex vector (magnitude = global semantics, phase = token↔context); updated by
  interference/modulation. Single layer: **90.91%** (interference) / **91.66%** (modulation) on
  AG News, ~19–20 pts over a single transformer layer w/ BERT embeddings, approaching BERT-base
  **94.64%**, at low VRAM/time. *Supports C1/C2 at classification scale (doc 01).*
- **Token2Wave** — arXiv:2411.06989. **[vendor/early]**
  Forward/backward dynamics and gradient flow for Wave-Network complex token representations.
  *Supports "it's trainable by standard backprop" (doc 04).*
- **Complex-valued Neural Network-based Quantum Language Models (C-NNQLM)** — P. Zhang, W. Hui,
  B. Wang, D. Zhao, D. Song, C. Lioma, J. G. Simonsen, *ACM Transactions on Information Systems*
  40(4), Art. 84, 2022, doi:10.1145/3505138. **[landmark]** Semantic units (sememe→word→sentence)
  on a single Semantic Hilbert Space; complex-valued composition; **complex beats real-valued**
  across six classification datasets. *The published analogue of H1
  (doc 01).*

## Theme 2 — Tensor networks

- **Stoudenmire & Schwab, Supervised Learning with Tensor Networks** — NeurIPS 2016,
  arXiv:1605.05775. **[landmark]** MPS gives **exponential parameter reduction** for
  low-bond-dimension weight tensors. *The efficiency precedent for docs 08–09.*

## Theme 3 — Unitary / complex neural networks

- **Arjovsky, Shah & Bengio, Unitary Evolution Recurrent Neural Networks** — ICML 2016,
  arXiv:1511.06464. **[landmark]** Unitary recurrent matrix (`U†U=I`) solves **copy/long-memory**
  tasks LSTMs fail. *The evidence for the unitary leg (doc 04, Phase 5).*

## Theme 4 — Why NOT quantum hardware

- **McClean et al., Barren plateaus in quantum neural network training landscapes** — Nature
  Communications 2018. **[landmark]** `Var[∂L/∂θ] ~ 2^{−αn}` — gradients vanish exponentially in
  qubit count. *Wall 1 (doc 02).*
- **Tang, Dequantizing algorithms to understand quantum advantage in machine learning** — Nature
  Reviews Physics 4, 692–702 (2022). **[landmark]** Many QML "speedups" have classical algorithms
  only polynomially slower. *Wall 2 (doc 02).*
- **Liu, Arunachalam & Temme, A rigorous and robust quantum speed-up in supervised machine
  learning** — Nature Physics 17, 1013–1017 (2021). **[landmark]** The one proven advantage — a
  **contrived discrete-log dataset**, data-scarce regime. *The honest exception (doc 02).*
- **Cerezo, Larocca et al., Does provable absence of barren plateaus imply classical
  simulability?** — arXiv:2312.09121; *Nature Communications* (2025). **[landmark]** The
  **simulability pincer**: plateau-free ⇒ classically simulable. *Wall 3 (doc 02).*

## Theme 5 — Pattern-as-token / JEPA / Coconut / Large Concept Models

- **Training Large Language Models to Reason in a Continuous Latent Space (Coconut)** —
  arXiv:2412.06769. **[vendor/early]** Reason in continuous latent space; multiple paths in
  superposition; decode at the end. Theory follow-up **arXiv:2505.12514**. *Motivates G1 (doc 07).*
- **Large Concept Models: Language Modeling in a Sentence Representation Space** —
  arXiv:2412.08821. **[vendor/early, quarantined (uses SONAR)]** Predict in SONAR sentence-embedding
  space; **beats same-size LLMs zero-shot multilingual**. Follow-up **SONAR-LLM** arXiv:2508.05305.
  *Motivates C2; SONAR is quarantined under C1 (docs 07, 08).*
- **LeCun, A Path Towards Autonomous Machine Intelligence (JEPA)** — position paper, 2022;
  I-JEPA/V-JEPA follow-ups. **[vendor/early]** Predict **representations, not inputs**, with
  stop-gradient/EMA targets. *The blueprint for `L_pattern` + anti-collapse (doc 07).*

## Theme 6 — Associative memory & neuroscience

- **Ramsauer et al., Hopfield Networks is All You Need** — arXiv:2008.02217. **[landmark]** Modern
  Hopfield networks = attention; **exponential** storage capacity; retrieval = one attention step
  (`X·softmax(β XᵀΞ)`). *Resolves the storage-capacity gap (doc 08); bridges C3 to attention.*
- **Hopfield, Neural networks and physical systems with emergent collective computational
  abilities** — PNAS 1982. **[landmark]** Classical associative memory; capacity **≈ 0.138N**.
  *The "naive storage" number we reject in favor of modern Hopfield (doc 08).*
- **Cell assemblies / engrams / population coding** — Hebb (1949, *The Organization of Behavior*);
  Josselyn & Tonegawa (engram cells, *Science* 2020); population-coding literature. **[landmark]**
  Concepts as distributed ensembles, strengthened by co-activation. *Neuroscience grounding for
  pattern-as-token + Hebbian glow (docs 07, 10).*
- **Busemeyer & Bruza, Quantum Models of Cognition and Decision** — Cambridge Univ. Press, 2012.
  **[landmark]** Order effects and the conjunction fallacy modeled by non-commuting projective
  measurements in Hilbert space. *The "why quantum math might fit cognition" argument (doc 00).*

## Theme 7 — Energy / cost context for LLMs

- **Energy and carbon accounting for LLMs** (e.g. Strubell et al. 2019 on NLP training cost;
  Patterson et al. 2021 on carbon). **[landmark]** Context for the on-device / efficiency value
  story. *Framing for docs 05–06 (small-scale efficiency as the defensible claim).*

---

## Citation-hygiene notes

- **Landmark vs vendor/early.** The *walls* (Theme 4) and the *foundations* (Hopfield, uRNN, MPS,
  C-NNQLM, quantum cognition) are peer-reviewed **[landmark]**. The *exciting recent directions*
  (Wave Network, Coconut, LCM) are **[vendor/early]** preprints — cited as **motivation**, and
  every one of their numbers is re-established under C7 before it enters a verdict table (doc 05).
- **Quarantine (C1).** Any pretrained artifact (SONAR in LCM; CLIP/ImageBind referenced in doc 08)
  is **[quarantined]** — motivation and upper-bound only, never a controlled number.
- **Numbers are tagged to their task.** The Wave Network 90.91/91.66% is **AG News classification**
  with **BERT embeddings** in the transformer reference — it is not a generation result and not
  produced under our fairness rules; it is a hypothesis about what we might see, not our result
  (doc 01 scar).

---

### Interview questions this doc answers

- *"What's your strongest citation for each leg?"* Complex/wave → Wave Network + C-NNQLM; unitary →
  Arjovsky et al.; capacity → Ramsauer et al. (modern Hopfield); why-not-QC → Cerezo/Larocca pincer;
  pattern-as-token → Coconut + LCM; cognition → Busemeyer & Bruza.
- *"Which of these do you actually trust?"* The **[landmark]** peer-reviewed ones as foundations;
  the **[vendor/early]** preprints as motivation only, re-verified under C7 before use.

### Operator's scar

We once let a **[vendor/early]** preprint number sit in a verdict table unqualified for a month; a
reviewer treated it as our measured result and built a forecast on it. Now every citation carries a
hygiene tag *in the bibliography itself*, and preprint numbers are physically barred from verdict
tables until reproduced — because an untagged citation is an invitation to mistake someone else's
hope for your evidence.
