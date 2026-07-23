# 02 — Why classical, not a quantum computer

The most common objection is "if the math is quantum, shouldn't you run it on a quantum computer?"
The answer is **no**, and not as a compromise — a real quantum computer is the *wrong tool* for
this problem. Three independent walls make that case, and a fourth observation shows that the
feature we actually want (a full-state screenshot) is something **only the classical
implementation can deliver**.

---

## Wall 1 — Barren plateaus (the gradient vanishes exponentially)

For a broad class of parameterized quantum circuits, the variance of the loss gradient **decays
exponentially** in the number of qubits:

```
Var[∂L/∂θ] ~ 2^(−α n)      (α > 0, n = number of qubits)
```

As `n` grows, the loss landscape becomes an exponentially flat plateau: gradients are
indistinguishable from zero to within shot noise, so gradient-based training **stalls** before it
starts. This is the *barren plateau* phenomenon (McClean et al., 2018, and a large follow-up
literature; see [`references.md`](references.md)). It is the quantum analogue of vanishing
gradients, but exponential in system size and not fixable by the usual tricks, because the flatness
is a property of the state space, not the optimizer.

**Consequence for us.** A language model needs *many* parameters. Precisely where a QC would need
to be big (large `n`), training becomes exponentially harder. Our classical complex-net has
ordinary, well-behaved gradients (Wirtinger calculus, native in PyTorch/Burn — doc 04).

---

## Wall 2 — Dequantization (the classical algorithm catches up)

**Reference:** Tang, *Dequantizing algorithms to understand quantum advantage in machine learning*,
Nature Reviews Physics 4, 692–702 (2022).

A series of *dequantization* results showed that several headline "quantum machine learning
speedups" (quantum recommendation systems, low-rank linear algebra, PCA, some kernel methods) have
**classical algorithms with only polynomially-worse runtime** once you give the classical algorithm
the same *sample-and-query* access to the data that the quantum algorithm implicitly assumes. The
exponential separation evaporates.

**The one honestly-proven exception.** There *is* a rigorous quantum-learning advantage
(Liu, Arunachalam & Temme, *A rigorous and robust quantum speed-up in supervised machine learning*,
Nature Physics 17, 1013–1017, 2021) — but it is for a **contrived discrete-logarithm-based
dataset** in the **data-scarce regime**. It is a real theorem, and it is *not* language: it is a
cryptographically-structured classification problem chosen precisely so a classical learner
provably cannot keep up. Natural language has no such structure, and language modeling is not
data-scarce.

**Consequence for us.** For the kind of ML we are doing (dense, data-rich, low-rank-ish structure),
the burden of proof is on anyone claiming a *quantum* advantage, and the known theorems point the
other way. Running classically is not settling; it is following the proofs.

---

## Wall 3 — The simulability pincer (you can't win both games)

**References:** Cerezo, Larocca et al., arXiv:2312.09121; *Nature Communications* (2025).

This is the deepest wall, and it is a **pincer**:

- If a variational quantum model has **no barren plateau** (trainable gradients), that same absence
  of plateau tends to imply the model lives in a **classically simulable** corner of state space —
  so a classical computer can reproduce it, and there is no quantum advantage.
- If the model is **not** classically simulable (genuinely using the exponential Hilbert space),
  it tends to **have** a barren plateau — so you **cannot train it**.

```
   trainable (no plateau)  ⇒  classically simulable  ⇒  no quantum advantage
   not classically simulable  ⇒  barren plateau  ⇒  not trainable
```

You are squeezed from both sides: the regime that is trainable is the regime a classical machine
can already handle. The pincer is why "just use a QC" is not a free lunch even in principle for
variational ML.

**Consequence for us.** If the model is trainable at all, a classical implementation can run it —
so we should just run it classically and skip the qubits, decoherence, and shot noise.

---

## Observation 4 — Measurement collapse: the classical impl is *better* at the feature we want

A central product feature (docs 08, 10) is the **whole-state screenshot**: read out the *entire*
superposition — every concept's amplitude and phase, the brightness counts, the effective rank —
for inspection and calibration (C8).

On a **real quantum computer this is impossible**: measurement **collapses** the state to a single
outcome, and you only get a sample. Reconstructing the full state (tomography) costs a number of
measurements **exponential** in the system size. The quantum machine is *structurally* unable to
hand you the screenshot you want.

On a **classical** machine, `ψ` is just an array of floats. You can read every amplitude and phase
at any time, non-destructively, for free. **The inspectability that is the product is a property of
the classical implementation, not the quantum one.** This is not a caveat — it is a reason to
prefer classical.

---

## Summary table — three walls plus one feature

| Reason | Statement | Why it favors classical |
|---|---|---|
| Barren plateaus | `Var[∂L/∂θ] ~ 2^(−αn)` | Gradients vanish exactly where a QC would need to be big; classical gradients are fine. |
| Dequantization | Tang 2022; proven advantage only on a contrived discrete-log, data-scarce task (Liu et al. 2021) | For dense, data-rich ML the separation evaporates; language has no discrete-log structure. |
| Simulability pincer | Cerezo/Larocca 2023–2025 | Trainable ⇒ simulable; not simulable ⇒ untrainable. Either way, run classically. |
| Measurement collapse | QC measurement collapses; tomography is exponential | The full-state screenshot (our product) is only feasible classically. |

---

### Interview questions this doc answers

- *"Why not run this on a quantum computer?"* Three walls — barren plateaus (`Var ~ 2^{−αn}`),
  dequantization (advantage proven only on a contrived data-scarce task), and the simulability
  pincer (trainable ⇒ simulable). A QC is strictly worse for this.
- *"Isn't there any proven quantum ML advantage?"* Yes — Liu/Arunachalam/Temme 2021 — but only for a
  cryptographic discrete-log dataset in the data-scarce regime. Language is neither.
- *"What can the classical version do that a QC can't?"* Hand you the full-state screenshot
  non-destructively; on a QC, measurement collapses the state and tomography is exponential.

### Operator's scar

An early investor asked, in good faith, whether we should "reserve quantum credits for the training
run." Saying "no" convincingly required exactly this doc — three walls plus the collapse
observation. The scar: *the case for classical has to be made proactively and with citations,
because "quantum math" primes everyone to assume "quantum hardware," and that assumption, left
uncorrected, silently doubles your imagined budget and halves your credibility with anyone who
knows the QML literature.*
