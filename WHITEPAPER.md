# Amplitude Language Models

### A Classical Quantum-Formalism Architecture for Pattern-Predictive Language Modeling: Theory, Capacity Bounds, and a Falsifiable Research Program

**Author:** Rachit Srivastava¹
**¹** Independent Research. *Author and affiliation are placeholders for review and may be updated prior to publication.*

**Correspondence:** rachit.srivastava1994@gmail.com
**Version:** 1.0 · **Status:** Working paper (pre-review) · **Compute target:** Apple M1 (classical CPU/GPU)

---

## Abstract

We present the theory and a falsifiable research program for **Amplitude Language Models (ALMs)**: language models in which the unit of representation is a **complex probability amplitude** (magnitude and phase), units compose by **wave interference**, evolve by **unitary** maps, and are read out by the **Born rule**, while the predictive objective operates on **distributed activation patterns** ("pattern-as-token") rather than discrete vocabulary symbols. Every component is realized in **classical** complex linear algebra; we argue quantitatively that a fault-tolerant quantum computer is not merely unnecessary but *strictly disadvantageous* for this problem, invoking barren plateaus ($\mathrm{Var}[\partial_\theta L]\sim 2^{-\alpha n}$), dequantization, and the recently formalized simulability pincer. We contribute: (i) a self-contained derivation of the interference, unitary, and Born-readout algebra, including a proof that unitary evolution on $\mathbb{C}^d$ is structured orthogonal evolution on $\mathbb{R}^{2d}$ and hence trainable by ordinary autodiff; (ii) a formal treatment of the pattern-prediction objective, its representation-collapse failure mode, and three anti-collapse regularizers; (iii) a capacity analysis separating *naming* capacity (combinatorial and Johnson–Lindenstrauss bounds, exponential in dimension) from *storage* capacity (classical Hopfield $0.138N$ vs. modern Hopfield's exponential capacity), resolving an order-of-magnitude fallacy; (iv) a Bayesian–Hebbian "glow" mechanism yielding a calibrated, inspectable confidence signal, with a mandatory stabilization against frequency runaway; and (v) a complete parameter, memory, and FLOP budget demonstrating that all *decision-grade* experiments fit an M1 (a $12.6$M-parameter pattern model with a $\sim 151$ MB training footprint, $2.3\times$ smaller than a parameter-matched token baseline). We close with five pre-registered, individually killable hypotheses. The headline empirical precedent — a single complex-valued layer reaching $90.91\%$/$91.66\%$ on AG News versus $\sim 71.7\%$ for a single Transformer layer [1] — is treated as motivation to be re-established under strict fairness controls, not as a result we inherit.

**Keywords:** complex-valued neural networks, quantum-inspired language models, Born rule, unitary recurrence, associative memory, modern Hopfield networks, joint-embedding prediction, representation collapse, calibration.

---

## 1. Introduction

### 1.1 The token paradigm and its hidden costs

Contemporary large language models factorize text autoregressively over a fixed vocabulary $V$, maximizing $\sum_n \log p(x_{n+1}\mid x_{\le n})$ where each $x$ is a discrete symbol. Two structural costs of this paradigm are rarely stated as costs:

1. **A one-hot learning signal.** The training target at each position carries at most $\log_2|V|$ bits and, as a gradient signal, is a single "correct class" among $|V|$. The model's continuous internal state is supervised only through this discrete bottleneck.
2. **A vocabulary tax on parameters.** The input/output embedding tables occupy $|V|\cdot d$ parameters. At small scale this is not a rounding error: for $|V|=32000$, $d=512$ it is $16.4$M parameters, which we show below is **39–57%** of a $4$–$8$ layer model (§4.4).

Neither cost is fundamental to *language*; both are artifacts of choosing the discrete symbol as the unit of prediction.

### 1.2 Thesis

We test a different base. Represent each unit as a **complex amplitude** $\alpha = r e^{i\phi}$; the novelty relative to standard networks is **not** continuous magnitude (networks have always been continuous) but the orthogonal degree of freedom of **phase**, which supplies *interference*: two equal-magnitude units can reinforce or cancel depending only on their relative phase. Make the predictive unit the **distributed pattern** a concept evokes, and predict the **next pattern** in a shared latent space, decoding to a word only at emission. Store concepts as **sparse attractors** whose retrieval is a single attention step, and let patterns **brighten** with exposure to yield a **calibrated, inspectable confidence** signal.

Crucially, "quantum" here names the **mathematics** (Hilbert-space states, unitary evolution, Born-rule measurement), executed on classical hardware. §7 argues this is the correct engineering choice on the merits, not a compromise.

### 1.3 Contributions and scope

This is a **theory-and-protocol** paper. Its contributions are the derivations of §3, the objective analysis of §4, the capacity theorems of §5, the calibration mechanism of §6, the quantitative case against quantum hardware in §7, the full resource accounting of §8, and the falsification design of §9. Its scope is deliberately bounded: we make claims at **small scale** (classification and character/byte-level modeling on an M1) and treat **generation at scale** as unproven, gated behind an explicit readout-cost wall (§8.2). We do not claim, and our design cannot support, a head-to-head against frontier models; §9 defines the only comparisons we consider valid.

All experimental commitments obey a purity charter (no pretrained components, raw-byte defaults, built-from-scratch parameter-matched baselines, bit-for-bit determinism) detailed in the companion `GROUND-UP-CONSTRAINTS.md`; here we state results as *hypotheses with kill numbers*.

---

## 2. Background and related work

**Complex-valued and quantum-inspired language models.** The *Wave Network* [1] represents each token as a complex vector whose magnitude encodes global semantics and whose phase encodes the token–context relationship, updated by interference or modulation; a *single* such layer attains $90.91\%$ (interference) and $91.66\%$ (modulation) on AG News, exceeding a single Transformer layer built on BERT embeddings by $19.23$ and $19.98$ points and approaching fine-tuned BERT-base ($94.64\%$), using a $\sim 2.4$M-parameter model and reducing memory/time by $77.34\%$/$85.62\%$ versus BERT-base. Its backward dynamics are worked out in [2]. *Complex-valued Neural Network-based Quantum Language Models* [3] model semantic units (sememe→word→sentence) on a single Semantic Hilbert Space with complex-valued composition and report complex outperforming real-valued counterparts across six classification datasets. These are our closest precedents; we re-establish the *comparison* under fairness controls (§9) because [1]'s reference used pretrained embeddings.

**Unitary and tensor-network models.** Unitary-evolution RNNs [4] constrain the recurrent operator to be unitary, preserving hidden-state norm and solving copy/long-memory tasks that LSTMs fail. Matrix-product-state classifiers [5] give exponential parameter compression for low-bond-dimension weight tensors. Both are classical demonstrations that quantum *formalism* buys concrete efficiency.

**Prediction in latent/continuous space.** *Coconut* [6] feeds a model's last hidden state back as the next input embedding, reasoning in a continuous latent space that can encode multiple next steps and perform an emergent breadth-first search, improving token efficiency on planning-heavy reasoning. *Large Concept Models* [7] predict in a sentence-embedding space (SONAR) and outperform same-size LLMs zero-shot across many languages. LeCun's **JEPA** program predicts *representations* under a stop-gradient/EMA target rather than reconstructing inputs. These motivate pattern-as-token; where they use pretrained embedding spaces (e.g. SONAR) we quarantine those and learn the space from scratch.

**Associative memory.** Classical Hopfield networks [8] store $\approx 0.138N$ patterns in $N$ neurons [9]. *Modern* Hopfield networks [10] use a log-sum-exp energy with **exponential** storage capacity in the representation dimension, retrieve in one update, and have an update rule identical to Transformer attention — the bridge that makes "concepts as attractors" computationally cheap (§5).

**Why not quantum hardware.** Barren plateaus [11] show the gradient variance of wide parameterized quantum circuits vanishes exponentially in qubit number. Dequantization [12] shows many claimed QML speedups have classical algorithms only polynomially slower. The one rigorously proven advantage [13] is for a discrete-log-structured dataset in the data-scarce regime. The simulability pincer [14] collects evidence that circuits provably free of barren plateaus are also classically simulable. §7 develops these quantitatively.

---

## 3. The amplitude representation

We write a state as $\psi \in \mathbb{C}^d$, stored in implementation as a pair of real tensors $(\mathrm{Re}\,\psi, \mathrm{Im}\,\psi)\in\mathbb{R}^d\times\mathbb{R}^d$. The Hermitian inner product is $\langle a\mid b\rangle = \sum_k \overline{a_k}\, b_k$, and $\|\psi\|^2 = \langle\psi\mid\psi\rangle$.

### 3.1 Amplitudes and the Born rule

A single amplitude is $\alpha = r e^{i\phi}$ with magnitude $r\ge 0$ and phase $\phi\in[0,2\pi)$. Given a normalized state ($\|\psi\|=1$) and an orthonormal measurement basis $\{\,|e_k\rangle\,\}$, the **Born rule** assigns outcome probabilities

$$
p_k = \bigl|\langle e_k\mid\psi\rangle\bigr|^2, \qquad \sum_k p_k = \sum_k |\langle e_k\mid\psi\rangle|^2 = \|\psi\|^2 = 1 .
$$

Normalization is therefore not a convenience but the statement that the state lives on the unit sphere $S^{2d-1}\subset\mathbb{C}^d$, which pairs naturally with norm-preserving (unitary) dynamics (§3.4).

### 3.2 Interference (the algebra that phase buys)

Consider superposing two amplitudes $\psi_j = r_j e^{i\phi_j}$ (scalar case; the vector case is coordinatewise). The Born-rule intensity of the sum is derived by expanding the modulus-squared:

$$
|\psi_1+\psi_2|^2 = (\psi_1+\psi_2)\overline{(\psi_1+\psi_2)}
= |\psi_1|^2 + |\psi_2|^2 + \psi_1\overline{\psi_2} + \overline{\psi_1}\psi_2 .
$$

Since $\psi_1\overline{\psi_2} = r_1 r_2\, e^{i(\phi_1-\phi_2)}$ and $z+\bar z = 2\,\mathrm{Re}(z)$,

$$
\boxed{\;|\psi_1+\psi_2|^2 = r_1^2 + r_2^2 + \underbrace{2\,r_1 r_2\cos(\phi_1-\phi_2)}_{\text{interference term}}\;}
$$

The interference term depends **only on the relative phase** $\Delta\phi=\phi_1-\phi_2$:

| $\Delta\phi$ | $\cos\Delta\phi$ | intensity | interpretation |
|---|---|---|---|
| $0$ | $+1$ | $(r_1+r_2)^2$ | constructive (agreement/reinforcement) |
| $\pi/2$ | $0$ | $r_1^2+r_2^2$ | independent (classical addition) |
| $\pi$ | $-1$ | $(r_1-r_2)^2$ | destructive (e.g. negation/cancellation) |

Equal-magnitude units ($r_1=r_2=r$) either quadruple in intensity ($4r^2$) or annihilate ($0$) purely as a function of $\Delta\phi$. A real-valued network must *learn* cancellation with additional nonlinear capacity; a complex model obtains it in the algebra. Whether this pays off at equal parameters is hypothesis **H1** (§9).

### 3.3 Modulation (multiplicative binding)

The Hadamard (element-wise) product binds two states multiplicatively:

$$
\psi_1\odot\psi_2 = (r_1\odot r_2)\, e^{i(\phi_1+\phi_2)} ,
$$

multiplying magnitudes and **adding** phases. Interference is additive superposition (phases can cancel); modulation is a gating/binding operation (phases accumulate). Both are $O(d)$ and both were competitive in [1].

### 3.4 Unitary evolution and its real form

We evolve states by a linear map $\psi'=U\psi$ constrained to be **unitary**, $U^\dagger U = I$, so that $\|\psi'\|=\|\psi\|$ (Born normalization is preserved and, as in [4], gradients neither vanish nor explode across depth/time).

**Proposition 1 (unitary $\mathbb{C}^d$ = structured orthogonal $\mathbb{R}^{2d}$).**
Write $U = A + iB$ with $A,B\in\mathbb{R}^{d\times d}$, and represent $\psi=x+iy$ by the stacked real vector $\left(\begin{smallmatrix}x\\ y\end{smallmatrix}\right)\in\mathbb{R}^{2d}$. Then complex multiplication by $U$ is the real linear map
$$
M \;=\; \begin{pmatrix} A & -B\\[2pt] B & A\end{pmatrix},
\qquad\text{and}\qquad
U^\dagger U = I \iff M^\top M = I_{2d}.
$$

*Proof.* $U^\dagger U = (A^\top - iB^\top)(A+iB) = (A^\top A + B^\top B) + i(A^\top B - B^\top A)$, so $U^\dagger U=I$ iff $A^\top A + B^\top B = I$ and $A^\top B = B^\top A$. Compute
$$
M^\top M = \begin{pmatrix} A^\top & B^\top\\ -B^\top & A^\top\end{pmatrix}\begin{pmatrix} A & -B\\ B & A\end{pmatrix}
= \begin{pmatrix} A^\top A + B^\top B & -A^\top B + B^\top A\\ -B^\top A + A^\top B & B^\top B + A^\top A\end{pmatrix}.
$$
This equals $I_{2d}$ under exactly the same two conditions. $\qquad\blacksquare$

Consequently a unitary complex layer is a real orthogonal layer with tied $2\times 2$ block structure — implementable with real tensors and differentiated by ordinary backpropagation. We parameterize $U$ three ways:

- **Matrix exponential:** $U=e^{iH}$, $H=H^\dagger$. Exact but $O(d^3)$ (eigendecomposition) and awkward to differentiate stably.
- **Cayley transform:** $U=(I-A)(I+A)^{-1}$ with $A$ skew-Hermitian ($A^\dagger=-A$). One $O(d^3)$ solve; numerically friendly; preferred on CPU.
- **Givens/Householder products:** $U=\prod_m G_m$ of $2$-D rotations/reflections; $O(d)$ each, **inverse-free**, preferred on Metal/GPU.

**Proposition 2 (Cayley is unitary).** For skew-Hermitian $A$, $U=(I-A)(I+A)^{-1}$ satisfies $U^\dagger U=I$.
*Proof.* $(I+A)^\dagger = I - A$ and $(I-A)^\dagger = I + A$, so $U^\dagger = (I-A)^{-1}(I+A)$. Because $(I-A)$ and $(I+A)$ are polynomials in $A$ they commute, hence
$U^\dagger U = (I-A)^{-1}(I+A)(I-A)(I+A)^{-1} = (I-A)^{-1}(I-A)(I+A)(I+A)^{-1} = I.$ $\;\blacksquare$
(The transform is well defined whenever $-1\notin\mathrm{spec}(A)$; we bound the spectrum of $A$ to keep $(I+A)$ well-conditioned.)

### 3.5 Training: Wirtinger gradients, no quantum taxes

For a real loss $L$ of complex variables, the relevant update direction is the **Wirtinger derivative** $\partial L/\partial\overline{z}$, and $z \leftarrow z - \eta\,\partial L/\partial\overline{z}$. This is native in autodiff frameworks with complex support and, on the $(\mathrm{Re},\mathrm{Im})$ representation, is *literally* real-valued autodiff. We therefore pay **none** of the quantum-hardware taxes: no parameter-shift gradient estimation, no measurement shot noise, and — critically (§7.1) — no barren plateau. The only overhead is a constant factor from complex arithmetic (§8.1).

---

## 4. Pattern-as-token: the predictive objective

### 4.1 Two objectives, contrasted

**Next-token** minimizes cross-entropy against a one-hot target:
$$
L_{\text{token}} = -\log p(x_{n+1}\mid x_{\le n}), \qquad p(\cdot\mid x_{\le n})\in\Delta^{|V|-1}.
$$

**Next-pattern** predicts the continuous latent of the next unit and matches it to a stop-gradient target in the *same* space:
$$
L_{\text{pattern}} = D\!\left(\hat z_{n+1},\ \mathrm{sg}\!\left(z_{n+1}\right)\right),
$$
where $z_{n+1}=f_\theta(\text{unit}_{n+1})$ is the encoder's pattern, $\hat z_{n+1}=g_\theta(x_{\le n})$ the predictor's guess, $D$ a distance/similarity, and $\mathrm{sg}(\cdot)$ the stop-gradient. The target lives in pattern space, so prediction is dense: every coordinate of $z$ carries gradient, rather than the single "which symbol" bit of $L_{\text{token}}$.

### 4.2 The collapse failure mode (formal)

$L_{\text{pattern}}$ admits a **trivial minimizer**: any constant encoder $f_\theta\equiv c$ makes $\hat z = z = c$ and $D=0$, learning nothing. More generally the objective is minimized by any **low-rank** degenerate solution. We therefore monitor the **effective rank** (participation ratio) of the matrix $Z\in\mathbb{R}^{n\times d}$ of predicted patterns over a held-out set. If $\sigma_1\ge\cdots\ge\sigma_d\ge 0$ are its singular values,
$$
\mathrm{erank}(Z) = \frac{\left(\sum_i \sigma_i\right)^2}{\sum_i \sigma_i^2}\in[1,d],
$$
with $\mathrm{erank}=1$ signalling full collapse and $\mathrm{erank}=d$ an isotropic code. A loss curve plunging toward $0$ is, absent a rank guarantee, **evidence of collapse, not success** — this is the single most common way the objective lies to the experimenter, and instrumentation for it is wired *before* any training run (§9, H3).

### 4.3 Three anti-collapse regularizers

We treat all three as switchable and comparable:

**(a) JEPA + VICReg.** An **EMA target encoder** $\bar\theta \leftarrow \tau\bar\theta + (1-\tau)\theta$ produces the (stop-gradient) target, preventing target/predictor co-collapse. VICReg adds explicit variance and covariance terms:
$$
L = \underbrace{D(\hat z,\mathrm{sg}\,z)}_{\text{invariance}} \;+\; \lambda_v\sum_{j=1}^{d}\max\!\bigl(0,\ \gamma - \sqrt{\mathrm{Var}(z_{\cdot j})+\epsilon}\bigr) \;+\; \lambda_c\sum_{i\ne j}\bigl[\mathrm{Cov}(Z)\bigr]_{ij}^2 .
$$
The variance hinge forces each dimension's std above $\gamma$ (defeating the constant solution); the covariance term decorrelates dimensions (defeating low-rank).

**(b) InfoNCE (contrastive).** With temperature $\tau$ and negatives $\{z^-\}$,
$$
L_{\text{InfoNCE}} = -\log\frac{\exp\!\bigl(\mathrm{sim}(\hat z, z^+)/\tau\bigr)}{\sum_{z^-}\exp\!\bigl(\mathrm{sim}(\hat z, z^-)/\tau\bigr)} ,
$$
whose uniformity term makes constant outputs high-loss by construction.

**(c) VQ codebook.** Quantize targets to a learned codebook of $K$ codes (EMA updates + dead-code reinitialization); the predictor targets a code index, which cannot collapse without collapsing codebook usage (monitored via codebook perplexity). A side benefit is cheap readout over $K\ll|V|$ codes (§8.2).

**Selection rule (H2/H3):** whichever regularizer yields no collapse at the best perplexity at equal parameters proceeds; if *none* avoids collapse within the perplexity budget, the project stops.

### 4.4 Parameter reclamation: a general calculation

Consider a decoder-style body of $L$ layers at model width $d$. Standard per-layer parameter count is
$$
\underbrace{4d^2}_{W_Q,W_K,W_V,W_O} + \underbrace{8d^2}_{\text{FFN }(d\to 4d\to d)} = 12d^2 ,
$$
so the body has $12Ld^2$ parameters (layer-norm and biases are $O(d)$, negligible). A token model additionally carries a vocabulary table $|V|d$ (tied input/output). Hence the fraction of parameters spent on the vocabulary is
$$
\boxed{\,f_{\text{vocab}} = \frac{|V|d}{|V|d + 12Ld^2} = \frac{|V|}{|V| + 12Ld}\,}
$$
For $|V|=32000$, $d=512$ (so $12Ld = 6144\,L$):

| depth $L$ | body $12Ld^2$ | +vocab $|V|d$ | total | $f_{\text{vocab}}$ |
|---|---:|---:|---:|---:|
| 4 | $12.58$M | $16.38$M | $28.97$M | **56.6%** |
| 6 | $18.87$M | $16.38$M | $35.26$M | **46.5%** |
| 8 | $25.17$M | $16.38$M | $41.55$M | **39.4%** |

A **pattern model** predicting in the shared latent needs no $|V|d$ output projection; with raw-byte input its embedding is $256\cdot 512\approx0.13$M (negligible). At $L=4$ it is therefore $\approx 12.7$M versus the token model's $28.97$M — a factor
$$
28.97/12.71 = 2.28 \approx 2.3\times
$$
smaller, entirely by reclaiming the vocabulary table. At small scale this reclamation is the dominant lever, not a marginal one; the "$\sim 46\%$" headline is the $L=6$ instance of the formula above. This is the quantitative core of the pattern-as-token efficiency claim, supported by the same-size wins of [6, 7].

---

## 5. Representational capacity

We separate two questions that are routinely conflated, differing by roughly nine orders of magnitude.

### 5.1 Naming capacity (combinatorial)

*How many distinct patterns can a code with $N$ nodes represent?* For sparse $k$-of-$N$ codes the count is $\binom{N}{k}$. Take a target of $36$ billion patterns:

| scheme | nodes | count | calculation |
|---|---:|---:|---|
| localist (1 node/concept) | $3.6\times10^{10}$ | $3.6\times10^{10}$ | one node each |
| sparse $k{=}5$ ($\approx1.5\%$) | $339$ | $\binom{339}{5}\approx 3.6\times10^{10}$ | $\tfrac{339^5}{5!}\!\cdot\!\prod_{j=0}^{4}\!\bigl(1-\tfrac{j}{339}\bigr)\approx 3.62\times10^{10}$ |
| dense binary | $36$ | $2^{36}\approx 6.87\times10^{10}$ | each node $\in\{0,1\}$ |
| $16$-level analog | $9$ | $16^9=2^{36}\approx6.87\times10^{10}$ | each node $\in\{0,\dots,15\}$ |

Pushing sparsity: a $2\%$-sparse $512$-node code names
$$
\binom{512}{10} = \frac{\prod_{j=0}^{9}(512-j)}{10!} \approx \frac{512^{10}\,e^{-45/512}}{3.6288\times10^{6}} \approx \frac{(1.24\times10^{27})(0.916)}{3.63\times10^{6}} \approx 3.1\times10^{20}
$$
distinct patterns. Naming is not the bottleneck.

### 5.2 Johnson–Lindenstrauss: near-orthogonal packing is exponential in $d$

Naming assumes symbolic distinctness; for a *distributed* code we care how many patterns are mutually **distinguishable** (near-orthogonal). For independent uniform unit vectors $u,v\in S^{d-1}$, the inner product concentrates at $0$ with sub-Gaussian tails,
$$
\Pr\!\bigl[\,|\langle u,v\rangle| > \epsilon\,\bigr] \le 2\exp\!\left(-\tfrac{d\epsilon^2}{2}\right).
$$
By the probabilistic method, sampling $n$ random vectors and union-bounding over $\binom{n}{2}<n^2/2$ pairs, all pairwise inner products are $\le\epsilon$ with positive probability whenever $n^2\exp(-d\epsilon^2/2)<1$, i.e.
$$
\boxed{\,n \lesssim \exp\!\left(\tfrac{\epsilon^2 d}{4}\right).\,}
$$
The packing is exponential in $d$, with a base set by the tolerance $\epsilon$ (note the per-pair standard deviation is $1/\sqrt d$, so for $d=512$ it is $\approx0.044$ and $\epsilon$ is measured in a few of these units):

| $\epsilon$ (max $|{\langle\cdot,\cdot\rangle}|$) | $n\approx e^{\epsilon^2 d/4}$, $d{=}512$ |
|---:|---:|
| $0.2$ | $1.7\times10^{2}$ |
| $0.3$ | $1.0\times10^{5}$ |
| $0.4$ | $7.9\times10^{8}$ |
| $0.5$ | $7.9\times10^{13}$ |

This is a (loose) existence lower bound; it establishes the *headroom* that makes "hold many concepts in superposition and separate them at readout" quantitatively plausible rather than cramped. Tighter spherical-code constructions do better.

### 5.3 Storage capacity: naming $\ne$ storing

*How many patterns can be stored as **reliable attractors**, so that a noisy cue for concept $c$ retrieves $\xi_c$?* This is a strictly harder question, and the classical answer is sobering. The Amit–Gutfreund–Sompolinsky analysis of the Hopfield model [9] gives a critical capacity
$$
P_{\max} \approx 0.138\,N .
$$
To store $P=3.6\times10^{10}$ attractors classically would require
$$
N = \frac{P}{0.138} = \frac{3.6\times10^{10}}{0.138} \approx 2.6\times10^{11}\ \text{neurons},
$$
i.e. $\sim 260$ billion neurons and $N^2/2\approx 3.4\times10^{22}$ weights — infeasible by any measure. **If "store concepts as attractors" meant classical Hopfield, the entire capacity story would collapse:** hundreds of billions of nodes to *store* what a few hundred can *name*.

### 5.4 Resolution: modern Hopfield = attention

Modern Hopfield networks [10] replace the quadratic energy with a log-sum-exp energy, yielding **exponential** storage capacity in the representation dimension (for well-separated binary patterns, $\sim 2^{d/2}$), retrieval in a **single** update, and an update rule identical to attention:
$$
\mathrm{retrieve}(\xi) = X\,\mathrm{softmax}\!\bigl(\beta\,X^\top\xi\bigr), \qquad \beta = 1/\sqrt d,
$$
with $X$ the stored patterns as columns. Inverting the capacity bound, storing $P=3.6\times10^{10}$ patterns needs only
$$
2^{d/2}\ge P \iff d \ge 2\log_2 P = 2\times 35.07 \approx 70,
$$
so a representation of a few *hundred* dimensions (for margin) suffices, with an associative matrix of $d^2$ weights ($\approx 0.26$M for $d=512$, $\sim 1$ MB in fp32) that is **independent of the number of stored patterns** because patterns superpose additively. The capacity lives in the superposition, not in an enumerated table; concepts are never listed. This resolution — keep the combinatorial *naming* intuition, reject the classical $0.138N$ *storage* model, adopt modern-Hopfield/attention — is the crux of the capacity analysis and directly connects C3 to the same attention primitive used elsewhere in the model.

---

## 6. Frequency salience ("glow"): calibrated confidence

### 6.1 Hebbian deepening and a Born-consistent prior

On each encounter of concept $c$ with sparse pattern $\xi_c$ we update
$$
W \leftarrow W + \eta\,\xi_c\xi_c^\top, \qquad n_c \leftarrow n_c + 1,
$$
deepening $c$'s basin (Hebbian well-deepening) and incrementing an evidence count. We define **brightness** by a Weber–Fechner (log) law with salience weighting and homeostatic normalization (§6.3):
$$
\pi_c = \frac{\log(1+n_c)\cdot \mathrm{idf}(c)}{\sum_{c'}\log(1+n_{c'})\cdot \mathrm{idf}(c')} .
$$
Brightness is simultaneously a **prior**. To make prior probability track brightness, $p(c)\propto\pi_c$, the *amplitude* must scale as
$$
|\alpha_c| \propto \sqrt{\pi_c},
$$
because the Born rule gives $p(c)=|\alpha_c|^2$. The glow layer is thus Born-consistent by construction rather than bolted on.

### 6.2 Confidence as a retrieval margin

Brightness is a prior; **confidence** is decisiveness of retrieval — the top-1/top-2 margin,
$$
\text{conf} = \pi_{(1)} - \pi_{(2)} ,
$$
which is directly inspectable: for any emission one can print how close the runner-up was. A usable abstention rule is a single inequality: **emit if $\text{conf}>\tau$, else hedge.** For an on-device model that cannot defer to a larger cloud model, a built-in, calibrated "I don't know" is the product.

### 6.3 The frequency-runaway instability and its mandatory fix

Naive brightness $\pi_c\propto n_c$ is unstable: the most frequent token dominates. In English the argmax is the space character or *the* ($n\approx10^6$), which would become the brightest, deepest, highest-prior concept — the model would be "confident" everywhere. Three components, **all mandatory and on by default**, remove this:

1. **Log-compression** $\log(1+n_c)$ (Weber–Fechner). A token seen $10^6$ times is only $\log(10^6)/\log(10^3)=2\times$ brighter than one seen $10^3$ times, not $10^3\times$.
2. **Divisive/homeostatic normalization** $\sum_c\pi_c = 1$ (with an optional temporal decay $n_c\leftarrow\lambda n_c$), bounding total brightness so raising one concept lowers others.
3. **IDF-like salience** $\mathrm{idf}(c)=\log\bigl(\text{contexts}/\text{contexts}(c)\bigr)$: a concept firing in *every* context is uninformative and receives salience $\to 0$; a context-specific concept is salient.

Omitting any one restores the runaway. This is not tuning; it is load-bearing — the difference between a calibrated signal and a model loudly certain about a blank space.

### 6.4 Calibration metric

We evaluate calibration with the **Expected Calibration Error** over $M$ confidence bins $\{B_m\}$:
$$
\mathrm{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{n}\,\bigl|\,\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\,\bigr| .
$$
The product claim reduces to one testable inequality (H5): a glow model must achieve $\mathrm{ECE}(\text{glow}) < \mathrm{ECE}(\text{no-glow})$ on the **held-out** split; otherwise brightness is uncorrelated with correctness and we narrow the claim from "calibrated confidence" to "salience weighting."

---

## 7. Why classical, not a quantum computer

Three walls and one structural feature make a fault-tolerant quantum computer the *wrong* tool for this workload.

### 7.1 Barren plateaus

For wide parameterized quantum circuits that approximate $2$-designs, the loss-gradient variance vanishes exponentially in the qubit count $n$ [11]:
$$
\mathrm{Var}\!\left[\partial_\theta L\right] \sim 2^{-\alpha n}, \qquad \alpha>0.
$$
The consequence is a *sampling* catastrophe. Estimating a gradient component from measurements incurs shot noise of order $1/\sqrt{M}$ for $M$ circuit repetitions; resolving a signal whose standard deviation is $\sim 2^{-\alpha n/2}$ to fixed signal-to-noise requires
$$
M \gtrsim \frac{1}{\mathrm{Var}[\partial_\theta L]} \sim 2^{\alpha n}
$$
repetitions — exponentially many measurements exactly where a language model needs many parameters. Our classical complex network has ordinary, exact (to float precision) gradients with no such scaling.

### 7.2 Dequantization

A sequence of dequantization results [12] shows that several flagship QML "exponential speedups" (recommendation systems, low-rank linear algebra, PCA, certain kernels) admit classical algorithms only *polynomially* slower, once the classical algorithm is granted the same sample-and-query access the quantum algorithm implicitly assumes. For the dense, data-rich, approximately-low-rank regime of language modeling, the burden of proof for a quantum advantage is unmet.

### 7.3 The one proven advantage is not language

The rigorous exception [13] constructs a dataset for which, *assuming the classical hardness of discrete logarithm*, no classical learner beats random guessing by an inverse polynomial, while a quantum-kernel SVM classifies accurately and robustly. It is a genuine theorem — and it is a cryptographically structured, data-scarce classification problem, not natural language. Language has no discrete-log structure and is not data-scarce.

### 7.4 The simulability pincer

The deepest obstruction is a pincer [14]:
$$
\text{trainable (plateau-free)} \Rightarrow \text{classically simulable} \Rightarrow \text{no quantum advantage},
$$
$$
\text{not classically simulable} \Rightarrow \text{barren plateau} \Rightarrow \text{not trainable}.
$$
The regime that is trainable is the regime a classical machine can already reproduce. If the model trains at all, run it classically.

### 7.5 Measurement collapse: the screenshot is a classical feature

A central deliverable (§5, §6) is the **whole-state screenshot**: read every amplitude, phase, brightness, and effective-rank statistic for inspection and calibration. On real quantum hardware this is impossible — measurement collapses the state to a single sample, and full-state tomography costs a number of measurements exponential in system size. On a classical machine $\psi$ is an array of floats, readable non-destructively at any time. **The inspectability that is the product is a property the classical implementation has and the quantum one cannot.** This is a reason to prefer classical, not a concession.

---

## 8. Complexity and hardware budget

### 8.1 Arithmetic overhead of complex layers

A real linear map $y=Wx$, $W\in\mathbb{R}^{d\times d}$, costs $d^2$ multiplies and $d(d-1)$ adds, $\approx 2d^2$ FLOPs. A complex multiply requires $4$ real multiplies and $2$ real adds (naive) or $3$ multiplies and $5$ adds (Karatsuba/Gauss). Counting multiplies — the dominant cost — a complex-valued layer at equal width $d$ costs
$$
\text{FLOP}_{\mathbb{C}} \approx (3\text{–}4)\times \text{FLOP}_{\mathbb{R}} .
$$
Unitary parameterization adds an $O(d^3)$ term (Cayley) or $O(d^2)$ (Givens) per unitary layer. All overheads are polynomial, deterministic, and — as shown next — M1-affordable. Fair comparisons therefore hold *parameters* equal and report cost on the Pareto axis (§9).

### 8.2 The generation wall and its mitigations

Born-rule generation computes $p_k=|\langle e_k\mid O\psi\rangle|^2$ for every $k\in V$: an output map $O$ of $|V|d$ parameters and $|V|d$ MACs per step. For $|V|=32000,d=512$ that is $16.38$M MACs and $16.38$M parameters **per step** — the same wall a large-vocabulary softmax faces. Mitigations (each an ablation), with concrete savings:

| mitigation | mechanism | params / MACs per step | factor vs $16.38$M |
|---|---|---:|---:|
| full Born | $O\in\mathbb{R}^{|V|\times d}$ | $16.38$M | $1\times$ |
| low-rank $r{=}64$ | $O=UV^\top$, $r(|V|+d)$ | $2.08$M | $7.9\times$ |
| VQ codebook $K{=}1024$ | readout over $K$ codes, $Kd$ | $0.52$M | $31\times$ |
| hierarchical | balanced tree, $d\log_2|V|$ | $7.66$k | $\sim 2140\times$ |
| tied I/O | reuse input table | $0$ extra | (removes $16.38$M) |

Coconut-style feedback [6] further **amortizes** the wall: feeding the continuous predicted pattern back in runs generation in pattern space and pays the $O(|V|d)$ readout only at emission points, not every internal step. If no mitigation reaches parity, we narrow to encoder-only tasks (classification/retrieval), where the evidence [1,3] is already strong.

### 8.3 Parameter and memory budget (the M1 fit)

Using §4.4 at $L=4$, $d=512$: the pattern model is $\approx 12.6$M parameters. Its training footprint in fp32 with Adam (which stores first and second moments $m,v$):
$$
\underbrace{12.6\text{M}\times 4\,\text{B}}_{\text{params }50.4\,\text{MB}} + \underbrace{2\times 12.6\text{M}\times 4\,\text{B}}_{\text{Adam }m,v\ 100.8\,\text{MB}} \approx 151\,\text{MB},
$$
plus gradients ($\approx 50$ MB) and batch-dependent activations. The associative memory is a single $d\times d$ matrix, $512^2\times 4\,\text{B}\approx 1.0$ MB, independent of concept count. Explicit codebooks, if materialized, scale as $Kd\times4$ B: $K=10^4\Rightarrow 0.02$ GB, $K=10^6\Rightarrow 2.0$ GB (the latter is the M1 ceiling — hence we default to the superposed matrix and small $K$, and never materialize $3.6\times10^{10}$ of anything). All figures sit within an $8$–$16$ GB unified-memory M1.

### 8.4 Training-time estimate (shown, not asserted)

Using the standard training-FLOP estimate $C\approx 6ND$ ($N$ parameters, $D$ tokens processed) and an M1 GPU throughput of $\sim 2.6\times10^{12}$ FP32 FLOP/s:

- **AG News** ($N\approx 2.4$M as in [1]; $D\approx 1.2\times10^5$ samples $\times\ 50$ tok $\times\ 10$ ep $=6\times10^{7}$): $C\approx 6(2.4\times10^6)(6\times10^7)=8.6\times10^{14}$ FLOP $\Rightarrow \approx 330$ s $\approx 5.5$ min; at $N\approx12.6$M, $\approx 29$ min. Estimate range **$\sim 6$–$15$ min** for the small models we run.
- **Char/byte LM** ($N\approx12.6$M; $D\approx 10^8$): $C\approx 6(12.6\times10^6)(10^8)=7.6\times10^{15}$ FLOP $\Rightarrow \approx 48$ min; longer schedules push to **$\sim 50$–$125$ min**.

These are engineering budgets derived from FLOP counts, to be replaced by measured wall-clock under logged configs (they are not empirical results). Their purpose is to show that sweeping $\ge 5$ seeds $\times$ several ablations is a *single-day* activity on one machine — a precondition for the fairness protocol of §9.

### 8.5 The one real gotcha: Metal

Apple Metal has only partial complex support. We (i) treat the CPU (NdArray) backend as the source of truth for bit-for-bit-reproducible headline numbers, and (ii) represent all complex state as $(\mathrm{Re},\mathrm{Im})$ pairs so that, by Proposition 1, unitary complex ops become structured *orthogonal real* ops the GPU handles natively. Metal is a stopwatch (Pareto/latency), never the witness for a headline number.

---

## 9. Experimental protocol and falsification

### 9.1 Five hypotheses, each with a kill number

| # | hypothesis | metric | **kill number** |
|---|---|---|---|
| **H1** | complex **phase** carries information a real net cannot recover at equal params | accuracy gap: full-complex vs **phase$=0$** ablation (param-matched) | gap $\le 1\sigma$ of seed noise $\Rightarrow$ **drop "quantum"** |
| **H2** | next-pattern is competitive with next-token at equal params | val perplexity (fixed decoder) vs param-matched token baseline | pattern $>10\%$ worse **or** collapse $\Rightarrow$ **stop project** |
| **H3** | pattern prediction does not collapse | effective rank / per-dim variance | $\mathrm{erank}<0.5d$ or variance within $10\times$ constant $\Rightarrow$ **stop** |
| **H4** | a concept reliably re-triggers its attractor under noise ("bee test") | node-overlap hit-rate, clean vs noisy cue | hit-rate $<\sim90\%$ $\Rightarrow$ narrow to soft retrieval |
| **H5** | brightness tracks correctness (calibration) | ECE, glow vs no-glow on held-out | $\mathrm{ECE}(\text{glow})\ge\mathrm{ECE}(\text{no-glow})\Rightarrow$ narrow to salience |

H1 governs the *branding*; H2/H3 govern the *project*; H4/H5 govern the memory and product claims. Each fails to a specific, pre-registered number.

### 9.2 Fairness controls

(1) **Equal parameters** ($\pm 5\%$), verified by an automated counter before any run. (2) **Equal tuning budget** — identical hyperparameter-search size and early-stopping rule for both arms. (3) **Equal data, splits, loop** — fixed, hashed, leak-checked splits; same optimizer, schedule, and compute. (4) **One switch per comparison** — named ablations `phase=0`, `no_unitary`, `no_interference`, `no_modulation`, `no_glow`, `bpe` (subword is an ablation, never a default). (5) **Full config with every number** — seed, versions, git SHA; reproduces bit-for-bit on CPU. (6) **Seeds as a distribution** — report mean $\pm\sigma$ over $\ge5$ seeds.

### 9.3 The statistical decision

For H1 the estimand is $\Delta=\mathrm{metric}(\text{complex})-\mathrm{metric}(\text{phase}=0)$. Let $s$ be the pooled per-seed standard deviation. We pre-register a one-sided decision at effect size $\Delta>s$ (a "win" smaller than one seed-$\sigma$ is treated as noise). A supporting diagnostic — the phase histogram — must show phase is *used*; if the free-phase model drives $\phi\to0$ on its own, that is itself an H1 kill (the data did not want phase). The same $\sigma$-thresholded discipline applies to every headline comparison; the threshold is written down before the run to prevent goalpost-moving.

### 9.4 What "test against LLMs" means

It does **not** mean a head-to-head against frontier models; that comparison is uninformative across a $\sim 10^3\times$ scale gap and any framing implying we might win it is a credibility hazard. It means three apples-to-apples things: **(a) mechanism parity** — compare against a *parameter-matched Transformer we build ourselves* at equal data and budget; **(b) the Pareto frontier** — plot quality against cost (parameters, FLOPs, memory, latency, energy) across several small sizes; **(c) scaling extrapolation** — fit the small-scale trend and extrapolate with intervals, explicitly labeled as extrapolation, never presented as a measured large-scale result.

---

## 10. Limitations and threats to validity

1. **Generation at scale is unproven.** Every quantum-inspired win we cite [1,3] is classification/matching; the $O(|V|d)$ Born-readout wall is unbroken at large vocabulary. §8.2 offers mitigations and an honest fallback (encoder-only), not a solution.
2. **The phase advantage may not survive a fair ablation.** [1]'s reference used pretrained embeddings; a parameter- and tuning-matched real baseline may tie (H1). We have pre-committed to dropping the "quantum" framing if so.
3. **Unitary maps are linear.** A pure unitary layer cannot gate/forget; practical models reintroduce nonlinearity, partially returning the complexity the formalism promised [4]. The needed amount is an open empirical question.
4. **Anti-collapse regularizers have their own failure modes** (VICReg hyperparameter sensitivity; contrastive negative-sampling; VQ codebook collapse). H3 instruments for all of them, but "no collapse" is a continual measurement, not a one-time guarantee.
5. **Capacity bounds are existence/aggregate results.** The JL packing (§5.2) is a loose lower bound; modern-Hopfield capacity [10] depends on pattern separation $\Delta$, and real learned patterns are neither uniform nor maximally separated.
6. **Calibration can be gamed by evaluation leakage.** ECE on training-distribution data is trivially flattering; H5 is defined on held-out splits precisely to prevent this.
7. **Small-scale conclusions may not scale.** Nothing here establishes that complex/pattern/unitary structure keeps winning into the billions of parameters; §9.4 confines our claims to the frontier we can measure plus labeled extrapolation.

---

## 11. Conclusion

Amplitude Language Models recombine four established but individually promising ideas — complex-amplitude (phase-bearing) representations, pattern-space prediction, attractor memory, and Hebbian–Bayesian salience — into a single classical architecture whose every claim is instrumented and killable. The mathematics is quantum in *form* and classical in *execution*, and we have argued quantitatively that this is the correct choice: a fault-tolerant quantum computer is obstructed by barren plateaus, undercut by dequantization, squeezed by the simulability pincer, and structurally unable to deliver the inspectable full-state readout that is the product. The efficiency argument is concrete — a $2.3\times$ parameter reduction from reclaiming the vocabulary table (§4.4), an associative store whose $\sim 1$ MB is independent of concept count (§5.4), and a training regime that decides every hypothesis on an M1 in a day (§8). The honest boundary is equally concrete: generation at scale is gated behind the readout wall, and the "quantum" branding is contingent on a single pre-registered ablation. The contribution of this paper is not a benchmark victory but a *decidable* research program — five hypotheses, five kill numbers, one machine — that can be run to a definite yes or no.

---

## References

[1] X. Zhang and V. S. Sheng. **Wave Network: An Ultra-Small Language Model.** arXiv:2411.02674, 2024.

[2] X. Zhang and V. S. Sheng. **The Backpropagation of the Wave Network (Token2Wave).** arXiv:2411.06989, 2024.

[3] P. Zhang, W. Hui, B. Wang, D. Zhao, D. Song, C. Lioma, and J. G. Simonsen. **Complex-valued Neural Network-based Quantum Language Models.** *ACM Transactions on Information Systems* 40(4), Article 84, 2022. doi:10.1145/3505138.

[4] M. Arjovsky, A. Shah, and Y. Bengio. **Unitary Evolution Recurrent Neural Networks.** *ICML* 2016. arXiv:1511.06464.

[5] E. Stoudenmire and D. J. Schwab. **Supervised Learning with Tensor Networks.** *NeurIPS* 2016. arXiv:1605.05775.

[6] S. Hao, S. Sukhbaatar, D. Su, X. Li, Z. Hu, J. Weston, and Y. Tian. **Training Large Language Models to Reason in a Continuous Latent Space (Coconut).** *COLM* 2025. arXiv:2412.06769. (Theory: Z. Zhu et al., arXiv:2505.12514.)

[7] LCM Team (L. Barrault, P.-A. Duquenne, M. Elbayad, et al.). **Large Concept Models: Language Modeling in a Sentence Representation Space.** arXiv:2412.08821, 2024. (See also SONAR-LLM, arXiv:2508.05305.)

[8] J. J. Hopfield. **Neural networks and physical systems with emergent collective computational abilities.** *PNAS* 79(8):2554–2558, 1982.

[9] D. J. Amit, H. Gutfreund, and H. Sompolinsky. **Storing infinite numbers of patterns in a spin-glass model of neural networks.** *Physical Review Letters* 55(14):1530–1533, 1985. (Critical capacity $\alpha_c\approx0.138$.)

[10] H. Ramsauer, B. Schäfl, J. Lehner, et al. **Hopfield Networks is All You Need.** *ICLR* 2021. arXiv:2008.02217.

[11] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven. **Barren plateaus in quantum neural network training landscapes.** *Nature Communications* 9:4812, 2018.

[12] E. Tang. **Dequantizing algorithms to understand quantum advantage in machine learning.** *Nature Reviews Physics* 4:692–702, 2022. doi:10.1038/s42254-022-00511-w.

[13] Y. Liu, S. Arunachalam, and K. Temme. **A rigorous and robust quantum speed-up in supervised machine learning.** *Nature Physics* 17(9):1013–1017, 2021. doi:10.1038/s41567-021-01287-z. arXiv:2010.02174.

[14] M. Cerezo, M. Larocca, D. García-Martín, N. L. Diaz, P. Braccia, E. Fontana, et al. **Does provable absence of barren plateaus imply classical simulability?** *Nature Communications* 16:7907, 2025. arXiv:2312.09121.

[15] A. Bardes, J. Ponce, and Y. LeCun. **VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.** *ICLR* 2022. arXiv:2105.04906.

[16] A. van den Oord, Y. Li, and O. Vinyals. **Representation Learning with Contrastive Predictive Coding (InfoNCE).** arXiv:1807.03748, 2018.

[17] M. Assran, Q. Duval, I. Misra, et al. **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).** *CVPR* 2023. arXiv:2301.08243.

[18] A. van den Oord, O. Vinyals, and K. Kavukcuoglu. **Neural Discrete Representation Learning (VQ-VAE).** *NeurIPS* 2017. arXiv:1711.00937.

[19] W. B. Johnson and J. Lindenstrauss. **Extensions of Lipschitz mappings into a Hilbert space.** *Contemporary Mathematics* 26:189–206, 1984.

[20] J. R. Busemeyer and P. D. Bruza. **Quantum Models of Cognition and Decision.** Cambridge University Press, 2012.

[21] D. O. Hebb. **The Organization of Behavior.** Wiley, 1949.

[22] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. **On Calibration of Modern Neural Networks.** *ICML* 2017. arXiv:1706.04599. (ECE.)

[23] D. P. Kingma and J. Ba. **Adam: A Method for Stochastic Optimization.** *ICLR* 2015. arXiv:1412.6980.

[24] J. Hoffmann, S. Borgeaud, A. Mensch, et al. **Training Compute-Optimal Large Language Models (Chinchilla).** arXiv:2203.15556, 2022. (Source of the $C\approx6ND$ estimate.)

---

## Appendix A — Notation

| symbol | meaning |
|---|---|
| $\psi\in\mathbb{C}^d$ | complex state; stored as $(\mathrm{Re}\,\psi,\mathrm{Im}\,\psi)$ |
| $\alpha=re^{i\phi}$ | single amplitude: magnitude $r$, phase $\phi$ |
| $\langle a\mid b\rangle$ | Hermitian inner product $\sum_k\overline{a_k}b_k$ |
| $\odot$ | Hadamard (element-wise) product |
| $U,\ H,\ A$ | unitary map; Hermitian generator; (skew-Hermitian) Cayley argument |
| $z,\hat z$ | target pattern; predicted pattern |
| $\mathrm{sg}(\cdot)$ | stop-gradient |
| $\mathrm{erank}(Z)$ | effective rank (participation ratio) of $Z$ |
| $N,k$ | code size; number of active (sparse) nodes |
| $\pi_c,\ n_c$ | brightness of concept $c$; its encounter count |
| $|V|,d,L$ | vocabulary size; model width; depth |

## Appendix B — Key numerical results at a glance

| quantity | value | source |
|---|---:|---|
| Wave Network single-layer AG News (interference / modulation) | $90.91\%$ / $91.66\%$ | [1], §2 |
| gain over single Transformer layer | $+19.23$ / $+19.98$ pts | [1] |
| vocabulary parameter fraction, $L{=}6$ | $46.5\%$ | §4.4 |
| pattern vs token model size, $L{=}4$ | $2.3\times$ smaller | §4.4 |
| $2\%$-sparse $512$-node naming capacity | $\binom{512}{10}\approx3.1\times10^{20}$ | §5.1 |
| near-orthogonal packing, $d{=}512$, $\epsilon{=}0.5$ | $\sim7.9\times10^{13}$ | §5.2 |
| classical Hopfield nodes to store $3.6\times10^{10}$ | $\sim2.6\times10^{11}$ | §5.3 |
| modern-Hopfield dimension to store $3.6\times10^{10}$ | $d\gtrsim70$ | §5.4 |
| associative matrix size, $d{=}512$ | $\sim1$ MB | §8.3 |
| pattern-model training footprint | $\sim151$ MB | §8.3 |
| barren-plateau measurement cost | $M\sim2^{\alpha n}$ | §7.1 |
| complex-layer arithmetic overhead | $3$–$4\times$ real | §8.1 |
| Born-generation readout cost | $O(|V|d)=16.38$M MAC/step | §8.2 |
