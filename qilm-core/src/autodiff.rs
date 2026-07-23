//! autodiff — a tensor-valued, index-based reverse-mode tape (Wengert list),
//! Phase 1's foundational engine (implementation/tests/PHASE-1.md; reference:
//! implementation/PRIOR-ART-AND-REUSE.md "Reverse-mode autodiff for Phase 1").
//!
//! Ports the ARCHITECTURE of karpathy/micrograd — topological reverse pass +
//! per-node local backward — but NOT its scalar granularity: micrograd is one
//! node per scalar add/mul, far too slow for a 12.6M-param model. Here each
//! `Node` holds a whole tensor (`Vec<f64>` + a `Shape`), forward appends nodes
//! to an arena (`Vec<Node>`), and `backward()` walks the arena in reverse
//! accumulating gradients. Index/arena beats `Rc<RefCell>`: no interior
//! mutability, cache-friendly, and a node's inputs are always at lower indices
//! than itself (the tape IS the topological order — no separate sort needed).
//!
//! Every tensor here is 2-D, row-major: `shape = (rows, cols)`. A "vector" is
//! a `(1, n)` row. This is enough to express `linear` (`x·W + b`, `x: (batch,
//! in)`, `W: (in, out)`, `b: (1, out)` broadcast over the batch) and
//! `log_softmax`/`cross_entropy` (row-wise over the batch), which is all
//! Phase 1 needs (no rank>2 tensors, no conv).
//!
//! OPS FOR THIS INCREMENT (enumerated, nothing else): `add` (elementwise,
//! optional row-broadcast bias), `matmul`/`linear` (`x·W+b`), `tanh`, and the
//! numerically-stable `log_softmax`+`cross_entropy` pair BPB needs. Plus one
//! small reduction, `sum_squares`, built first as the minimal scalar-izer that
//! proves the arena/backward machinery itself (leaf -> loss -> backward) before
//! any real op is layered on top, and that carries the mandated property test
//! (backward of a sum-of-squares loss w.r.t. a leaf is `2·leaf`) — it is not
//! part of the model-facing API the other four ops form.
//!
//! Phase-1 addition — `born_logits` (T1.1): the differentiable Born-rule byte
//! head. On real per-class amplitudes `a` the Born distribution is
//! `p_i = a_i² / Σ_j a_j²`, which is *exactly* `softmax(ln a_i²)`. So the head
//! is `cross_entropy(log_softmax(born_logits(a)), target)` — reusing the
//! already-gradchecked softmax/CE backward — where `born_logits(a) = ln(a² + ε)`
//! is the one new elementwise op (ε a small stability floor so `ln` and its
//! gradient stay finite as `a → 0`). This mirrors `born.rs`'s squared-magnitude
//! -> normalize semantics (T0.4), expressed in the tape's log domain.
//!
//! CORRECTNESS: every op here is validated in `qilm-oracle/tests/` by a
//! `gradcheck_autodiff_*` test against a hand-derived, tape-independent oracle
//! (closed-form forward + closed-form analytic gradient), cross-checked
//! against `qilm_oracle::gradcheck`'s central finite difference (the fully
//! independent numeric oracle) AND against this tape's own forward/backward
//! output. A wrong local derivative in any `backward_node` arm fails that op's
//! gradcheck at `max_rel_err < 1e-4` — see VERIFICATION.md §3.
#![allow(dead_code)]

/// Stability floor for `born_logits`: `ln(a² + BORN_EPS)`. Keeps the log and
/// its derivative `2a/(a² + ε)` finite as an amplitude approaches zero, at the
/// cost of flooring each Born probability by `ε / Σ_j(a_j² + ε)` (negligible
/// for `a` of order 1). Fixed, not tunable — it is part of the op's definition,
/// so the gradcheck oracle uses the same constant.
pub const BORN_EPS: f64 = 1e-6;

/// Row-major 2-D shape. A vector is `Shape { rows: 1, cols: n }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}

impl Shape {
    pub fn mat(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    pub fn row(n: usize) -> Self {
        Self { rows: 1, cols: n }
    }

    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Index into the tape's node arena. Nodes are only ever created in
/// topological order (an op's inputs are always created, hence indexed,
/// before the op's own output node), so `NodeId` order IS reverse-pass order.
pub type NodeId = usize;

#[derive(Clone, Debug)]
enum Op {
    Leaf,
    SumSquares {
        x: NodeId,
    },
    Add {
        a: NodeId,
        b: NodeId,
        /// true when `b` is a `(1, cols)` bias broadcast over `a`'s rows.
        broadcast_b: bool,
    },
    MatMul {
        a: NodeId,
        b: NodeId,
    },
    Tanh {
        x: NodeId,
    },
    BornLogits {
        x: NodeId,
    },
    LogSoftmax {
        x: NodeId,
    },
    CrossEntropy {
        logp: NodeId,
        /// one target class index per row of `logp`.
        targets: Vec<usize>,
    },
}

struct Node {
    value: Vec<f64>,
    shape: Shape,
    grad: Vec<f64>,
    op: Op,
}

/// The arena / Wengert list. Build a graph by calling the op methods (each
/// appends a node and returns its `NodeId`); call `backward(root)` to
/// populate every node's `grad` buffer with `d(root)/d(node)`.
pub struct Tape {
    nodes: Vec<Node>,
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

impl Tape {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn push(&mut self, value: Vec<f64>, shape: Shape, op: Op) -> NodeId {
        assert_eq!(
            value.len(),
            shape.len(),
            "Tape::push: value length {} != shape {:?}",
            value.len(),
            shape
        );
        let grad = vec![0.0; value.len()];
        self.nodes.push(Node {
            value,
            shape,
            grad,
            op,
        });
        self.nodes.len() - 1
    }

    /// Create a leaf (input/parameter) node holding `value` with the given
    /// `shape`. Leaves have no inputs; `backward` stops here.
    pub fn leaf(&mut self, value: Vec<f64>, shape: Shape) -> NodeId {
        self.push(value, shape, Op::Leaf)
    }

    pub fn value(&self, id: NodeId) -> &[f64] {
        &self.nodes[id].value
    }

    pub fn grad(&self, id: NodeId) -> &[f64] {
        &self.nodes[id].grad
    }

    pub fn shape(&self, id: NodeId) -> Shape {
        self.nodes[id].shape
    }

    /// Scalar reduction `L = sum_i x_i^2`. Not one of Phase 1's four model
    /// ops; exists to (a) turn any tensor-valued node into a backward-able
    /// scalar loss for gradcheck tests, and (b) carry the mandated property
    /// test (`d(sum x_i^2)/dx_i = 2*x_i`).
    pub fn sum_squares(&mut self, x: NodeId) -> NodeId {
        let loss: f64 = self.nodes[x].value.iter().map(|v| v * v).sum();
        self.push(vec![loss], Shape::mat(1, 1), Op::SumSquares { x })
    }

    /// Elementwise add. If `b`'s shape equals `a`'s, this is a plain
    /// elementwise add. If `b` is `(1, cols)` and `a` is `(rows, cols)`, `b`
    /// is broadcast as a bias row over every row of `a` (this is the "+b" in
    /// `linear`). Any other shape mismatch panics.
    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_shape = self.nodes[a].shape;
        let b_shape = self.nodes[b].shape;

        if a_shape == b_shape {
            let value: Vec<f64> = self.nodes[a]
                .value
                .iter()
                .zip(&self.nodes[b].value)
                .map(|(x, y)| x + y)
                .collect();
            self.push(
                value,
                a_shape,
                Op::Add {
                    a,
                    b,
                    broadcast_b: false,
                },
            )
        } else if b_shape.rows == 1 && b_shape.cols == a_shape.cols {
            let cols = a_shape.cols;
            let mut value = self.nodes[a].value.clone();
            let bias = &self.nodes[b].value;
            for r in 0..a_shape.rows {
                for c in 0..cols {
                    value[r * cols + c] += bias[c];
                }
            }
            self.push(
                value,
                a_shape,
                Op::Add {
                    a,
                    b,
                    broadcast_b: true,
                },
            )
        } else {
            panic!(
                "Tape::add: shapes {a_shape:?} and {b_shape:?} are neither equal nor \
                 bias-broadcastable (b must be (1, a.cols))"
            );
        }
    }

    /// Matrix multiply: `a: (m,k)`, `b: (k,n)` -> `(m,n)`.
    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_shape = self.nodes[a].shape;
        let b_shape = self.nodes[b].shape;
        assert_eq!(
            a_shape.cols, b_shape.rows,
            "Tape::matmul: inner dims mismatch, a={a_shape:?} b={b_shape:?}"
        );
        let (m, k, n) = (a_shape.rows, a_shape.cols, b_shape.cols);
        let a_val = &self.nodes[a].value;
        let b_val = &self.nodes[b].value;
        let mut value = vec![0.0; m * n];
        for i in 0..m {
            for kk in 0..k {
                let aik = a_val[i * k + kk];
                if aik == 0.0 {
                    continue;
                }
                for j in 0..n {
                    value[i * n + j] += aik * b_val[kk * n + j];
                }
            }
        }
        self.push(value, Shape::mat(m, n), Op::MatMul { a, b })
    }

    /// `x·W + b`: `x: (batch,in)`, `w: (in,out)`, `b: (1,out)` broadcast over
    /// the batch. Built from `matmul` + `add`, so its gradient is exactly the
    /// composition of theirs (nothing new to prove beyond those two ops).
    pub fn linear(&mut self, x: NodeId, w: NodeId, b: NodeId) -> NodeId {
        let xw = self.matmul(x, w);
        self.add(xw, b)
    }

    /// Elementwise `tanh`, the nonlinearity for this increment.
    pub fn tanh(&mut self, x: NodeId) -> NodeId {
        let shape = self.nodes[x].shape;
        let value: Vec<f64> = self.nodes[x].value.iter().map(|v| v.tanh()).collect();
        self.push(value, shape, Op::Tanh { x })
    }

    /// Born-rule byte head, log domain: `born_logits(a)_i = ln(a_i² + ε)`
    /// (elementwise, ε = [`BORN_EPS`]). Feeding this to `log_softmax` yields the
    /// Born distribution `p_i = (a_i² + ε) / Σ_j (a_j² + ε)` — the tape's
    /// differentiable stand-in for `born.rs`'s squared-magnitude readout (T0.4),
    /// so the pattern model's BPB flows through the proven softmax/CE backward.
    pub fn born_logits(&mut self, x: NodeId) -> NodeId {
        let shape = self.nodes[x].shape;
        let value: Vec<f64> = self.nodes[x]
            .value
            .iter()
            .map(|a| (a * a + BORN_EPS).ln())
            .collect();
        self.push(value, shape, Op::BornLogits { x })
    }

    /// Row-wise, numerically-stable log-softmax: subtract each row's max
    /// before exponentiating (VERIFICATION.md §3's stability requirement).
    pub fn log_softmax(&mut self, x: NodeId) -> NodeId {
        let shape = self.nodes[x].shape;
        let xv = &self.nodes[x].value;
        let mut value = vec![0.0; xv.len()];
        for r in 0..shape.rows {
            let row = &xv[r * shape.cols..(r + 1) * shape.cols];
            let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = row.iter().map(|v| (v - max).exp()).sum();
            let log_sum_exp = sum_exp.ln() + max;
            for c in 0..shape.cols {
                value[r * shape.cols + c] = xv[r * shape.cols + c] - log_sum_exp;
            }
        }
        self.push(value, shape, Op::LogSoftmax { x })
    }

    /// Mean negative log-likelihood of the target class, one target per row
    /// of `logp` (which must already be a `log_softmax` output). This is
    /// what BPB is built on (implementation/METRICS-AND-GATES.md §1).
    pub fn cross_entropy(&mut self, logp: NodeId, targets: &[usize]) -> NodeId {
        let shape = self.nodes[logp].shape;
        assert_eq!(
            targets.len(),
            shape.rows,
            "Tape::cross_entropy: need exactly one target per row"
        );
        let lv = &self.nodes[logp].value;
        let mut loss = 0.0;
        for (r, &t) in targets.iter().enumerate() {
            assert!(
                t < shape.cols,
                "Tape::cross_entropy: target class {t} out of range (0..{})",
                shape.cols
            );
            loss += -lv[r * shape.cols + t];
        }
        loss /= shape.rows as f64;
        self.push(
            vec![loss],
            Shape::mat(1, 1),
            Op::CrossEntropy {
                logp,
                targets: targets.to_vec(),
            },
        )
    }

    fn accumulate(&mut self, id: NodeId, g: &[f64]) {
        let node = &mut self.nodes[id];
        debug_assert_eq!(node.grad.len(), g.len());
        for (acc, gi) in node.grad.iter_mut().zip(g) {
            *acc += gi;
        }
    }

    /// Zero every node's grad buffer, seed `root`'s grad to 1 (`root` must be
    /// a scalar, shape `(1,1)` — the loss), then walk the arena in reverse
    /// from `root` to `0`, dispatching each node's local backward. Because
    /// the tape is built in topological order, a plain reverse index walk
    /// visits every node after all of its consumers — no separate sort.
    pub fn backward(&mut self, root: NodeId) {
        for n in &mut self.nodes {
            for g in n.grad.iter_mut() {
                *g = 0.0;
            }
        }
        assert_eq!(
            self.nodes[root].value.len(),
            1,
            "Tape::backward: root node must be scalar (shape (1,1)), got {:?}",
            self.nodes[root].shape
        );
        self.nodes[root].grad[0] = 1.0;

        for i in (0..=root).rev() {
            self.backward_node(i);
        }
    }

    fn backward_node(&mut self, i: NodeId) {
        let node_grad = self.nodes[i].grad.clone();
        let op = self.nodes[i].op.clone();

        match op {
            Op::Leaf => {}
            Op::SumSquares { x } => {
                let x_val = self.nodes[x].value.clone();
                let g0 = node_grad[0];
                let gx: Vec<f64> = x_val.iter().map(|v| 2.0 * g0 * v).collect();
                self.accumulate(x, &gx);
            }
            Op::Add { a, b, broadcast_b } => {
                self.accumulate(a, &node_grad);
                if broadcast_b {
                    let a_shape = self.nodes[a].shape;
                    let cols = a_shape.cols;
                    let mut bg = vec![0.0; cols];
                    for r in 0..a_shape.rows {
                        for c in 0..cols {
                            bg[c] += node_grad[r * cols + c];
                        }
                    }
                    self.accumulate(b, &bg);
                } else {
                    self.accumulate(b, &node_grad);
                }
            }
            Op::MatMul { a, b } => {
                let a_shape = self.nodes[a].shape;
                let b_shape = self.nodes[b].shape;
                let (m, k, n) = (a_shape.rows, a_shape.cols, b_shape.cols);
                let a_val = self.nodes[a].value.clone();
                let b_val = self.nodes[b].value.clone();

                // Y = A(m,k) * B(k,n); dA = dY * B^T (m,k); dB = A^T * dY (k,n).
                let mut ga = vec![0.0; m * k];
                for i2 in 0..m {
                    for p in 0..k {
                        let mut s = 0.0;
                        for j in 0..n {
                            s += node_grad[i2 * n + j] * b_val[p * n + j];
                        }
                        ga[i2 * k + p] = s;
                    }
                }
                let mut gb = vec![0.0; k * n];
                for p in 0..k {
                    for j in 0..n {
                        let mut s = 0.0;
                        for i2 in 0..m {
                            s += a_val[i2 * k + p] * node_grad[i2 * n + j];
                        }
                        gb[p * n + j] = s;
                    }
                }
                self.accumulate(a, &ga);
                self.accumulate(b, &gb);
            }
            Op::Tanh { x } => {
                // node_value here IS tanh(x) (cached forward output), so
                // d/dx tanh(x) = 1 - tanh(x)^2 = 1 - node_value^2 needs no
                // access to x's own value.
                let node_value = self.nodes[i].value.clone();
                let gx: Vec<f64> = node_grad
                    .iter()
                    .zip(&node_value)
                    .map(|(g, y)| g * (1.0 - y * y))
                    .collect();
                self.accumulate(x, &gx);
            }
            Op::BornLogits { x } => {
                // y_i = ln(x_i² + ε); dy_i/dx_i = 2 x_i / (x_i² + ε). Needs x's
                // own value (unlike tanh/log_softmax, the forward output y does
                // not determine the derivative here).
                let x_val = self.nodes[x].value.clone();
                let gx: Vec<f64> = node_grad
                    .iter()
                    .zip(&x_val)
                    .map(|(g, a)| g * (2.0 * a) / (a * a + BORN_EPS))
                    .collect();
                self.accumulate(x, &gx);
            }
            Op::LogSoftmax { x } => {
                // y = log_softmax(x); softmax(x)_c = exp(y_c) (no need to
                // revisit x's own value). Standard log-softmax backward:
                // dL/dx_i = dL/dy_i - softmax(x)_i * sum_j dL/dy_j (per row).
                let node_value = self.nodes[i].value.clone();
                let shape = self.nodes[i].shape;
                let cols = shape.cols;
                let mut gx = vec![0.0; node_value.len()];
                for r in 0..shape.rows {
                    let row_grad = &node_grad[r * cols..(r + 1) * cols];
                    let row_y = &node_value[r * cols..(r + 1) * cols];
                    let sum_g: f64 = row_grad.iter().sum();
                    for c in 0..cols {
                        let softmax_c = row_y[c].exp();
                        gx[r * cols + c] = row_grad[c] - softmax_c * sum_g;
                    }
                }
                self.accumulate(x, &gx);
            }
            Op::CrossEntropy { logp, targets } => {
                // L = -mean_r logp[r, target[r]]; dL/dlogp[r,c] =
                // -(1/rows) if c == target[r] else 0, scaled by the
                // incoming node_grad[0] (the loss node is a scalar).
                let logp_shape = self.nodes[logp].shape;
                let (rows, cols) = (logp_shape.rows, logp_shape.cols);
                let mut g = vec![0.0; rows * cols];
                let scale = node_grad[0] / rows as f64;
                for (r, &t) in targets.iter().enumerate() {
                    g[r * cols + t] = -scale;
                }
                self.accumulate(logp, &g);
            }
        }
    }
}
