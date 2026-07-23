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
//! CORRECTNESS: every op here is validated in `qilm-oracle/tests/` by a
//! `gradcheck_autodiff_*` test against a hand-derived, tape-independent oracle
//! (closed-form forward + closed-form analytic gradient), cross-checked
//! against `qilm_oracle::gradcheck`'s central finite difference (the fully
//! independent numeric oracle) AND against this tape's own forward/backward
//! output. A wrong local derivative in any `backward_node` arm fails that op's
//! gradcheck at `max_rel_err < 1e-4` — see VERIFICATION.md §3.
#![allow(dead_code)]

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
        }
    }
}
