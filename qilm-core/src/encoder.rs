//! encoder — the context→pattern encoder (T1.1), expressed entirely in the
//! autodiff tape so its parameters train by backprop.
//!
//! A byte context is fed in already reduced to a `bag` matrix `m` of shape
//! `(batch × vocab)`, where `m[i][b]` is the fraction of position `i`'s context
//! window occupied by byte `b` (counts / context_len). This makes the embedding
//! lookup + mean-pool a single matmul `c = m · E` (E the `vocab × d_emb`
//! embedding table) — no gather op is needed in the tape, and the gradient
//! flows to `E` correctly because the bag weights are constants. The encoder
//! then projects and squashes: `z = tanh(c · W_enc + b_enc)`.

use crate::autodiff::{NodeId, Tape};

/// Build the encoder subgraph and return the node for `z` (shape
/// `batch × d_z`). Caller owns all leaves:
/// - `bag`  : `(batch × vocab)` context bag weights (a constant input leaf),
/// - `e`    : `(vocab × d_emb)` embedding table (parameter),
/// - `w_enc`: `(d_emb × d_z)` projection (parameter),
/// - `b_enc`: `(1 × d_z)` bias (parameter, row-broadcast).
pub fn encode(tape: &mut Tape, bag: NodeId, e: NodeId, w_enc: NodeId, b_enc: NodeId) -> NodeId {
    let c = tape.matmul(bag, e); // (batch × d_emb) — embed + mean-pool
    let pre = tape.linear(c, w_enc, b_enc); // (batch × d_z)
    tape.tanh(pre)
}
