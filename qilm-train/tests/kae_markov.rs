//! kae_markov — the known-answer end-to-end for Phase 1 (T1.5): a trained model
//! must reach the source's ANALYTIC entropy floor. The target `H` (and the
//! order-0 floor `H0`) come from the Markov transition matrix that DEFINES the
//! fixture — computed independently of any model (qilm_data, AGENTS.md rule 1) —
//! so this replaces "loss went down" with "reached the information floor we
//! computed by hand."
//!
//! Scope note: this asserts the BYTE-SOFTMAX BASELINE reaches the floor, which
//! validates the KAE methodology, the training path, and that the floor is
//! actually reachable. Whether the BORN-HEAD PATTERN model reaches it is the G0
//! feasibility gate's job (a project-kill gate, not a unit test) — and on the
//! recorded run it did NOT: it collapsed to the order-0 marginal (bpb_ratio ≈
//! 1.50, collapse gate FAIL). That verdict lives in the generated
//! reports/PHASE-1.md, not in a hand-asserted test here.

use qilm_data::synth_markov;
use qilm_train::metrics::bpb::bpb;
use qilm_train::model_token::TokenModel;
use qilm_train::train::{init_params, sgd, to_symbols, Corpus, SgdConfig};

const CTX: usize = 1;

fn setup() -> (Corpus, Corpus, f64, f64, usize) {
    // n >= 512000 is the fixture's state-space requirement (K^(order+1) <= n/2000).
    let (bytes, truth) = synth_markov(1, 7, 520_000);
    let (syms, k) = to_symbols(&bytes);
    let train = Corpus::new(syms[..500_000].to_vec(), k);
    let held = Corpus::new(syms[500_000..].to_vec(), k);
    (
        train,
        held,
        truth.entropy_bits_per_byte,
        truth.order0_entropy,
        k,
    )
}

#[test]
fn kae_markov_baseline_reaches_floor() {
    let (train, held, h, h0, k) = setup();
    let (ebag, ebatch, etgt) = held.eval_bags(CTX);

    let tok = TokenModel::new(k, 12, CTX);
    let cfg = SgdConfig {
        steps: 1800,
        lr: 0.5,
        batch: 64,
        seed: 42,
    };
    let p = sgd(init_params(tok.num_params(), 0.1, 1), &cfg, |p, rng| {
        let b = train.sample(CTX, cfg.batch, rng);
        tok.ce_loss_and_grad(p, &b.bag_ctx, b.batch, &b.targets).1
    });
    let bpb_val = bpb(&tok.logp_batch(&p, &ebag, ebatch), k, &etgt);

    // Reaches the conditional floor: H <= BPB <= H + 0.15 (small tolerance under
    // H for held-out sampling noise) ...
    assert!(
        bpb_val >= h - 0.02 && bpb_val <= h + 0.15,
        "baseline BPB {bpb_val} not within [H, H+0.15] = [{h}, {}]",
        h + 0.15
    );
    // ... and beats the order-0 (unigram) floor by a clear margin.
    assert!(
        bpb_val <= h0 - 0.30,
        "baseline BPB {bpb_val} did not beat the order-0 floor H0-0.30 = {}",
        h0 - 0.30
    );
}

/// Anti-vacuity canary: an UNTRAINED model must NOT reach the floor — its BPB
/// stays at/above the order-0 level. Proves `kae_markov_baseline_reaches_floor`
/// is testing convergence, not passing for any model (Rule 2).
#[test]
fn kae_markov_untrained_does_not_reach_floor() {
    let (_train, held, _h, h0, k) = setup();
    let (ebag, ebatch, etgt) = held.eval_bags(CTX);

    let tok = TokenModel::new(k, 12, CTX);
    let untrained = init_params(tok.num_params(), 0.1, 5);
    let bpb_val = bpb(&tok.logp_batch(&untrained, &ebag, ebatch), k, &etgt);

    assert!(
        bpb_val > h0 - 0.30,
        "untrained model BPB {bpb_val} must NOT reach the floor (H0-0.30 = {}) — \
         if it does, the eval is peeking or the floor test is vacuous",
        h0 - 0.30
    );
}
