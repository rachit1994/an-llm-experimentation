//! nc_bpb — Phase-1 BPB controls (T1.4). Two negative controls plus a
//! hand-computed known-answer test and its anti-vacuity canary.
//!
//!  - kat_bpb_uniform: on a uniform distribution BPB = log2(vocab) exactly
//!    (hand-computed, independent of the code), with a canary showing a
//!    confident-correct model scores below it and a confident-wrong one above.
//!  - nc_untrained_bpb: a random-init byte model scores BPB ≥ 7.5 (≈ log2 256).
//!    A low BPB from an untrained model means the eval peeks — threat T3.
//!  - nc_param_match: the token model's param count matches its hand-derived
//!    formula, and `params_within` accepts a ±5% match / rejects a large gap.

use qilm_train::metrics::bpb::{bpb, params_within};
use qilm_train::model_token::TokenModel;
use qilm_train::testkit::random_init_model;

#[test]
fn kat_bpb_uniform_is_log2_vocab() {
    // Uniform over vocab=8: every logp = ln(1/8); BPB = -log2(1/8) = 3 exactly,
    // for ANY targets. Hand-computed, independent of bpb()'s internals.
    let vocab = 8;
    let n = 5;
    let uniform_logp = (1.0f64 / vocab as f64).ln();
    let logp = vec![uniform_logp; n * vocab];
    let targets = vec![0usize, 3, 7, 1, 5];
    let got = bpb(&logp, vocab, &targets);
    assert!(
        (got - 3.0).abs() < 1e-12,
        "uniform BPB should be 3.0, got {got}"
    );
}

/// Anti-vacuity canary: BPB must MOVE the right way — a confident-correct
/// distribution scores well below log2(vocab), a confident-wrong one well
/// above. If BPB returned a constant, both would equal the uniform value.
#[test]
fn kat_bpb_can_say_no() {
    let vocab = 4;
    let uniform = (vocab as f64).log2(); // = 2.0 bits

    // Confident CORRECT: put ~all mass on the true byte (target 2).
    let mut logp_correct = vec![(1e-6f64).ln(); vocab];
    logp_correct[2] = (1.0f64 - 3e-6).ln();
    let bpb_correct = bpb(&logp_correct, vocab, &[2]);

    // Confident WRONG: ~all mass on byte 0, true byte is 2.
    let mut logp_wrong = vec![(1e-6f64).ln(); vocab];
    logp_wrong[0] = (1.0f64 - 3e-6).ln();
    let bpb_wrong = bpb(&logp_wrong, vocab, &[2]);

    assert!(
        bpb_correct < uniform - 0.5,
        "confident-correct BPB {bpb_correct} should be well below uniform {uniform}"
    );
    assert!(
        bpb_wrong > uniform + 0.5,
        "confident-wrong BPB {bpb_wrong} should be well above uniform {uniform}"
    );
}

#[test]
fn nc_untrained_bpb() {
    let model = TokenModel::new(256, 8, 4);
    // Untrained: random parameters, no training.
    let params = random_init_model(0xC0FFEE, model.num_params());

    // Held-out byte stream, independent of the model (deterministic spread over
    // all 256 byte values). An untrained model cannot predict it.
    let bytes: Vec<u8> = (0..512u32).map(|i| ((i * 167 + 13) % 256) as u8).collect();

    let score = model.bpb_on(&params, &bytes);
    assert!(
        score >= 7.5,
        "untrained model BPB should be ≥ 7.5 (≈ log2 256 = 8); got {score}. \
         A lower value means the eval is peeking at the scored bytes (threat T3)."
    );
}

#[test]
fn nc_param_match() {
    let model = TokenModel::new(256, 8, 4);
    // Hand-derived: vocab*d + d*vocab + vocab = 256*8 + 8*256 + 256 = 4352.
    let expected = 256 * 8 + 8 * 256 + 256;
    assert_eq!(
        model.num_params(),
        expected,
        "token model param count must match the hand-derived formula"
    );

    // params_within: a 3% gap is within 5%; a 20% gap is not.
    assert!(
        params_within(10_000, 10_300, 0.05),
        "3% gap should be within 5%"
    );
    assert!(
        !params_within(10_000, 12_000, 0.05),
        "20% gap should NOT be within 5%"
    );
    // TODO(phase1-integration): once the Born-head pattern arm exists, assert
    // params_within(pattern.num_params(), token.num_params(), 0.05) directly.
}
