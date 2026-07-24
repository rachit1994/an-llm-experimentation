//! g0 — the Phase-1 feasibility run (T1.6). Trains the byte-softmax baseline and
//! the Born-head pattern model (the full jepa_vicreg objective) on the synthetic
//! order-1 Markov corpus over `>=5` seeds, at MATCHED parameters, and records the
//! provenance-stamped metrics the G0 / collapse gates read:
//!   bpb_ratio   = BPB_pattern / BPB_baseline   (g0 gate, <= 1.10)
//!   erank_ratio, meanstd_ratio                 (collapse gate, both >= 0.50)
//!   gradcheck_max_rel_err                      (infra gate, < 1e-4)
//! The PASS/FAIL and the worth-pursuing verdict are then written by
//! `report --phase 1` from this artifact — never by hand (AGENTS.md rule 5).
//!
//! This binary only MEASURES and records; it does not decide. Run it, then
//! `cargo run -p qilm-train --bin report -- --phase 1`.

use qilm_data::synth_markov;
use qilm_oracle::gradcheck::gradcheck;
use qilm_train::loss::{total_loss, total_loss_and_grad, Batch, LossConfig};
use qilm_train::metrics::bpb::bpb;
use qilm_train::metrics::collapse::collapse_ratios;
use qilm_train::model_pattern::PatternModel;
use qilm_train::model_token::TokenModel;
use qilm_train::provenance::{discover_workspace_root, sha256_hex, write_metrics, MetricsInput};
use qilm_train::train::{adam, init_params, to_symbols, Corpus, SgdConfig};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde_json::json;
use std::time::Instant;

const N_SEEDS: u64 = 5;
const CTX: usize = 1;

/// Isotropic, unit-variance reference `Z*` (`n × d`): the non-collapsed ideal
/// (erank ≈ d, per-dim std ≈ 1). Independent of the trained model (AGENTS.md
/// rule 1). Built from a fixed seed via a 12-uniform CLT approximation of N(0,1).
fn reference_z(n: usize, d: usize) -> Vec<f64> {
    let mut r = ChaCha20Rng::seed_from_u64(20240707);
    (0..n * d)
        .map(|_| {
            let s: f64 = (0..12).map(|_| r.random::<f64>()).sum();
            s - 6.0
        })
        .collect()
}

fn train_token(k: usize, train: &Corpus, seed: u64) -> (TokenModel, Vec<f64>) {
    let tok = TokenModel::new(k, 15, CTX);
    let cfg = SgdConfig {
        steps: 6000,
        lr: 0.01,
        batch: 128,
        seed,
    };
    let p = adam(
        init_params(tok.num_params(), 0.1, seed + 1000),
        &cfg,
        |p, rng| {
            let b = train.sample(CTX, cfg.batch, rng);
            tok.ce_loss_and_grad(p, &b.bag_ctx, b.batch, &b.targets).1
        },
    );
    (tok, p)
}

fn train_pattern(
    k: usize,
    train: &Corpus,
    cfg_loss: &LossConfig,
    seed: u64,
) -> (PatternModel, Vec<f64>) {
    let pat = PatternModel::new(k, 4, 12, CTX);
    let cfg = SgdConfig {
        steps: 6000,
        lr: 0.01,
        batch: 128,
        seed,
    };
    let lc = *cfg_loss;
    let p = adam(
        init_params(pat.num_params(), 0.3, seed + 2000),
        &cfg,
        |p, rng| {
            let b = train.sample(CTX, cfg.batch, rng);
            let bt = Batch {
                batch: b.batch,
                bag_ctx: &b.bag_ctx,
                bag_next: &b.bag_next,
                bag_view2: &b.bag_view2,
                targets: &b.targets,
            };
            total_loss_and_grad(&pat, &lc, p, &bt).1
        },
    );
    (pat, p)
}

/// Finite-difference gradcheck of the pattern full loss on a tiny batch — the
/// Phase-1 infra number. Independent numeric oracle vs the tape's backprop.
fn pattern_gradcheck(k: usize) -> f64 {
    let pat = PatternModel::new(k, 4, 12, CTX);
    let cfg = LossConfig {
        no_stopgrad: true,
        ..LossConfig::default()
    };
    let ctxs: Vec<&[u8]> = vec![&[0], &[3], &[1], &[2]];
    let nxt: Vec<&[u8]> = vec![&[1], &[4], &[3], &[0]];
    let v2: Vec<&[u8]> = vec![&[0], &[2], &[1], &[4]];
    let (bc, bn, bv) = (
        pat.bag_from_contexts(&ctxs),
        pat.bag_from_contexts(&nxt),
        pat.bag_from_contexts(&v2),
    );
    let targets = vec![3usize, 1, 4, 0];
    let params = init_params(pat.num_params(), 0.3, 1);
    let b = Batch {
        batch: 4,
        bag_ctx: &bc,
        bag_next: &bn,
        bag_view2: &bv,
        targets: &targets,
    };
    let loss = |p: &[f64]| total_loss(&pat, &cfg, p, &b);
    let grad = |p: &[f64]| total_loss_and_grad(&pat, &cfg, p, &b).1;
    gradcheck(loss, grad, &params, 1e-4)
}

fn main() {
    let start = Instant::now();

    // Fixed data (seed 7); vary the training/init seed across runs.
    let (bytes, truth) = synth_markov(1, 7, 600_000);
    let (syms, k) = to_symbols(&bytes);
    let split = 500_000;
    let train = Corpus::new(syms[..split].to_vec(), k);
    let held = Corpus::new(syms[split..].to_vec(), k);
    let (ebag, ebatch, etgt) = held.eval_bags(CTX);

    // Retry config (authorized single retry): full jepa_vicreg objective now
    // WITH the covariance/decorrelation term (lambda_cov), byte-CE-anchored so
    // BPB is the priority, trained with Adam on the un-pinched model.
    let loss_cfg = LossConfig {
        lambda_byte: 1.0,
        lambda_pattern: 0.2,
        lambda_inv: 0.2,
        lambda_var: 1.0,
        lambda_cov: 1.0,
        ..LossConfig::default()
    };

    let mut bpb_ratios = Vec::new();
    let mut erank_ratios = Vec::new();
    let mut meanstd_ratios = Vec::new();
    let mut bpb_pat_v = Vec::new();
    let mut bpb_base_v = Vec::new();

    for seed in 0..N_SEEDS {
        let (tok, tp) = train_token(k, &train, seed);
        let bpb_base = bpb(&tok.logp_batch(&tp, &ebag, ebatch), k, &etgt);

        let (pat, pp) = train_pattern(k, &train, &loss_cfg, seed);
        let bpb_pat = bpb(&pat.logp_batch(&pp, &ebag, ebatch), k, &etgt);

        // collapse ratios on a fresh sample of pattern representations.
        let mut rng = ChaCha20Rng::seed_from_u64(seed + 900);
        let sb = train.sample(CTX, 256, &mut rng);
        let z = pat.z_batch(&pp, &sb.bag_ctx, sb.batch);
        let zstar = reference_z(256, pat.d_z);
        let (er, mr) = collapse_ratios(&z, sb.batch, &zstar, 256, pat.d_z);

        eprintln!(
            "seed {seed}: BPB_base={bpb_base:.4} BPB_pat={bpb_pat:.4} ratio={:.4} erank_r={er:.4} meanstd_r={mr:.4}",
            bpb_pat / bpb_base
        );
        bpb_ratios.push(bpb_pat / bpb_base);
        erank_ratios.push(er);
        meanstd_ratios.push(mr);
        bpb_pat_v.push(bpb_pat);
        bpb_base_v.push(bpb_base);
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let bpb_ratio = mean(&bpb_ratios);
    let erank_ratio = mean(&erank_ratios);
    let meanstd_ratio = mean(&meanstd_ratios);
    let gradcheck_max_rel_err = pattern_gradcheck(k);

    let wall_clock_s = start.elapsed().as_secs_f64();
    let config_bytes = format!(
        "g0-retry:markov(order=1,seed=7,n=600000):pattern(k={k},d_emb=4,d_z=12,ctx={CTX}):token(d=15):adam:vicreg+cov:seeds={N_SEEDS}"
    );
    let config_sha256 = sha256_hex(config_bytes.as_bytes());

    let input = MetricsInput {
        config_sha256: config_sha256.clone(),
        dataset_sha256: "none".to_string(),
        seed: 0,
        backend: "cpu-sgd".to_string(),
        wall_clock_s,
        metrics: json!({
            "gradcheck_max_rel_err": gradcheck_max_rel_err,
            "bpb_ratio": bpb_ratio,
            "erank_ratio": erank_ratio,
            "meanstd_ratio": meanstd_ratio,
            "n_seeds": N_SEEDS,
            "bpb_pattern_mean": mean(&bpb_pat_v),
            "bpb_baseline_mean": mean(&bpb_base_v),
            "source_entropy_h": truth.entropy_bits_per_byte,
            "source_entropy_h0": truth.order0_entropy,
        }),
    };

    let runs_dir = discover_workspace_root().join("runs");
    match write_metrics(&runs_dir, "phase1_g0", input) {
        Ok(path) => {
            eprintln!("g0: wrote {}", path.display());
            eprintln!(
                "g0: bpb_ratio={bpb_ratio:.4} erank_ratio={erank_ratio:.4} meanstd_ratio={meanstd_ratio:.4} gradcheck={gradcheck_max_rel_err:.3e}"
            );
        }
        Err(e) => {
            eprintln!("g0: failed to write metrics: {e}");
            std::process::exit(1);
        }
    }
}
