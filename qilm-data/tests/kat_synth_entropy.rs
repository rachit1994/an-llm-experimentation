//! kat_synth_entropy — analytic entropy from transition matrix must match independent empirical estimator.
//!
//! This test proves the fixture's own target (entropy_bits_per_byte computed from the Markov
//! transition matrix) is correct by verifying it against a SECOND, INDEPENDENT estimator:
//! empirical plug-in entropy computed over 2M+ sampled bytes.
//!
//! PASS ⟺ |H_analytic - H_empirical| ≤ 0.02 bits/byte

use qilm_data::synth;

#[test]
fn kat_synth_entropy() {
    // Generate 2.5M bytes from a known order-2 Markov source.
    const N_BYTES: usize = 2_500_000;
    const ORDER: usize = 2;
    const SEED: u64 = 42;
    const TOLERANCE_BITS_PER_BYTE: f64 = 0.02;

    eprintln!("Generating {} bytes with order {} seed {}", N_BYTES, ORDER, SEED);
    let (bytes, truth) = synth::synth_markov(ORDER, SEED, N_BYTES);
    eprintln!("Generated {} bytes", bytes.len());

    // Analytic entropy from the transition matrix (computed during generation).
    let h_analytic = truth.entropy_bits_per_byte;

    // Empirical entropy rate: the average conditional entropy -log2 P(byte | context).
    // This is computed by building the transition matrix from the sample and computing
    // the empirical conditional probabilities.
    use std::collections::HashMap;
    let mut transition_counts: HashMap<Vec<u8>, HashMap<u8, u64>> = HashMap::new();

    // Build transition matrix from the sequence.
    for i in ORDER..N_BYTES {
        let context = bytes[i - ORDER..i].to_vec();
        let byte = bytes[i];
        *transition_counts.entry(context).or_insert_with(HashMap::new)
            .entry(byte)
            .or_insert(0) += 1;
    }

    // Compute empirical entropy rate.
    // H_empirical = Σ_context P(context) * H(X | context)
    //            = Σ_context P(context) * Σ_x P(x|context) * (-log P(x|context))
    let mut h_empirical = 0.0;
    let n_transitions = (N_BYTES - ORDER) as f64;

    for (_context, next_bytes) in &transition_counts {
        let context_count: u64 = next_bytes.values().sum();
        if context_count == 0 {
            continue;
        }

        let p_context = context_count as f64 / n_transitions;
        let mut h_given_context = 0.0;

        for (&_byte, &count) in next_bytes {
            if count > 0 {
                let p_byte_given_context = count as f64 / context_count as f64;
                h_given_context -= p_byte_given_context * p_byte_given_context.log2();
            }
        }

        h_empirical += p_context * h_given_context;
    }

    let diff = (h_analytic - h_empirical).abs();

    eprintln!(
        "H_analytic = {:.6} bits/byte",
        h_analytic
    );
    eprintln!(
        "H_empirical = {:.6} bits/byte",
        h_empirical
    );
    eprintln!(
        "|H_analytic - H_empirical| = {:.6} bits/byte",
        diff
    );

    assert!(
        diff <= TOLERANCE_BITS_PER_BYTE,
        "Entropy mismatch exceeds tolerance: analytic={:.6}, empirical={:.6}, diff={:.6}, threshold={:.6}",
        h_analytic,
        h_empirical,
        diff,
        TOLERANCE_BITS_PER_BYTE
    );
}

#[test]
fn test_synth_determinism() {
    // Same seed must produce byte-identical output.
    const N_BYTES: usize = 100_000;
    const ORDER: usize = 2;
    const SEED: u64 = 7;

    let (bytes1, truth1) = synth::synth_markov(ORDER, SEED, N_BYTES);
    let (bytes2, truth2) = synth::synth_markov(ORDER, SEED, N_BYTES);

    // Bytes must be identical.
    assert_eq!(
        bytes1, bytes2,
        "Same seed should produce byte-identical output"
    );

    // Truth values should be very close (within floating-point precision).
    let ent_diff = (truth1.entropy_bits_per_byte - truth2.entropy_bits_per_byte).abs();
    let order0_diff = (truth1.order0_entropy - truth2.order0_entropy).abs();

    assert!(
        ent_diff < 1e-10,
        "Entropy must be identical for same seed, diff={}",
        ent_diff
    );
    assert!(
        order0_diff < 1e-10,
        "Order-0 entropy must be identical for same seed, diff={}",
        order0_diff
    );
}
