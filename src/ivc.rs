//! Incrementally Verifiable Computation (IVC) proof aggregation.
//!
//! Per-participant zkVM proofs are aggregated into a single compact proof
//! representing the privacy compliance of an entire training round.
//! In production, this uses folding schemes (e.g., Nova/SuperNova);
//! here we simulate with hash-based aggregation.

use crate::types::{AggregatedProof, ZkProof};
use sha2::{Digest, Sha256};

/// Aggregate multiple per-participant proofs into a single round proof.
///
/// The aggregated proof attests that ALL participants in the round
/// correctly executed DP-SGD with their declared parameters.
/// Round epsilon is taken as the maximum across participants (worst case
/// for the vertical FL composition, since all parties contribute to the
/// same model update).
pub fn aggregate_proofs(round: usize, proofs: &[ZkProof]) -> AggregatedProof {
    // IVC aggregation: H(pi_1 || pi_2 || ... || pi_K)
    let mut hasher = Sha256::new();
    hasher.update(b"IVC_AGGREGATION_V1");
    hasher.update(round.to_le_bytes());

    let mut participant_proof_hashes = Vec::new();
    let mut max_epsilon = 0.0_f64;

    for proof in proofs {
        hasher.update(proof.proof_hash.as_bytes());
        participant_proof_hashes.push(proof.proof_hash.clone());
        max_epsilon = max_epsilon.max(proof.epsilon_claimed);
    }

    let aggregated_hash = hex::encode(hasher.finalize());

    AggregatedProof {
        round,
        aggregated_hash,
        participant_proofs: participant_proof_hashes,
        round_epsilon: max_epsilon,
        valid: None,
    }
}

/// Verify an aggregated proof by checking the hash chain.
///
/// In production, this verifies the IVC/folding proof in O(1) time.
/// Here we recompute the aggregation hash.
pub fn verify_aggregated_proof(
    proof: &mut AggregatedProof,
    original_proofs: &[ZkProof],
) -> bool {
    // All individual proofs must have been verified
    if !original_proofs
        .iter()
        .all(|p| p.valid == Some(true))
    {
        proof.valid = Some(false);
        return false;
    }

    // Recompute aggregation hash
    let mut hasher = Sha256::new();
    hasher.update(b"IVC_AGGREGATION_V1");
    hasher.update(proof.round.to_le_bytes());
    for p in original_proofs {
        hasher.update(p.proof_hash.as_bytes());
    }
    let expected = hex::encode(hasher.finalize());

    let valid = expected == proof.aggregated_hash;
    proof.valid = Some(valid);
    valid
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm_mock;

    #[test]
    fn test_aggregation_and_verification() {
        let g1 = vec![0.1, 0.2];
        let g2 = vec![-0.1, 0.3];
        let mut p1 = zkvm_mock::generate_proof("party_a", 1, &g1, 100, 1.0, 1e-5);
        let mut p2 = zkvm_mock::generate_proof("party_b", 1, &g2, 200, 0.8, 1e-5);
        zkvm_mock::verify_proof(&mut p1, 2.0, 1e-4);
        zkvm_mock::verify_proof(&mut p2, 2.0, 1e-4);

        let mut agg = aggregate_proofs(1, &[p1.clone(), p2.clone()]);
        assert!(verify_aggregated_proof(&mut agg, &[p1, p2]));
        assert!((agg.round_epsilon - 1.0).abs() < 1e-9);
    }
}
