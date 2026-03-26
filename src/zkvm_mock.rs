//! Mock zkVM proof generation and verification.
//!
//! In a production system, each participant executes every DP-SGD step
//! inside a zkVM (RISC Zero or SP1), which produces a succinct proof
//! attesting that the declared (epsilon, delta) parameters were correctly
//! enforced. Here we simulate this with SHA-256 commitment schemes.

use crate::types::ZkProof;
use sha2::{Digest, Sha256};

/// Simulate zkVM guest program execution and proof generation.
///
/// In the real system, the zkVM guest program implements:
///   1. Per-sample gradient clipping to bound C
///   2. Gaussian noise generation from committed seed
///   3. Noise addition to clipped gradient
///   4. Weight update
///
/// The proof attests all four steps were correctly executed with the
/// declared privacy parameters, without revealing data, weights, or seed.
pub fn generate_proof(
    participant_id: &str,
    round: usize,
    clipped_gradient: &[f64],
    noise_seed: u64,
    epsilon: f64,
    delta: f64,
) -> ZkProof {
    // Commitment: H(clipped_gradient || noise_seed || epsilon || delta)
    let mut hasher = Sha256::new();
    for g in clipped_gradient {
        hasher.update(g.to_le_bytes());
    }
    hasher.update(noise_seed.to_le_bytes());
    hasher.update(epsilon.to_le_bytes());
    hasher.update(delta.to_le_bytes());
    let commitment = hex::encode(hasher.finalize());

    // Proof: H(commitment || participant_id || round)
    // In real zkVM, this would be a STARK/SNARK proof of ~3-4 KB
    let mut proof_hasher = Sha256::new();
    proof_hasher.update(commitment.as_bytes());
    proof_hasher.update(participant_id.as_bytes());
    proof_hasher.update(round.to_le_bytes());
    let proof_hash = hex::encode(proof_hasher.finalize());

    ZkProof {
        participant_id: participant_id.to_string(),
        round,
        commitment,
        epsilon_claimed: epsilon,
        delta_claimed: delta,
        proof_hash,
        valid: None,
    }
}

/// Verify a zkVM proof.
///
/// In the real system, this runs the STARK/SNARK verifier (2-5ms, <300K gas).
/// Here we simulate verification by recomputing the proof hash and checking
/// that the claimed parameters are within acceptable bounds.
pub fn verify_proof(proof: &mut ZkProof, max_epsilon: f64, max_delta: f64) -> bool {
    // Check claimed parameters are within declared bounds
    if proof.epsilon_claimed > max_epsilon || proof.delta_claimed > max_delta {
        proof.valid = Some(false);
        return false;
    }

    // Recompute proof hash from commitment (simulates SNARK verification)
    let mut proof_hasher = Sha256::new();
    proof_hasher.update(proof.commitment.as_bytes());
    proof_hasher.update(proof.participant_id.as_bytes());
    proof_hasher.update(proof.round.to_le_bytes());
    let expected_hash = hex::encode(proof_hasher.finalize());

    let valid = expected_hash == proof.proof_hash;
    proof.valid = Some(valid);
    valid
}

/// Compute Poseidon hash commitment for a noise seed.
/// (Mocked with SHA-256; real system uses Poseidon for zkVM efficiency.)
pub fn commit_seed(seed: u64, participant_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"SEED_COMMITMENT_V1");
    hasher.update(seed.to_le_bytes());
    hasher.update(participant_id.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_generation_and_verification() {
        let gradient = vec![0.1, -0.2, 0.3];
        let mut proof = generate_proof("buildings_dept", 1, &gradient, 12345, 1.0, 1e-5);
        assert!(verify_proof(&mut proof, 2.0, 1e-4));
        assert_eq!(proof.valid, Some(true));
    }

    #[test]
    fn test_proof_rejected_if_epsilon_exceeds_bound() {
        let gradient = vec![0.1, -0.2, 0.3];
        let mut proof = generate_proof("buildings_dept", 1, &gradient, 12345, 5.0, 1e-5);
        assert!(!verify_proof(&mut proof, 2.0, 1e-4));
        assert_eq!(proof.valid, Some(false));
    }

    #[test]
    fn test_seed_commitment_deterministic() {
        let c1 = commit_seed(42, "party_a");
        let c2 = commit_seed(42, "party_a");
        assert_eq!(c1, c2);
    }
}
