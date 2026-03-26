//! Mock on-chain coordination layer (simulates Ethereum L2 smart contract).
//!
//! The contract:
//!   1. Verifies the IVC-aggregated proof
//!   2. Updates the Merkle root of model parameters
//!   3. Records per-round and cumulative epsilon consumption
//!   4. Permanently halts training when budget ceiling is reached
//!   5. Maintains a tamper-proof audit log of all rounds

use crate::privacy_budget::PrivacyBudget;
use crate::types::{AggregatedProof, OnChainState, RoundRecord};
use sha2::{Digest, Sha256};

/// Simulated Ethereum L2 smart contract for privacy budget enforcement.
pub struct OnChainContract {
    pub state: OnChainState,
    pub budget: PrivacyBudget,
    /// Registered seed commitments per participant.
    pub seed_commitments: Vec<(String, String)>,
}

impl OnChainContract {
    /// Deploy the contract with a privacy budget ceiling.
    pub fn deploy(epsilon_budget: f64) -> Self {
        Self {
            state: OnChainState {
                current_round: 0,
                model_merkle_root: "0x0000000000000000000000000000000000000000000000000000000000000000".to_string(),
                epsilon_total: 0.0,
                epsilon_budget,
                training_halted: false,
                audit_log: Vec::new(),
            },
            budget: PrivacyBudget::new(epsilon_budget),
            seed_commitments: Vec::new(),
        }
    }

    /// Register a participant's seed commitment before training begins.
    pub fn register_seed_commitment(&mut self, participant_id: &str, commitment: &str) {
        self.seed_commitments
            .push((participant_id.to_string(), commitment.to_string()));
    }

    /// Submit a training round: verify proof, update state, enforce budget.
    ///
    /// Returns Ok(round_record) if accepted, Err(reason) if rejected.
    pub fn submit_round(
        &mut self,
        proof: &AggregatedProof,
        model_weights_hash: &str,
    ) -> Result<RoundRecord, String> {
        // Check if training has been halted
        if self.state.training_halted {
            return Err("Training permanently halted: privacy budget exhausted.".into());
        }

        // Verify the aggregated proof
        if proof.valid != Some(true) {
            return Err("Aggregated proof verification failed. Round rejected.".into());
        }

        // Enforce privacy budget
        let new_total = self.budget.consume(proof.round_epsilon)?;

        // Update model Merkle root
        let merkle_root = compute_merkle_root(model_weights_hash, &self.state.model_merkle_root);

        // Create audit record
        self.state.current_round += 1;
        self.state.model_merkle_root = merkle_root.clone();
        self.state.epsilon_total = new_total;

        if self.budget.exhausted {
            self.state.training_halted = true;
        }

        let record = RoundRecord {
            round: self.state.current_round,
            aggregated_proof_hash: proof.aggregated_hash.clone(),
            model_merkle_root: merkle_root,
            epsilon_round: proof.round_epsilon,
            epsilon_cumulative: new_total,
            timestamp: format!("2026-04-{:02}T10:00:00Z", self.state.current_round),
            verified: true,
        };

        self.state.audit_log.push(record.clone());
        Ok(record)
    }

    /// Query current on-chain state (public view function).
    pub fn get_state(&self) -> &OnChainState {
        &self.state
    }

    /// Emit RoundCompleted event (simulated).
    pub fn emit_round_completed(&self, record: &RoundRecord) -> String {
        format!(
            "Event RoundCompleted {{ round: {}, merkle_root: {:.16}..., \
             epsilon_round: {:.6}, epsilon_total: {:.6}, budget_remaining: {:.6} }}",
            record.round,
            record.model_merkle_root,
            record.epsilon_round,
            record.epsilon_cumulative,
            self.budget.remaining()
        )
    }
}

/// Compute a simple Merkle root update: H(new_hash || old_root).
fn compute_merkle_root(new_hash: &str, old_root: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(new_hash.as_bytes());
    hasher.update(old_root.as_bytes());
    format!("0x{}", hex::encode(hasher.finalize()))
}

/// Compute a hash of model weights for Merkle root updates.
pub fn hash_model_weights(weights: &[Vec<f64>]) -> String {
    let mut hasher = Sha256::new();
    for row in weights {
        for w in row {
            hasher.update(w.to_le_bytes());
        }
    }
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AggregatedProof;

    #[test]
    fn test_contract_deployment_and_round_submission() {
        let mut contract = OnChainContract::deploy(5.0);
        let proof = AggregatedProof {
            round: 1,
            aggregated_hash: "abc123".to_string(),
            participant_proofs: vec!["p1".into(), "p2".into()],
            round_epsilon: 1.0,
            valid: Some(true),
        };
        let result = contract.submit_round(&proof, "model_hash_r1");
        assert!(result.is_ok());
        assert_eq!(contract.state.current_round, 1);
        assert!((contract.state.epsilon_total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_budget_enforcement_halts_training() {
        let mut contract = OnChainContract::deploy(2.0);
        for i in 1..=3 {
            let proof = AggregatedProof {
                round: i,
                aggregated_hash: format!("hash_{}", i),
                participant_proofs: vec![],
                round_epsilon: 1.0,
                valid: Some(true),
            };
            let _ = contract.submit_round(&proof, &format!("model_{}", i));
        }
        assert!(contract.state.training_halted);
    }
}
