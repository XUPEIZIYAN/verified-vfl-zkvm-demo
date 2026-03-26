//! Core types shared across all modules.

use serde::{Deserialize, Serialize};

/// A single building record with vertically partitioned features.
/// Each participant holds a different subset of columns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingRecord {
    pub building_id: String,
    pub features: Vec<f64>,
}

/// Intermediate embedding produced by a participant's local subnetwork.
/// This is the cut-layer representation transmitted to the coordinator.
/// Raw data never leaves the participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub participant_id: String,
    pub building_id: String,
    pub values: Vec<f64>,
}

/// Gradient vector for a participant's local subnetwork parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gradient {
    pub participant_id: String,
    pub values: Vec<f64>,
}

/// DP-SGD configuration for a participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpSgdConfig {
    /// Privacy parameter epsilon per round.
    pub epsilon: f64,
    /// Privacy parameter delta.
    pub delta: f64,
    /// Gradient clipping bound C.
    pub clip_bound: f64,
    /// Noise multiplier sigma = C * sqrt(2 * ln(1.25 / delta)) / epsilon.
    pub noise_sigma: f64,
    /// Learning rate.
    pub learning_rate: f64,
}

impl DpSgdConfig {
    pub fn new(epsilon: f64, delta: f64, clip_bound: f64, learning_rate: f64) -> Self {
        let noise_sigma = clip_bound * (2.0 * (1.25_f64 / delta).ln()).sqrt() / epsilon;
        Self {
            epsilon,
            delta,
            clip_bound,
            noise_sigma,
            learning_rate,
        }
    }
}

/// A zkVM proof attesting correct DP-SGD execution.
/// In this prototype, proofs are simulated using SHA-256 commitments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkProof {
    pub participant_id: String,
    pub round: usize,
    /// Hash commitment: H(gradient_clipped || noise_seed || epsilon || delta)
    pub commitment: String,
    /// Claimed epsilon consumed this round.
    pub epsilon_claimed: f64,
    /// Claimed delta.
    pub delta_claimed: f64,
    /// Proof bytes (mock: SHA-256 of the full computation trace).
    pub proof_hash: String,
    /// Whether the proof is valid (set by verifier).
    pub valid: Option<bool>,
}

/// IVC-aggregated proof for an entire training round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedProof {
    pub round: usize,
    /// Aggregated hash: H(proof_1 || proof_2 || ... || proof_K)
    pub aggregated_hash: String,
    /// Per-participant proof hashes included.
    pub participant_proofs: Vec<String>,
    /// Total epsilon consumed this round (sum across participants, worst case).
    pub round_epsilon: f64,
    pub valid: Option<bool>,
}

/// On-chain state maintained by the smart contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnChainState {
    /// Current training round.
    pub current_round: usize,
    /// Merkle root of model parameters.
    pub model_merkle_root: String,
    /// Cumulative epsilon consumed.
    pub epsilon_total: f64,
    /// Pre-agreed privacy budget ceiling.
    pub epsilon_budget: f64,
    /// Whether training has been permanently halted.
    pub training_halted: bool,
    /// Audit log of all rounds.
    pub audit_log: Vec<RoundRecord>,
}

/// A single round's record in the on-chain audit log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundRecord {
    pub round: usize,
    pub aggregated_proof_hash: String,
    pub model_merkle_root: String,
    pub epsilon_round: f64,
    pub epsilon_cumulative: f64,
    pub timestamp: String,
    pub verified: bool,
}

/// Model prediction output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenancePrediction {
    pub building_id: String,
    /// Predicted urgency score (0.0 = no urgency, 1.0 = critical).
    pub urgency_score: f64,
    /// Predicted maintenance type.
    pub maintenance_type: String,
    /// Time horizon in years.
    pub horizon_years: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    pub building_id: String,
    /// Point estimate in HKD.
    pub cost_estimate: f64,
    /// Lower bound of 90% confidence interval.
    pub cost_lower: f64,
    /// Upper bound of 90% confidence interval.
    pub cost_upper: f64,
    /// Repair type description.
    pub repair_type: String,
}
