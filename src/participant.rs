//! VFL Participant: holds a local subnetwork over its own feature partition.
//!
//! Each participant trains a local neural network (simplified as a linear
//! layer in this prototype) on its own feature columns and produces
//! cut-layer embeddings. DP-SGD is applied to all local training steps.

use crate::dp_sgd::{self, RenyiAccountant};
use crate::types::{BuildingRecord, DpSgdConfig, Embedding, Gradient, ZkProof};
use crate::zkvm_mock;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// A participant in the vertical federated learning system.
pub struct Participant {
    /// Unique identifier (e.g., "buildings_dept", "property_mgr").
    pub id: String,
    /// Local subnetwork weights: matrix of shape [input_dim x embedding_dim].
    pub weights: Vec<Vec<f64>>,
    /// Bias vector of shape [embedding_dim].
    pub bias: Vec<f64>,
    /// DP-SGD configuration.
    pub dp_config: DpSgdConfig,
    /// Committed noise seed for this participant.
    pub noise_seed_base: u64,
    /// Seed commitment hash (submitted on-chain before training).
    pub seed_commitment: String,
    /// Rényi DP accountant.
    pub accountant: RenyiAccountant,
    /// Input dimension (number of features this participant holds).
    pub input_dim: usize,
    /// Embedding dimension (cut-layer width).
    pub embedding_dim: usize,
}

impl Participant {
    pub fn new(
        id: &str,
        input_dim: usize,
        embedding_dim: usize,
        dp_config: DpSgdConfig,
        noise_seed_base: u64,
    ) -> Self {
        // Xavier initialization
        let mut rng = StdRng::seed_from_u64(noise_seed_base + 1000);
        let std_dev = (2.0 / (input_dim + embedding_dim) as f64).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();

        let weights: Vec<Vec<f64>> = (0..input_dim)
            .map(|_| (0..embedding_dim).map(|_| normal.sample(&mut rng)).collect())
            .collect();
        let bias = vec![0.0; embedding_dim];

        let seed_commitment = zkvm_mock::commit_seed(noise_seed_base, id);

        Self {
            id: id.to_string(),
            weights,
            bias,
            accountant: RenyiAccountant::new(8.0, dp_config.delta),
            dp_config,
            noise_seed_base,
            seed_commitment,
            input_dim,
            embedding_dim,
        }
    }

    /// Forward pass: compute cut-layer embedding for a building.
    /// embedding = ReLU(W^T * features + b)
    pub fn forward(&self, record: &BuildingRecord) -> Embedding {
        let mut values = vec![0.0; self.embedding_dim];
        for j in 0..self.embedding_dim {
            let mut sum = self.bias[j];
            for i in 0..self.input_dim.min(record.features.len()) {
                sum += record.features[i] * self.weights[i][j];
            }
            // ReLU activation
            values[j] = sum.max(0.0);
        }
        Embedding {
            participant_id: self.id.clone(),
            building_id: record.building_id.clone(),
            values,
        }
    }

    /// Backward pass with DP-SGD: receive gradient from coordinator,
    /// clip it, add noise, update local weights, and generate zkVM proof.
    pub fn backward_with_proof(
        &mut self,
        upstream_gradient: &[f64],
        input_features: &[f64],
        round: usize,
    ) -> ZkProof {
        // Compute local gradient: dL/dW = features^T * upstream_gradient
        let mut grad_values = Vec::new();
        for i in 0..self.input_dim {
            for j in 0..self.embedding_dim.min(upstream_gradient.len()) {
                grad_values.push(input_features[i.min(input_features.len() - 1)] * upstream_gradient[j]);
            }
        }

        let mut gradient = Gradient {
            participant_id: self.id.clone(),
            values: grad_values,
        };

        // Flatten weights for DP-SGD step
        let mut flat_weights: Vec<f64> = self.weights.iter().flatten().copied().collect();
        let noise_seed = self.noise_seed_base + round as u64;

        // Execute DP-SGD step (in real system, this runs inside zkVM)
        let noised_gradient = dp_sgd::dp_sgd_step(
            &mut flat_weights,
            &mut gradient,
            &self.dp_config,
            noise_seed,
        );

        // Unflatten weights back
        for i in 0..self.input_dim {
            for j in 0..self.embedding_dim {
                let idx = i * self.embedding_dim + j;
                if idx < flat_weights.len() {
                    self.weights[i][j] = flat_weights[idx];
                }
            }
        }

        // Update privacy accountant
        self.accountant.accumulate_round(
            self.dp_config.noise_sigma,
            self.dp_config.clip_bound,
        );

        // Generate zkVM proof attesting correct DP-SGD execution
        zkvm_mock::generate_proof(
            &self.id,
            round,
            &noised_gradient,
            noise_seed,
            self.dp_config.epsilon,
            self.dp_config.delta,
        )
    }

    /// Current cumulative epsilon consumed by this participant.
    pub fn current_epsilon(&self) -> f64 {
        self.accountant.current_epsilon()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_participant_forward() {
        let config = DpSgdConfig::new(1.0, 1e-5, 1.0, 0.01);
        let p = Participant::new("test", 3, 4, config, 42);
        let record = BuildingRecord {
            building_id: "B001".to_string(),
            features: vec![0.5, 0.3, 0.8],
        };
        let emb = p.forward(&record);
        assert_eq!(emb.values.len(), 4);
        assert_eq!(emb.participant_id, "test");
    }

    #[test]
    fn test_participant_backward_produces_proof() {
        let config = DpSgdConfig::new(1.0, 1e-5, 1.0, 0.01);
        let mut p = Participant::new("test", 3, 4, config, 42);
        let upstream = vec![0.1, -0.2, 0.05, 0.3];
        let features = vec![0.5, 0.3, 0.8];
        let proof = p.backward_with_proof(&upstream, &features, 1);
        assert_eq!(proof.participant_id, "test");
        assert_eq!(proof.round, 1);
        assert!(!proof.proof_hash.is_empty());
    }
}
