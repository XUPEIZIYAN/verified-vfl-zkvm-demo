//! Coordinator: runs the shared top model g(e_1 || ... || e_K).
//!
//! Receives cut-layer embeddings from all participants, concatenates them,
//! runs the top model forward pass, computes loss, and sends gradients
//! back to each participant for their local backward pass.

use crate::types::{CostEstimate, Embedding, MaintenancePrediction};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// The coordinator's top model in the split-network architecture.
pub struct Coordinator {
    /// Top model weights: [total_embedding_dim x output_dim].
    pub weights: Vec<Vec<f64>>,
    /// Bias vector.
    pub bias: Vec<f64>,
    /// Total embedding dimension (sum of all participants' embedding dims).
    pub input_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Learning rate.
    pub learning_rate: f64,
}

impl Coordinator {
    pub fn new(input_dim: usize, output_dim: usize, learning_rate: f64) -> Self {
        let mut rng = StdRng::seed_from_u64(999);
        let std_dev = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();

        let weights: Vec<Vec<f64>> = (0..input_dim)
            .map(|_| (0..output_dim).map(|_| normal.sample(&mut rng)).collect())
            .collect();
        let bias = vec![0.0; output_dim];

        Self {
            weights,
            bias,
            input_dim,
            output_dim,
            learning_rate,
        }
    }

    /// Concatenate embeddings from all participants into a single vector.
    pub fn concatenate_embeddings(&self, embeddings: &[Embedding]) -> Vec<f64> {
        embeddings.iter().flat_map(|e| e.values.clone()).collect()
    }

    /// Forward pass through the top model.
    /// output = sigmoid(W^T * concatenated_embeddings + b)
    pub fn forward(&self, concatenated: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.output_dim];
        for j in 0..self.output_dim {
            let mut sum = self.bias[j];
            for i in 0..self.input_dim.min(concatenated.len()) {
                sum += concatenated[i] * self.weights[i][j];
            }
            // Sigmoid activation
            output[j] = 1.0 / (1.0 + (-sum).exp());
        }
        output
    }

    /// Compute MSE loss against target.
    pub fn compute_loss(&self, prediction: &[f64], target: &[f64]) -> f64 {
        prediction
            .iter()
            .zip(target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / prediction.len() as f64
    }

    /// Backward pass: compute gradient of loss w.r.t. concatenated embeddings.
    /// Returns the full gradient vector, which is then split and sent to
    /// each participant for their local backward pass.
    pub fn backward(
        &mut self,
        concatenated: &[f64],
        prediction: &[f64],
        target: &[f64],
    ) -> Vec<f64> {
        let n = prediction.len() as f64;

        // dL/d_output = 2/n * (pred - target) * sigmoid'(x)
        let d_output: Vec<f64> = prediction
            .iter()
            .zip(target.iter())
            .map(|(p, t)| 2.0 / n * (p - t) * p * (1.0 - p))
            .collect();

        // dL/d_concatenated = W * d_output
        let mut d_concat = vec![0.0; self.input_dim];
        for i in 0..self.input_dim {
            for j in 0..self.output_dim {
                d_concat[i] += self.weights[i][j] * d_output[j];
            }
        }

        // Update top model weights: W -= lr * concatenated^T * d_output
        for i in 0..self.input_dim.min(concatenated.len()) {
            for j in 0..self.output_dim {
                self.weights[i][j] -= self.learning_rate * concatenated[i] * d_output[j];
            }
        }
        for j in 0..self.output_dim {
            self.bias[j] -= self.learning_rate * d_output[j];
        }

        d_concat
    }

    /// Split the concatenated gradient back to per-participant segments.
    pub fn split_gradient(
        &self,
        full_gradient: &[f64],
        embedding_dims: &[usize],
    ) -> Vec<Vec<f64>> {
        let mut result = Vec::new();
        let mut offset = 0;
        for &dim in embedding_dims {
            let end = (offset + dim).min(full_gradient.len());
            result.push(full_gradient[offset..end].to_vec());
            offset = end;
        }
        result
    }

    /// Interpret output as maintenance prediction.
    pub fn predict_maintenance(
        &self,
        output: &[f64],
        building_id: &str,
        horizon: u32,
    ) -> MaintenancePrediction {
        let urgency = output.first().copied().unwrap_or(0.5);
        let mtype = if urgency > 0.7 {
            "Structural"
        } else if urgency > 0.4 {
            "Facade/Exterior"
        } else {
            "Routine"
        };
        MaintenancePrediction {
            building_id: building_id.to_string(),
            urgency_score: urgency,
            maintenance_type: mtype.to_string(),
            horizon_years: horizon,
        }
    }

    /// Interpret output as cost estimate.
    pub fn predict_cost(
        &self,
        output: &[f64],
        building_id: &str,
        repair_type: &str,
    ) -> CostEstimate {
        // Scale sigmoid output to realistic HKD range
        let base = output.first().copied().unwrap_or(0.5);
        let cost = base * 2_000_000.0 + 50_000.0; // 50K - 2.05M HKD
        let spread = output.get(1).copied().unwrap_or(0.2) * 0.3;
        CostEstimate {
            building_id: building_id.to_string(),
            cost_estimate: cost,
            cost_lower: cost * (1.0 - spread),
            cost_upper: cost * (1.0 + spread),
            repair_type: repair_type.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_forward() {
        let coord = Coordinator::new(8, 2, 0.01);
        let input = vec![0.5; 8];
        let output = coord.forward(&input);
        assert_eq!(output.len(), 2);
        for v in &output {
            assert!(*v >= 0.0 && *v <= 1.0, "Sigmoid output must be in [0,1]");
        }
    }
}
