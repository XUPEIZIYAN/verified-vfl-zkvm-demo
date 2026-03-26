//! DP-SGD implementation: gradient clipping, calibrated noise injection,
//! and Rényi differential privacy composition tracking.

use crate::types::{DpSgdConfig, Gradient};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Clip per-sample gradients to the declared sensitivity bound C.
/// Each gradient vector is scaled down if its L2 norm exceeds clip_bound.
pub fn clip_gradient(gradient: &mut Gradient, clip_bound: f64) -> f64 {
    let norm: f64 = gradient.values.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > clip_bound {
        let scale = clip_bound / norm;
        for v in gradient.values.iter_mut() {
            *v *= scale;
        }
    }
    norm.min(clip_bound)
}

/// Generate calibrated Gaussian noise using a committed pseudorandom seed.
/// noise_sigma = C * sqrt(2 * ln(1.25 / delta)) / epsilon
/// The seed is committed via Poseidon hash (mocked with SHA-256) before
/// training begins, preventing seed grinding attacks.
pub fn generate_noise(dimension: usize, noise_sigma: f64, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, noise_sigma).expect("Invalid noise_sigma");
    (0..dimension).map(|_| normal.sample(&mut rng)).collect()
}

/// Add noise to clipped gradient and perform weight update.
/// Returns the noised gradient (for proof generation) and updated weights.
pub fn dp_sgd_step(
    weights: &mut Vec<f64>,
    gradient: &mut Gradient,
    config: &DpSgdConfig,
    noise_seed: u64,
) -> Vec<f64> {
    // Step 1: Clip gradient
    clip_gradient(gradient, config.clip_bound);

    // Step 2: Generate calibrated Gaussian noise
    let noise = generate_noise(gradient.values.len(), config.noise_sigma, noise_seed);

    // Step 3: Add noise to clipped gradient
    let noised_gradient: Vec<f64> = gradient
        .values
        .iter()
        .zip(noise.iter())
        .map(|(g, n)| g + n)
        .collect();

    // Step 4: Update weights
    for (w, ng) in weights.iter_mut().zip(noised_gradient.iter()) {
        *w -= config.learning_rate * ng;
    }

    noised_gradient
}

/// Rényi Differential Privacy accountant.
/// Tracks cumulative privacy loss across training rounds using
/// Rényi divergence composition (tighter than standard composition).
pub struct RenyiAccountant {
    /// Rényi divergence order alpha.
    alpha: f64,
    /// Accumulated Rényi divergence.
    rdp_accumulated: f64,
    /// Target delta for conversion to (epsilon, delta)-DP.
    target_delta: f64,
    /// History of per-round epsilon.
    pub round_epsilons: Vec<f64>,
}

impl RenyiAccountant {
    pub fn new(alpha: f64, target_delta: f64) -> Self {
        Self {
            alpha,
            rdp_accumulated: 0.0,
            target_delta,
            round_epsilons: Vec::new(),
        }
    }

    /// Record one round of DP-SGD with given noise_sigma and clip_bound.
    /// Computes the RDP guarantee for a single Gaussian mechanism application.
    pub fn accumulate_round(&mut self, noise_sigma: f64, clip_bound: f64) -> f64 {
        // RDP for Gaussian mechanism: rho(alpha) = alpha / (2 * sigma^2)
        // where sensitivity = clip_bound and sigma = noise_sigma / clip_bound
        let sigma_ratio = noise_sigma / clip_bound;
        let rdp_round = self.alpha / (2.0 * sigma_ratio * sigma_ratio);
        self.rdp_accumulated += rdp_round;

        // Convert accumulated RDP to (epsilon, delta)-DP
        let epsilon = self.rdp_accumulated - (self.target_delta.ln()) / (self.alpha - 1.0);
        self.round_epsilons.push(epsilon);
        epsilon
    }

    /// Current cumulative epsilon.
    pub fn current_epsilon(&self) -> f64 {
        *self.round_epsilons.last().unwrap_or(&0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_clipping() {
        let mut grad = Gradient {
            participant_id: "test".to_string(),
            values: vec![3.0, 4.0], // norm = 5.0
        };
        clip_gradient(&mut grad, 1.0);
        let norm: f64 = grad.values.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_noise_deterministic() {
        let n1 = generate_noise(10, 1.0, 42);
        let n2 = generate_noise(10, 1.0, 42);
        assert_eq!(n1, n2, "Same seed must produce identical noise");
    }

    #[test]
    fn test_renyi_accountant_monotonic() {
        let mut acc = RenyiAccountant::new(8.0, 1e-5);
        let e1 = acc.accumulate_round(1.0, 1.0);
        let e2 = acc.accumulate_round(1.0, 1.0);
        assert!(e2 > e1, "Cumulative epsilon must increase");
    }
}
