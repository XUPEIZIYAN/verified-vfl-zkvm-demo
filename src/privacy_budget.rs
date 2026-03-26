//! Privacy budget tracker with hard enforcement cap.
//!
//! Mirrors the on-chain smart contract logic: once cumulative epsilon
//! reaches the pre-agreed ceiling, no further training is permitted.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyBudget {
    /// Pre-agreed ceiling on total epsilon.
    pub budget_ceiling: f64,
    /// Current cumulative epsilon consumed.
    pub epsilon_consumed: f64,
    /// Per-round epsilon history.
    pub round_history: Vec<f64>,
    /// Whether the budget has been exhausted.
    pub exhausted: bool,
}

impl PrivacyBudget {
    pub fn new(budget_ceiling: f64) -> Self {
        Self {
            budget_ceiling,
            epsilon_consumed: 0.0,
            round_history: Vec::new(),
            exhausted: false,
        }
    }

    /// Attempt to consume epsilon for one training round.
    /// Returns Ok(new_total) if within budget, Err if budget would be exceeded.
    pub fn consume(&mut self, round_epsilon: f64) -> Result<f64, String> {
        if self.exhausted {
            return Err("Privacy budget permanently exhausted. Training halted.".into());
        }

        let new_total = self.epsilon_consumed + round_epsilon;
        if new_total > self.budget_ceiling {
            self.exhausted = true;
            return Err(format!(
                "Budget exceeded: consuming {:.4} would bring total to {:.4}, \
                 exceeding ceiling {:.4}. Training permanently halted.",
                round_epsilon, new_total, self.budget_ceiling
            ));
        }

        self.epsilon_consumed = new_total;
        self.round_history.push(round_epsilon);
        Ok(new_total)
    }

    /// Remaining budget.
    pub fn remaining(&self) -> f64 {
        (self.budget_ceiling - self.epsilon_consumed).max(0.0)
    }

    /// Fraction of budget consumed.
    pub fn utilization(&self) -> f64 {
        self.epsilon_consumed / self.budget_ceiling
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_enforcement() {
        let mut budget = PrivacyBudget::new(3.0);
        assert!(budget.consume(1.0).is_ok());
        assert!(budget.consume(1.0).is_ok());
        assert!(budget.consume(1.5).is_err());
        assert!(budget.exhausted);
        // Once exhausted, all further attempts fail
        assert!(budget.consume(0.1).is_err());
    }
}
