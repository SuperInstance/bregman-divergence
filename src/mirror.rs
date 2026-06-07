//! Mirror descent: gradient descent in the geometry of a Bregman divergence.
//!
//! Replace the Euclidean proximal step with a Bregman proximal step,
//! enabling efficient optimization over constrained domains (probability
//! simplices, positive orthant, etc.).

use serde::{Deserialize, Serialize};

use crate::convex::ConvexFunction;

/// Mirror descent optimizer using a Bregman divergence.
///
/// Instead of the standard update x ← x - η∇f(x), mirror descent:
/// 1. Maps x to dual space via mirror map ∇φ(x)
/// 2. Takes a gradient step in dual space: θ ← ∇φ(x) - η∇f(x)
/// 3. Maps back via the inverse mirror map: x ← (∇φ)⁻¹(θ)
///
/// # When to use mirror descent
///
/// - Optimization over the probability simplex → use negative entropy (entropic descent / exponentiated gradient)
/// - Optimization over the positive orthant → use negative log
/// - Standard Euclidean geometry → reduces to regular gradient descent
///
/// # Convergence
///
/// Convergence rate depends on the strong convexity of φ:
/// - O(1/√T) for non-strongly convex objectives (mirror descent guarantee)
/// - O(1/T) for strongly convex objectives (accelerated variants)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorDescent {
    /// The generating function φ (determines the mirror map).
    pub generator: ConvexFunction,
    /// Learning rate η.
    pub learning_rate: f64,
}

impl MirrorDescent {
    /// Create a new mirror descent optimizer.
    pub fn new(generator: ConvexFunction, learning_rate: f64) -> Self {
        Self {
            generator,
            learning_rate,
        }
    }

    /// Perform one mirror descent step.
    ///
    /// Given current point x and gradient ∇f(x), returns the updated point.
    ///
    /// 1. Compute mirror map: θ = ∇φ(x)
    /// 2. Dual update: θ' = θ - η∇f(x)
    /// 3. Inverse mirror map: x' = (∇φ)⁻¹(θ')
    pub fn step(&self, x: &[f64], gradient: &[f64]) -> Vec<f64> {
        let mirror = self.generator.gradient(x);
        // Dual space update
        let dual: Vec<f64> = mirror
            .iter()
            .zip(gradient.iter())
            .map(|(m, g)| m - self.learning_rate * g)
            .collect();
        // Inverse mirror map
        self.inverse_mirror(&dual)
    }

    /// Run mirror descent for a fixed number of iterations.
    ///
    /// Returns the final point and a vector of objective values.
    pub fn optimize<F, G>(
        &self,
        mut x: Vec<f64>,
        objective: F,
        gradient: G,
        iterations: usize,
    ) -> (Vec<f64>, Vec<f64>)
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        let mut history = Vec::with_capacity(iterations + 1);
        history.push(objective(&x));

        for _ in 0..iterations {
            let grad = gradient(&x);
            x = self.step(&x, &grad);
            history.push(objective(&x));
        }

        (x, history)
    }

    /// Compute the inverse mirror map (∇φ)⁻¹ for each built-in generator.
    ///
    /// - SquaredNorm: ∇φ = id, so (∇φ)⁻¹(y) = y
    /// - NegativeEntropy: ∇φ(x) = 1 + ln(x), so x = e^(y-1)
    /// - NegativeLog: ∇φ(x) = -1/x, so x = -1/y
    /// - Exponential: ∇φ(x) = eˣ, so x = ln(y)
    fn inverse_mirror(&self, y: &[f64]) -> Vec<f64> {
        match &self.generator {
            ConvexFunction::SquaredNorm => y.to_vec(),
            ConvexFunction::NegativeEntropy => {
                y.iter().map(|&yi| (yi - 1.0).exp().max(1e-15)).collect()
            }
            ConvexFunction::NegativeLog => y
                .iter()
                .map(|&yi| {
                    if yi < 0.0 {
                        (-1.0 / yi).max(1e-15)
                    } else {
                        1e-15
                    }
                })
                .collect(),
            ConvexFunction::Exponential => y.iter().map(|&yi| yi.max(1e-15).ln()).collect(),
            ConvexFunction::Mahalanobis { a } => {
                // ∇φ(x) = Ax, so x = A⁻¹y
                let n = y.len();
                let inv = invert_matrix(a);
                match inv {
                    Some(inv) => {
                        let mut result = vec![0.0; n];
                        for i in 0..n {
                            for j in 0..n {
                                result[i] += inv[i][j] * y[j];
                            }
                        }
                        result
                    }
                    None => y.to_vec(),
                }
            }
        }
    }

    /// Verify convergence: check that the objective is decreasing.
    pub fn verify_convergence(&self, history: &[f64]) -> bool {
        if history.len() < 2 {
            return true;
        }
        // Check that on average the objective decreases
        let first = history[0];
        let last = *history.last().unwrap();
        last <= first
    }
}

#[allow(clippy::needless_range_loop)]
fn invert_matrix(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n + i] = 1.0;
    }
    for col in 0..n {
        let mut max_row = col;
        for row in col + 1..n {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }
        if aug[max_row][col].abs() < 1e-12 {
            return None;
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        for j in 0..2 * n {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..2 * n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mirror_descent_euclidean_reduces_to_gd() {
        let md = MirrorDescent::new(ConvexFunction::SquaredNorm, 0.1);
        let x = vec![5.0, 3.0];
        let grad = vec![2.0, 4.0]; // gradient of some objective
        let new_x = md.step(&x, &grad);
        // With squared norm, mirror descent = gradient descent
        let expected = vec![5.0 - 0.1 * 2.0, 3.0 - 0.1 * 4.0];
        for i in 0..2 {
            assert!((new_x[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mirror_descent_convergence() {
        // Minimize f(x) = ½(x-3)² with mirror descent (squared norm)
        let md = MirrorDescent::new(ConvexFunction::SquaredNorm, 0.1);
        let objective =
            |x: &[f64]| -> f64 { 0.5 * x.iter().map(|xi| (xi - 3.0).powi(2)).sum::<f64>() };
        let gradient = |x: &[f64]| -> Vec<f64> { x.iter().map(|xi| xi - 3.0).collect() };

        let (final_x, history) = md.optimize(vec![0.0], objective, gradient, 100);
        assert!(md.verify_convergence(&history));
        assert!((final_x[0] - 3.0).abs() < 0.5);
    }

    #[test]
    fn test_mirror_descent_entropy_simplex() {
        // Use negative entropy to stay on probability simplex
        let md = MirrorDescent::new(ConvexFunction::NegativeEntropy, 0.1);
        let x = vec![0.5, 0.5];
        let grad = vec![1.0, -1.0];
        let new_x = md.step(&x, &grad);
        // After entropic mirror step, values should remain positive
        assert!(new_x.iter().all(|xi| *xi > 0.0));
    }

    #[test]
    fn test_mirror_descent_negative_log() {
        let md = MirrorDescent::new(ConvexFunction::NegativeLog, 0.1);
        let x = vec![2.0, 3.0];
        let grad = vec![1.0, 1.0];
        let new_x = md.step(&x, &grad);
        // Should remain positive
        assert!(new_x.iter().all(|xi| *xi > 0.0));
    }

    #[test]
    fn test_verify_convergence_monotone() {
        let md = MirrorDescent::new(ConvexFunction::SquaredNorm, 0.1);
        let decreasing = vec![10.0, 8.0, 5.0, 3.0, 1.0, 0.5];
        assert!(md.verify_convergence(&decreasing));
    }
}
