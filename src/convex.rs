//! Convex generating functions φ for Bregman divergences.
//!
//! Every Bregman divergence is determined by its *generating function* φ —
//! a strictly convex, differentiable function. This module provides:
//!
//! - A [`ConvexFunction`] trait for defining new generating functions
//! - Built-in generators: squared norm, negative entropy, negative log, exponential
//! - Legendre (convex) conjugate computation
//! - Convexity verification via positive-definite Hessian

use serde::{Deserialize, Serialize};

/// A strictly convex, differentiable generating function φ: ℝⁿ → ℝ.
///
/// Implementors define the function value, gradient, and Hessian at a point.
/// The Hessian must be positive definite everywhere (strict convexity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvexFunction {
    /// φ(x) = ½‖x‖² — generates squared Euclidean distance.
    SquaredNorm,
    /// φ(x) = Σ xᵢ log(xᵢ) — generates KL divergence (negative entropy).
    NegativeEntropy,
    /// φ(x) = -Σ log(xᵢ) — generates Itakura-Saito divergence.
    NegativeLog,
    /// φ(x) = Σ eˣⁱ — generates exponential Bregman divergence.
    Exponential,
    /// φ(x) = ½ xᵀAx with positive-definite A (Mahalanobis).
    Mahalanobis { a: Vec<Vec<f64>> },
}

impl ConvexFunction {
    /// Evaluate the generating function φ at point `x`.
    pub fn value(&self, x: &[f64]) -> f64 {
        match self {
            ConvexFunction::SquaredNorm => x.iter().map(|xi| 0.5 * xi * xi).sum(),
            ConvexFunction::NegativeEntropy => x
                .iter()
                .map(|&xi| if xi > 0.0 { xi * xi.ln() } else { 0.0 })
                .sum(),
            ConvexFunction::NegativeLog => x
                .iter()
                .map(|&xi| if xi > 0.0 { -xi.ln() } else { f64::INFINITY })
                .sum(),
            ConvexFunction::Exponential => x.iter().map(|xi| xi.exp()).sum(),
            ConvexFunction::Mahalanobis { a } => {
                let mut result = 0.0;
                for i in 0..x.len() {
                    for j in 0..x.len() {
                        result += x[i] * a[i][j] * x[j];
                    }
                }
                0.5 * result
            }
        }
    }

    /// Compute the gradient ∇φ(x).
    pub fn gradient(&self, x: &[f64]) -> Vec<f64> {
        match self {
            ConvexFunction::SquaredNorm => x.to_vec(),
            ConvexFunction::NegativeEntropy => x
                .iter()
                .map(|&xi| {
                    if xi > 0.0 {
                        1.0 + xi.ln()
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect(),
            ConvexFunction::NegativeLog => x
                .iter()
                .map(|&xi| {
                    if xi > 0.0 {
                        -1.0 / xi
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect(),
            ConvexFunction::Exponential => x.iter().map(|xi| xi.exp()).collect(),
            ConvexFunction::Mahalanobis { a } => {
                let n = x.len();
                let mut grad = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        grad[i] += a[i][j] * x[j];
                    }
                }
                grad
            }
        }
    }

    /// Compute the Hessian matrix Hφ(x) (diagonal for built-in generators).
    #[allow(clippy::needless_range_loop)]
    pub fn hessian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let n = x.len();
        match self {
            ConvexFunction::SquaredNorm => {
                let mut h = vec![vec![0.0; n]; n];
                for i in 0..n {
                    h[i][i] = 1.0;
                }
                h
            }
            ConvexFunction::NegativeEntropy => {
                let mut h = vec![vec![0.0; n]; n];
                for i in 0..n {
                    h[i][i] = if x[i] > 0.0 {
                        1.0 / x[i]
                    } else {
                        f64::INFINITY
                    };
                }
                h
            }
            ConvexFunction::NegativeLog => {
                let mut h = vec![vec![0.0; n]; n];
                for i in 0..n {
                    h[i][i] = if x[i] > 0.0 {
                        1.0 / (x[i] * x[i])
                    } else {
                        f64::INFINITY
                    };
                }
                h
            }
            ConvexFunction::Exponential => {
                let mut h = vec![vec![0.0; n]; n];
                for i in 0..n {
                    h[i][i] = x[i].exp();
                }
                h
            }
            ConvexFunction::Mahalanobis { a } => a.clone(),
        }
    }

    /// Verify strict convexity: Hessian must be positive definite at `x`.
    ///
    /// Uses Sylvester's criterion: all leading principal minors > 0.
    pub fn verify_convexity(&self, x: &[f64]) -> bool {
        let h = self.hessian(x);
        is_positive_definite(&h)
    }

    /// Compute the Legendre (convex) conjugate φ*(y) = sup_x { ⟨y, x⟩ - φ(x) }.
    ///
    /// For each built-in generator, we use the known closed-form conjugate:
    /// - SquaredNorm: φ*(y) = ½‖y‖² (self-conjugate!)
    /// - NegativeEntropy: φ*(y) = Σ e^(yᵢ - 1)
    /// - NegativeLog: φ*(y) = -Σ(1 + ln(-yᵢ)) for yᵢ < 0
    /// - Exponential: φ*(y) = Σ yᵢ(ln(yᵢ) - 1) for yᵢ > 0
    pub fn conjugate_value(&self, y: &[f64]) -> f64 {
        match self {
            ConvexFunction::SquaredNorm => y.iter().map(|yi| 0.5 * yi * yi).sum(),
            ConvexFunction::NegativeEntropy => y.iter().map(|&yi| (yi - 1.0).exp()).sum(),
            ConvexFunction::NegativeLog => y
                .iter()
                .map(|&yi| {
                    if yi < 0.0 {
                        -(1.0 + (-yi).ln())
                    } else {
                        f64::INFINITY
                    }
                })
                .sum(),
            ConvexFunction::Exponential => y
                .iter()
                .map(|&yi| if yi > 0.0 { yi * (yi.ln() - 1.0) } else { 0.0 })
                .sum(),
            ConvexFunction::Mahalanobis { a } => {
                // φ*(y) = ½ yᵀ A⁻¹ y — need inverse
                let inv = invert_matrix(a);
                if inv.is_none() {
                    return f64::INFINITY;
                }
                let inv = inv.unwrap();
                let n = y.len();
                let mut result = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        result += y[i] * inv[i][j] * y[j];
                    }
                }
                0.5 * result
            }
        }
    }
}

/// Check if a symmetric matrix is positive definite via Sylvester's criterion
/// (all leading principal minors strictly positive).
#[allow(clippy::needless_range_loop)]
pub fn is_positive_definite(matrix: &[Vec<f64>]) -> bool {
    let n = matrix.len();
    if n == 0 {
        return false;
    }
    for k in 1..=n {
        let det = leading_minor(matrix, k);
        if det <= 0.0 {
            return false;
        }
    }
    true
}

/// Compute the k-th leading principal minor (determinant of k×k submatrix).
fn leading_minor(matrix: &[Vec<f64>], k: usize) -> f64 {
    if k == 1 {
        return matrix[0][0];
    }
    if k == 2 {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }
    // Cofactor expansion along the last row
    let mut det = 0.0;
    for j in 0..k {
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        let minor = extract_minor(matrix, k - 1, j, k);
        det += sign * matrix[k - 1][j] * leading_minor(&minor, k - 1);
    }
    det
}

#[allow(clippy::needless_range_loop)]
fn extract_minor(
    matrix: &[Vec<f64>],
    skip_row: usize,
    skip_col: usize,
    size: usize,
) -> Vec<Vec<f64>> {
    let mut result = Vec::with_capacity(size - 1);
    for i in 0..size {
        if i == skip_row {
            continue;
        }
        let mut row = Vec::with_capacity(size - 1);
        for j in 0..size {
            if j == skip_col {
                continue;
            }
            row.push(matrix[i][j]);
        }
        result.push(row);
    }
    result
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
