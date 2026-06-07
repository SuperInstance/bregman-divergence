//! # bregman-divergence
//!
//! A Rust library for Bregman divergences and their applications.
//!
//! ## Modules
//!
//! - [`bregman`] — General Bregman divergence framework
//! - [`kl`] — Kullback-Leibler divergence
//! - [`euclidean`] — Squared Euclidean and Mahalanobis distances
//! - [`itakura_saito`] — Itakura-Saito divergence
//! - [`convex`] — Convex generating functions
//! - [`mirror`] — Mirror descent optimization

pub mod bregman;
pub mod convex;
pub mod euclidean;
pub mod itakura_saito;
pub mod kl;
pub mod mirror;

// Re-export key types
pub use bregman::BregmanDivergence;
pub use convex::ConvexFunction;
pub use euclidean::{MahalanobisDistance, SquaredEuclidean};
pub use itakura_saito::ItakuraSaito;
pub use kl::KullbackLeibler;
pub use mirror::MirrorDescent;

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// All Bregman divergences are non-negative.
    #[test]
    fn test_all_divergences_nonnegative() {
        let p = vec![0.3, 0.7];
        let q = vec![0.5, 0.5];

        // General Bregman with different generators
        let kl_div = BregmanDivergence::new(ConvexFunction::NegativeEntropy);
        assert!(kl_div.divergence(&p, &q) >= -1e-10);

        let se_div = BregmanDivergence::new(ConvexFunction::SquaredNorm);
        assert!(se_div.divergence(&p, &q) >= -1e-10);

        // Specific divergences
        let kl = KullbackLeibler::new();
        assert!(kl.divergence(&p, &q) >= -1e-10);

        let se = SquaredEuclidean::new();
        assert!(se.divergence(&p, &q) >= -1e-10);

        let p_pos = vec![1.5, 0.8];
        let q_pos = vec![0.7, 1.2];
        let is = ItakuraSaito::new();
        assert!(is.divergence(&p_pos, &q_pos) >= -1e-10);
    }

    /// All divergences return zero when p = q.
    #[test]
    fn test_all_zero_when_equal() {
        let p = vec![2.0, 3.0, 1.0];

        let se_div = BregmanDivergence::new(ConvexFunction::SquaredNorm);
        assert!(se_div.divergence(&p, &p).abs() < 1e-10);

        let se = SquaredEuclidean::new();
        assert!(se.divergence(&p, &p).abs() < 1e-10);

        let p_prob = vec![0.25, 0.25, 0.25, 0.25];
        let kl = KullbackLeibler::new();
        assert!(kl.divergence(&p_prob, &p_prob).abs() < 1e-10);

        let p_pos = vec![1.0, 2.0, 3.0];
        let is = ItakuraSaito::new();
        assert!(is.divergence(&p_pos, &p_pos).abs() < 1e-10);
    }

    /// Squared Euclidean is the unique symmetric Bregman divergence.
    #[test]
    fn test_euclidean_is_symmetric_others_not() {
        let p = vec![2.0, 3.0];
        let q = vec![1.0, 4.0];

        // Squared Euclidean: symmetric
        let se = SquaredEuclidean::new();
        assert!(se.verify_symmetry(&p, &q));

        // KL: not symmetric
        let kl = KullbackLeibler::new();
        let p_prob = vec![0.9, 0.1];
        let q_prob = vec![0.3, 0.7];
        assert!((kl.divergence(&p_prob, &q_prob) - kl.divergence(&q_prob, &p_prob)).abs() > 1e-6);

        // IS: not symmetric
        let is = ItakuraSaito::new();
        let p_pos = vec![2.0, 3.0];
        let q_pos = vec![1.0, 4.0];
        assert!((is.divergence(&p_pos, &q_pos) - is.divergence(&q_pos, &p_pos)).abs() > 1e-6);
    }

    /// Verify the Bregman divergence formula matches direct computation.
    #[test]
    fn test_formula_consistency() {
        // KL via Bregman should match KL via direct computation
        let kl_direct = KullbackLeibler::new();
        let kl_bregman = BregmanDivergence::new(ConvexFunction::NegativeEntropy);
        let p = vec![0.3, 0.7];
        let q = vec![0.5, 0.5];
        let direct = kl_direct.divergence(&p, &q);
        let bregman = kl_bregman.divergence(&p, &q);
        assert!((direct - bregman).abs() < 1e-10);
    }

    /// Squared Euclidean via Bregman should match direct computation.
    #[test]
    fn test_euclidean_formula_consistency() {
        let se_direct = SquaredEuclidean::new();
        let se_bregman = BregmanDivergence::new(ConvexFunction::SquaredNorm);
        let p = vec![1.0, 2.0, 3.0];
        let q = vec![4.0, 5.0, 6.0];
        assert!((se_direct.divergence(&p, &q) - se_bregman.divergence(&p, &q)).abs() < 1e-10);
    }

    /// IS divergence via Bregman should match direct computation.
    #[test]
    fn test_is_formula_consistency() {
        let is_direct = ItakuraSaito::new();
        let is_bregman = BregmanDivergence::new(ConvexFunction::NegativeLog);
        let p = vec![2.0, 0.5];
        let q = vec![1.0, 1.0];
        assert!((is_direct.divergence(&p, &q) - is_bregman.divergence(&p, &q)).abs() < 1e-10);
    }

    /// Pythagorean theorem for squared Euclidean (it's exact for flat submanifolds).
    #[test]
    fn test_pythagorean_euclidean() {
        let div = BregmanDivergence::new(ConvexFunction::SquaredNorm);
        // Three collinear points: D(p||z) should equal D(p||q) + D(q||z)
        // when q is between p and z (right angle in Euclidean sense)
        let p = vec![0.0, 3.0];
        let q = vec![0.0, 0.0];
        let z = vec![4.0, 0.0];
        // D(p||z) = ½(16+9) = 12.5
        // D(p||q) = ½(0+9) = 4.5
        // D(q||z) = ½(16+0) = 8.0
        // 4.5 + 8.0 = 12.5 ✓
        let (lhs, rhs, holds) = div.pythagorean(&p, &q, &z);
        assert!(holds, "Pythagorean failed: {lhs} != {rhs}");
    }

    /// Convexity verification for all generators.
    #[test]
    fn test_convexity_verification() {
        let x = vec![2.0, 3.0, 1.0];
        assert!(ConvexFunction::SquaredNorm.verify_convexity(&x));
        assert!(ConvexFunction::NegativeEntropy.verify_convexity(&x));
        assert!(ConvexFunction::NegativeLog.verify_convexity(&x));
        assert!(ConvexFunction::Exponential.verify_convexity(&x));
    }

    /// Convexity verification with Mahalanobis.
    #[test]
    fn test_mahalanobis_convexity() {
        let a = vec![vec![2.0, 0.0], vec![0.0, 3.0]];
        let maha = ConvexFunction::Mahalanobis { a };
        let x = vec![1.0, 1.0];
        assert!(maha.verify_convexity(&x));
    }

    /// Mirror descent with squared norm converges to minimum.
    #[test]
    fn test_mirror_descent_quadratic_convergence() {
        let md = MirrorDescent::new(ConvexFunction::SquaredNorm, 0.5);
        let objective = |x: &[f64]| 0.5 * x.iter().map(|xi| xi * xi).sum::<f64>();
        let gradient = |x: &[f64]| x.to_vec();
        let (final_x, history) = md.optimize(vec![10.0, -5.0], objective, gradient, 200);
        assert!(final_x.iter().all(|xi| xi.abs() < 0.5));
        assert!(md.verify_convergence(&history));
    }

    /// Legendre duality: Fenchel-Young inequality.
    /// φ(x) + φ*(y) ≥ ⟨x, y⟩ with equality iff y = ∇φ(x).
    #[test]
    fn test_fenchel_young_inequality() {
        let phi = ConvexFunction::SquaredNorm;
        let x = vec![2.0, 3.0];
        let y = vec![1.0, 4.0];
        let dot: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let lhs = phi.value(&x) + phi.conjugate_value(&y);
        assert!(lhs >= dot - 1e-10, "Fenchel-Young violated: {lhs} < {dot}");
    }

    /// Legendre duality with equality when y = ∇φ(x).
    #[test]
    fn test_fenchel_young_equality() {
        let phi = ConvexFunction::SquaredNorm;
        let x = vec![2.0, 3.0];
        let y = phi.gradient(&x); // y = x for squared norm
        let dot: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let lhs = phi.value(&x) + phi.conjugate_value(&y);
        assert!((lhs - dot).abs() < 1e-10, "Fenchel-Young equality failed");
    }

    /// Mirror descent with negative entropy keeps iterates positive.
    #[test]
    fn test_entropic_mirror_stays_positive() {
        let md = MirrorDescent::new(ConvexFunction::NegativeEntropy, 0.01);
        let mut x = vec![1.0, 1.0, 1.0];
        for _ in 0..50 {
            let grad = vec![0.5, -0.3, 0.1];
            x = md.step(&x, &grad);
            assert!(
                x.iter().all(|xi| *xi > 0.0),
                "Iterate went non-positive: {:?}",
                x
            );
        }
    }

    /// Scale invariance of IS divergence.
    #[test]
    fn test_is_scale_invariance() {
        let is = ItakuraSaito::new();
        let p = vec![2.0, 3.0, 1.5];
        let q = vec![1.0, 2.0, 0.5];
        assert!(is.verify_scale_invariance(&p, &q, 3.0));
        assert!(is.verify_scale_invariance(&p, &q, 100.0));
        assert!(is.verify_scale_invariance(&p, &q, 0.01));
    }

    /// Mahalanobis reduces to squared Euclidean for identity matrix.
    #[test]
    fn test_mahalanobis_is_euclidean_with_identity() {
        let se = SquaredEuclidean::new();
        let maha = MahalanobisDistance::new(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let p = vec![1.0, 2.0, 3.0];
        let q = vec![4.0, 5.0, 6.0];
        assert!((se.divergence(&p, &q) - maha.divergence(&p, &q)).abs() < 1e-10);
    }

    /// Convex generating function Hessian is positive definite.
    #[test]
    fn test_hessian_positive_definite() {
        use crate::convex::is_positive_definite;
        let phi = ConvexFunction::SquaredNorm;
        let x = vec![1.0, 2.0, 3.0];
        let h = phi.hessian(&x);
        assert!(is_positive_definite(&h));
    }

    /// Bregman information is non-negative.
    #[test]
    fn test_bregman_information_nonnegative() {
        let div = BregmanDivergence::new(ConvexFunction::SquaredNorm);
        let points = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let weights = vec![1.0, 2.0, 1.0];
        let info = div.bregman_information(&points, &weights);
        assert!(info >= -1e-10);
    }

    /// Cross-entropy = entropy + KL.
    #[test]
    fn test_cross_entropy_decomposition() {
        let kl = KullbackLeibler::new();
        let p = vec![0.3, 0.7];
        let q = vec![0.5, 0.5];
        let ce = kl.cross_entropy(&p, &q);
        let h = kl.entropy(&p);
        let kl_val = kl.divergence(&p, &q);
        // H(p,q) = H(p) + KL(p||q)
        assert!((ce - (h + kl_val)).abs() < 1e-10);
    }

    /// Spectral divergence normalizes by length.
    #[test]
    fn test_spectral_divergence() {
        let is = ItakuraSaito::new();
        let p = vec![2.0, 4.0];
        let q = vec![1.0, 2.0];
        let spec = is.spectral_divergence(&p, &q);
        let total = is.divergence(&p, &q);
        assert!((spec - total / 2.0).abs() < 1e-10);
    }

    /// Multiple mirror descent steps converge for convex objective.
    #[test]
    fn test_mirror_descent_2d_convergence() {
        let md = MirrorDescent::new(ConvexFunction::SquaredNorm, 0.3);
        let objective = |x: &[f64]| (x[0] - 5.0).powi(2) + (x[1] + 3.0).powi(2);
        let gradient = |x: &[f64]| vec![2.0 * (x[0] - 5.0), 2.0 * (x[1] + 3.0)];
        let (final_x, history) = md.optimize(vec![0.0, 0.0], objective, gradient, 100);
        assert!((final_x[0] - 5.0).abs() < 1.0);
        assert!((final_x[1] + 3.0).abs() < 1.0);
        assert!(md.verify_convergence(&history));
    }

    /// JS divergence is bounded in [0, ln(2)].
    #[test]
    fn test_js_bounded() {
        let kl = KullbackLeibler::new();
        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        let js = kl.jensen_shannon(&p, &q);
        assert!(js <= 2.0_f64.ln() + 1e-10);
        assert!(js >= 0.0);
    }

    /// Test serialization roundtrip.
    #[test]
    fn test_serde_roundtrip() {
        let kl = KullbackLeibler::new();
        let json = serde_json::to_string(&kl).unwrap();
        let kl2: KullbackLeibler = serde_json::from_str(&json).unwrap();
        let p = vec![0.5, 0.5];
        let q = vec![0.3, 0.7];
        assert!((kl.divergence(&p, &q) - kl2.divergence(&p, &q)).abs() < 1e-10);
    }

    /// Test ConvexFunction serialization.
    #[test]
    fn test_convex_function_serde() {
        let phi = ConvexFunction::SquaredNorm;
        let json = serde_json::to_string(&phi).unwrap();
        let phi2: ConvexFunction = serde_json::from_str(&json).unwrap();
        let x = vec![1.0, 2.0];
        assert!((phi.value(&x) - phi2.value(&x)).abs() < 1e-10);
    }

    /// Test MirrorDescent serialization.
    #[test]
    fn test_mirror_descent_serde() {
        let md = MirrorDescent::new(ConvexFunction::SquaredNorm, 0.1);
        let json = serde_json::to_string(&md).unwrap();
        let md2: MirrorDescent = serde_json::from_str(&json).unwrap();
        let x = vec![1.0, 2.0];
        let grad = vec![0.5, -0.5];
        let r1 = md.step(&x, &grad);
        let r2 = md2.step(&x, &grad);
        assert_eq!(r1, r2);
    }

    /// Entropy of a deterministic distribution is zero.
    #[test]
    fn test_entropy_deterministic() {
        let kl = KullbackLeibler::new();
        let p = vec![1.0, 0.0, 0.0];
        assert!((kl.entropy(&p)).abs() < 1e-10);
    }

    /// Exponential generating function conjugate.
    #[test]
    fn test_exponential_conjugate() {
        let phi = ConvexFunction::Exponential;
        let y = vec![2.0, 3.0];
        let conj = phi.conjugate_value(&y);
        // φ*(y) = Σ yᵢ(ln yᵢ - 1)
        let expected: f64 = y.iter().map(|&yi| yi * (yi.ln() - 1.0)).sum();
        assert!((conj - expected).abs() < 1e-10);
    }

    /// Negative entropy conjugate.
    #[test]
    fn test_negative_entropy_conjugate() {
        let phi = ConvexFunction::NegativeEntropy;
        let y = vec![1.0, 2.0];
        let conj = phi.conjugate_value(&y);
        // φ*(y) = Σ e^(yᵢ - 1)
        let expected: f64 = y.iter().map(|&yi| (yi - 1.0).exp()).sum();
        assert!((conj - expected).abs() < 1e-10);
    }
}
