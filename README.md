# bregman-divergence

> *One formula. Every distance in machine learning.*

Bregman divergences are the **unifying framework** behind the distance measures that power modern machine learning. KL divergence, squared Euclidean distance, Itakura-Saito divergence — they're not different beasts. They're the **same formula** with different generating functions plugged in. This crate gives you the whole family tree.

## Why Bregman Divergences Matter

If you've trained a neural network, clustered data, or estimated a probability distribution, you've used a Bregman divergence — you just didn't know it by that name.

Every "distance" in ML that's *not* a true metric (KL, cross-entropy, IS divergence) turns out to be a **Bregman divergence** — the gap between a convex function and its linear approximation:

```
D_φ(p‖q) = φ(p) - φ(q) - ⟨∇φ(q), p - q⟩
```

Change φ, and you get a different divergence:

```
φ(x) = ½‖x‖²         →  Squared Euclidean distance
φ(x) = Σ xᵢ ln(xᵢ)   →  KL divergence
φ(x) = -Σ ln(xᵢ)     →  Itakura-Saito divergence
```

**It's a mathematical family tree.** The Bregman formula is the parent. KL, Euclidean, and IS are the children — each born from a different generating function. This crate lets you work with the parent *and* the children, switching between them effortlessly.

## The Metaphor

Think of Bregman divergence as a **copy machine with interchangeable lenses**. The machine is always the same — it measures how far apart two points are by looking at the gap between a convex surface and its tangent plane. The *lens* is the generating function φ:

- Put on the **quadratic lens** (φ = ½‖x‖²), and you see Euclidean geometry
- Put on the **entropic lens** (φ = Σ xᵢ ln xᵢ), and you see information geometry
- Put on the **logarithmic lens** (φ = -Σ ln xᵢ), and you see spectral geometry

Same machine. Different worlds.

## Architecture

```
                    ┌─────────────────────────────┐
                    │     ConvexFunction (φ)       │
                    │  ┌─────┐ ┌────────┐ ┌─────┐ │
                    │  │x²   │ │x ln x  │ │-ln x│ │
                    │  └──┬──┘ └───┬────┘ └──┬──┘ │
                    │     │        │         │    │
                    └─────┼────────┼─────────┼────┘
                          │        │         │
          ┌───────────────┼────────┼─────────┼──────────────┐
          │  BregmanDivergence: D_φ(p‖q) = φ(p) - φ(q)     │
          │                - ⟨∇φ(q), p - q⟩                 │
          └───┬───────────┬┴────────┴┬───────────┬──────────┘
              │           │          │           │
     ┌────────┴──┐  ┌─────┴────┐  ┌─┴────────┐  │
     │ Squared   │  │Kullback  │  │Itakura   │  │
     │ Euclidean │  │Leibler   │  │Saito     │  │
     │  ½‖p-q‖²  │  │ Σ p ln(p/q)│ │ p/q-ln(p/q)-1│
     └───────────┘  └──────────┘  └──────────┘  │
              │           │          │           │
              └───────────┼──────────┼───────────┘
                          │          │
                    ┌─────┴──────────┴─────┐
                    │    Mirror Descent     │
                    │  Gradient descent in  │
                    │  Bregman geometry     │
                    └───────────────────────┘
```

## Module Overview

| Module | Purpose | Key Types |
|--------|---------|-----------|
| [`bregman`] | General Bregman divergence framework | `BregmanDivergence` |
| [`convex`] | Convex generating functions φ | `ConvexFunction` |
| [`kl`] | Kullback-Leibler divergence | `KullbackLeibler` |
| [`euclidean`] | Squared Euclidean & Mahalanobis | `SquaredEuclidean`, `MahalanobisDistance` |
| [`itakura_saito`] | Itakura-Saito divergence | `ItakuraSaito` |
| [`mirror`] | Mirror descent optimization | `MirrorDescent` |

## Quick Start

```toml
[dependencies]
bregman-divergence = "0.1"
```

### Compute KL Divergence

```rust
use bregman_divergence::KullbackLeibler;

let kl = KullbackLeibler::new();
let p = vec![0.3, 0.7];
let q = vec![0.5, 0.5];

let divergence = kl.divergence(&p, &q);
println!("KL(p‖q) = {:.6}", divergence);
// KL(p‖q) = 0.082283
```

### Compute Squared Euclidean Distance

```rust
use bregman_divergence::SquaredEuclidean;

let se = SquaredEuclidean::new();
let a = vec![1.0, 2.0, 3.0];
let b = vec![4.0, 5.0, 6.0];

let div = se.divergence(&a, &b);
let dist = se.distance(&a, &b);
println!("D(a‖b) = {}", div);  // 13.5
println!("‖a-b‖  = {}", dist);  // 5.196...
```

### Compute Itakura-Saito Divergence

```rust
use bregman_divergence::ItakuraSaito;

let is = ItakuraSaito::new();
let spectrum_p = vec![2.0, 3.0, 1.5, 0.8];
let spectrum_q = vec![1.0, 2.0, 1.0, 1.0];

let total = is.divergence(&spectrum_p, &spectrum_q);
let per_bin = is.spectral_divergence(&spectrum_p, &spectrum_q);
println!("Total IS: {}", total);
println!("Per-bin IS: {}", per_bin);
```

### General Bregman Divergence

```rust
use bregman_divergence::{BregmanDivergence, ConvexFunction};

// Any generating function → a Bregman divergence
let div = BregmanDivergence::new(ConvexFunction::NegativeEntropy);
let p = vec![0.3, 0.7];
let q = vec![0.5, 0.5];

let d = div.divergence(&p, &q);
println!("D_φ(p‖q) = {:.6}", d);

// Verify the Pythagorean theorem
let z = vec![0.4, 0.6];
let (lhs, rhs, holds) = div.pythagorean(&p, &q, &z);
println!("D(p‖z) = {}, D(p‖q) + D(q‖z) = {}", lhs, rhs);
println!("Pythagorean holds: {}", holds);
```

### Mirror Descent Optimization

```rust
use bregman_divergence::{MirrorDescent, ConvexFunction};

// Minimize f(x) = ½(x-3)² using Euclidean mirror descent (= gradient descent)
let md = MirrorDescent::new(ConvexFunction::SquaredNorm, 0.5);

let objective = |x: &[f64]| 0.5 * (x[0] - 3.0).powi(2);
let gradient = |x: &[f64]| vec![x[0] - 3.0];

let (solution, history) = md.optimize(vec![0.0], objective, gradient, 100);
println!("Minimum at x = {:.4}", solution[0]);  // ≈ 3.0

// Entropic mirror descent for simplex-constrained optimization
let md_ent = MirrorDescent::new(ConvexFunction::NegativeEntropy, 0.1);
let x = vec![0.5, 0.5];
let grad = vec![1.0, -1.0];
let new_x = md_ent.step(&x, &grad);
// new_x stays positive (on the simplex) automatically!
```

### Convex Generating Functions

```rust
use bregman_divergence::ConvexFunction;

let phi = ConvexFunction::NegativeEntropy;
let x = vec![0.3, 0.7];

// Function value
println!("φ(x) = {}", phi.value(&x));

// Gradient (mirror map)
println!("∇φ(x) = {:?}", phi.gradient(&x));

// Hessian (verify strict convexity)
println!("Convex: {}", phi.verify_convexity(&x));

// Legendre conjugate φ*(y) = sup_x {⟨y,x⟩ - φ(x)}
let y = vec![1.5, 2.0];
println!("φ*(y) = {}", phi.conjugate_value(&y));
```

### Mahalanobis Distance

```rust
use bregman_divergence::MahalanobisDistance;

// Weighted distance with diagonal covariance
let maha = MahalanobisDistance::new(vec![
    vec![4.0, 0.0],
    vec![0.0, 0.25],
]);
let p = vec![1.0, 2.0];
let q = vec![0.0, 0.0];

println!("Mahalanobis: {:.4}", maha.divergence(&p, &q));
```

## Mathematical Foundations

### The Bregman Divergence

Given a **strictly convex**, continuously differentiable function φ: ℝⁿ → ℝ (the *generating function*), the Bregman divergence is:

```
D_φ(p‖q) = φ(p) - φ(q) - ⟨∇φ(q), p - q⟩
```

**Geometric interpretation**: D_φ(p‖q) is the vertical distance between φ(p) and the tangent plane to φ at q, evaluated at p. It measures how much the convex function "bends away" from its linear approximation.

### The Family Tree

| Generating Function φ | Divergence D_φ | Domain |
|---|---|---|
| ½‖x‖² | ½‖p-q‖² (squared Euclidean) | ℝⁿ |
| Σ xᵢ ln(xᵢ) | Σ pᵢ ln(pᵢ/qᵢ) (KL) | Probability simplex |
| -Σ ln(xᵢ) | Σ [pᵢ/qᵢ - ln(pᵢ/qᵢ) - 1] (Itakura-Saito) | ℝ₊ⁿ |
| ½xᵀAx (PD A) | ½(p-q)ᵀA⁻¹(p-q) (Mahalanobis) | ℝⁿ |
| Σ eˣⁱ | Σ [eᵖⁱ - eᑫⁱ - eᑫⁱ(pᵢ-qᵢ)] | ℝⁿ |

### Legendre Duality

Every convex function φ has a **convex conjugate** φ* defined by the Legendre-Fenchel transform:

```
φ*(y) = sup_x { ⟨y, x⟩ - φ(x) }
```

The **Fenchel-Young inequality** states:

```
φ(x) + φ*(y) ≥ ⟨x, y⟩
```

with equality if and only if y = ∇φ(x). This duality is the mathematical backbone of mirror descent.

### Generalized Pythagorean Theorem

For a Bregman divergence D_φ and a convex set C, the projection of p onto C in the Bregman sense (minimizing D_φ(p‖q) over q ∈ C) satisfies:

```
D_φ(p‖z) = D_φ(p‖q) + D_φ(q‖z)
```

where q is the Bregman projection and z is any point in C on the "Bregman geodesic" between p and q. This generalizes the familiar Pythagorean theorem from Euclidean geometry to arbitrary Bregman geometries.

### Properties of Bregman Divergences

| Property | Description |
|---|---|
| **Non-negativity** | D_φ(p‖q) ≥ 0, with equality iff p = q |
| **Convexity in p** | D_φ(·‖q) is convex for any q |
| **Not a metric** | Generally non-symmetric, no triangle inequality |
| **Linearity in φ** | D_{αφ+βψ} = αD_φ + βD_ψ |
| **Three-point identity** | D_φ(p‖q) + D_φ(q‖z) - D_φ(p‖z) = ⟨∇φ(z) - ∇φ(q), p - q⟩ |

### Mirror Descent

Mirror descent replaces the Euclidean proximal step in gradient descent with a Bregman proximal step:

```
Standard GD:  x_{t+1} = x_t - η∇f(x_t)
Mirror:       θ_{t+1} = ∇φ(x_t) - η∇f(x_t)     (dual space update)
              x_{t+1} = (∇φ)⁻¹(θ_{t+1})         (primal space mapping)
```

The choice of φ determines the geometry:
- φ = ½‖x‖² → standard gradient descent
- φ = Σ xᵢ ln(xᵢ) → exponentiated gradient (stays on simplex)
- φ = -Σ ln(xᵢ) → stays in positive orthant

**Convergence rate**: O(1/√T) for convex objectives, dependent on the strong convexity properties of φ.

## Design Decisions

### Why Zero Dependencies (Except Serde)?

Math libraries should be lean. Every external dependency is a supply-chain risk and a compilation cost. We depend only on `serde` for serialization because interoperability matters — you should be able to send Bregman divergences over the wire, save them to disk, or share them between processes.

### Why an Enum for ConvexFunction?

We could have used a trait, but traits can't be serialized. Since the set of useful generating functions is small and well-known, an enum gives us:

- **Serializable types** — every public type derives `Serialize + Deserialize`
- **Exhaustive matching** — the compiler tells you if you forget a case
- **Zero allocation overhead** — no vtable indirection

### Why Scalar + Vector APIs?

Some divergences (Itakura-Saito) operate per-frequency-bin in signal processing but also make sense as a single scalar. We provide both `divergence_scalar` and `divergence` (vector) to match the use case.

### Why Not Generic Over f32/f64?

Precision matters in numerical divergence computation. The Hessian checks and Sylvester's criterion accumulate error. We use `f64` throughout to keep things correct. If you need `f32`, file an issue.

### Edition 2024

We use Rust Edition 2024 for the latest language features and idioms.

## Feature Comparison

| Feature | This crate | Typical alternative |
|---|---|---|
| General Bregman framework | ✅ | ❌ (usually just KL) |
| KL divergence | ✅ | ✅ |
| Squared Euclidean | ✅ | Manual |
| Itakura-Saito | ✅ | Rare |
| Mahalanobis | ✅ | Rare |
| Convex generating functions | ✅ | ❌ |
| Legendre conjugate | ✅ | ❌ |
| Mirror descent | ✅ | ❌ |
| Pythagorean theorem | ✅ | ❌ |
| Convexity verification | ✅ | ❌ |
| Serialization (serde) | ✅ | Varies |
| Zero external deps (except serde) | ✅ | ❌ |

## Testing

The crate has 54 tests covering:

- **Non-negativity** for all divergence types
- **Zero when p = q** for all divergences
- **Symmetry** (Euclidean) vs **non-symmetry** (KL, IS)
- **Formula consistency**: specific divergences match general Bregman with the right φ
- **Pythagorean theorem** verification
- **Fenchel-Young inequality** and equality cases
- **Mirror descent convergence** for quadratic and simplex objectives
- **Scale invariance** of IS divergence
- **Cross-entropy decomposition**: H(p,q) = H(p) + KL(p‖q)
- **Jensen-Shannon** symmetry and boundedness
- **Serde roundtrips** for all serializable types
- **Convexity verification** via Sylvester's criterion
- **Mahalanobis** reduces to Euclidean for identity matrix

Run tests:

```bash
cargo test
```

## License

Dual-licensed under MIT OR Apache-2.0.

## Contributing

PRs welcome. Please run `cargo test`, `cargo fmt --check`, and `cargo clippy -- -D warnings` before submitting.
