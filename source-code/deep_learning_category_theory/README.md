# Category-Theory Deep Learning Framework (Python)

**Reference:** Jia, Peng, Yang & Chen (2025).
*"Category-Theoretical and Topos-Theoretical Frameworks in Machine Learning: A Survey."*
Axioms 14(3):204. <https://doi.org/10.3390/axioms14030204>

## Overview

This framework implements all five categorical perspectives from the survey paper as
idiomatic, self-contained Python (no external ML libraries required).  It is a direct
port of the companion Racket implementation in
`Racket-AI-book/source-code/deep_learning_category_theory/`.

```bash
uv run deep_learning_category_theory.py
```

## Categorical Perspectives Implemented

### I. Para Category + Lens Composition (§2)

Every fully-connected layer is a **morphism in Para(Euc)**:

```
f : P × X → Y       (forward pass)
f*: ∇Y → ∇P × ∇X   (pullback / backward pass)
```

- `forward_para` returns `(activations, pullback_closure)` — this is the **lens**
- Backpropagation = composition of pullback morphisms in reverse: `f* ∘ g* ∘ h*`
- SGD update = **endomorphism** `u_η : Model → Model`
- Networks of arbitrary depth via `make_network` + architecture spec

**Demo:** XOR problem — `input(2) → hidden(4) → hidden(4) → output(1)`, 6000 epochs.

### II. Markov Categories — Stochastic Morphisms (§3)

A **Markov category** has symmetric monoidal structure where morphisms are
stochastic kernels (probability distributions over outputs).

- **Dropout** modelled as a stochastic lens: `X →_s X ⊗ Mask` where `Mask ~ Bernoulli(p)`.
  The same mask is reused in the backward pass (the "closed" optic requirement).
- **Bayesian layer** samples weights `W ~ N(μ, σ²)` at each forward pass —
  a stochastic morphism in the category **Stoch**.
- Monte Carlo uncertainty estimation quantifies epistemic uncertainty.

### III. Invariance & Equivariance (§4)

A layer `f` is **equivariant** w.r.t. group G if `f(g·x) = g·f(x)`.

- **Permutation-invariant pooling**: `Σ xᵢ` is the colimit over the set diagram —
  invariant to any permutation (basis of DeepSets architecture).
- **K-means as categorical colimit**: each centroid is the colimit (average) of
  its cluster; assignment morphisms are the universal maps.

### IV. Topos Framework (§5)

A topos E has a **subobject classifier** Ω and a "true" morphism `⊤ : 1 → Ω`.
Every binary classifier IS a characteristic morphism `χ : X → Ω`.

- **Subobject classifier**: the sigmoid output probability IS `χ_S(x)`.
  The decision boundary is `χ⁻¹(0.5)`.
- **Sheaf gluing**: local expert predictions are "sections"; the gluing lemma
  checks consistency (sheaf condition) before producing a global prediction.
- **Internal logic**: Ω carries a Heyting algebra — `∧`, `∨`, `¬`, `⇒` on `[0,1]`.

### V. Natural Transformations (§2)

A **natural transformation** `η : F ⇒ G` provides a coherent family of morphisms
`η_X : F(X) → G(X)`.  Demonstrated as a knowledge-distillation adapter that maps
the teacher network's 4-dim hidden representation to a 2-dim student space.

## Key Structures

| Class / Function | Categorical Role |
|---|---|
| `LayerParams` | Object in parameter space P |
| `LayerGrads` | Tangent vector at P (gradient) |
| `Model` | Product of layer parameter spaces |
| `BayesianLayer` | Stochastic morphism parameters |
| `SheafSection` | Local section F(U) of a sheaf |
| `NatTransform` | Components of η : F ⇒ G |
| `forward_para` | Para morphism (lens get + put) |
| `model_backward` | Pullback composition (backprop) |
| `model_update` | Endomorphism u_η : Model → Model |
| `make_dropout_lens` | Stochastic lens factory |
| `bayesian_predict_mc` | MC uncertainty estimation |
| `k_means` | Colimit-based clustering |
| `sheaf_glue` | Sheaf condition / gluing lemma |

## Running

```bash
# Using uv (recommended)
uv run deep_learning_category_theory.py
```

No third-party dependencies — uses only the Python standard library (`math`, `random`).

## Design Notes

- **No NumPy**: all linear algebra uses plain Python lists for maximum transparency.
  Each operation maps directly to the mathematical notation in the paper.
- **Typed wrappers** (`InputVec`, `TargetVal`, `Prediction`, …) prevent accidentally
  conflating categorical roles — the Python type is the semantic tag.
- **Closures as morphisms**: pullback closures returned by `forward_para` represent
  the categorical "put-back" of the lens.  This mirrors Racket's first-class functions
  exactly.
- **Arbitrary depth**: unlike the fixed-3-layer version in `neural_network_category_theory/`,
  this implementation supports arbitrary architectures via `make_network([(fi, fo), …])`.
