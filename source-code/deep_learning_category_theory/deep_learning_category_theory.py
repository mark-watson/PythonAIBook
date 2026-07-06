"""
Category-Theory Deep Learning Framework (Python)
=================================================

Reference: Jia, Peng, Yang & Chen (2025).
  "Category-Theoretical and Topos-Theoretical Frameworks in Machine Learning:
   A Survey."  Axioms 14(3):204.  https://doi.org/10.3390/axioms14030204

The paper organises categorical ML into four perspectives:

  I.   Gradient-based learning
         • Para (parametric maps) - neural-net layers as morphisms
         • Lenses / Optics        - bidirectional forward / backward pass
         • Compositional backprop - functor on base category

  II.  Probability-based learning
         • Markov categories      - stochastic morphisms
         • Bayesian inference functor
         • Dropout as a stochastic lens

  III. Invariance / Equivalence
         • Functor-equivariant layers
         • Categorical clustering (colimits)
         • Persistent homology (sketch)

  IV.  Topos-based learning
         • Subobject classifier   - binary decisions
         • Sheaf-style composition - context propagation

  V.   Natural Transformations
         • Knowledge distillation adapter η : F ⇒ G

This file implements working demonstrations of all five perspectives.

Usage::

    uv run deep_learning_category_theory.py

"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

# =============================================================================
# SECTION I - GRADIENT-BASED LEARNING
#             "Para" category + lens composition + compositional backprop
# =============================================================================
#
# I.1  Para category
#
# In the Para construction a morphism  f : A → B  is a triple
#   (parameter-space P, forward fn, backward fn)
# where
#   forward  : P × A → B
#   backward : P × A × ∇B → ∇P × ∇A    (the "residual" / lens put-back)
#
# Neural-network layers are exactly morphisms in Para(Euc), the category of
# smooth parametric maps between Euclidean spaces (Gavranović 2022; §2.1 of
# the survey).


# ---------------------------------------------------------------------------
# I.1  Typed wrappers - newtypes that encode semantic roles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InputVec:
    """An element of the InputSpace - carries the raw feature vector.

    Categorical role: object in the InputSpace category.
    """
    vals: tuple[float, ...]


@dataclass(frozen=True)
class TargetVal:
    """A supervised label - the target scalar y.

    Categorical role: element of the Target object.
    """
    v: float


@dataclass(frozen=True)
class Prediction:
    """The network's scalar output ŷ after the forward pass.

    Categorical role: element of the Prediction object - the codomain of the
    composed lens morphism.
    """
    v: float


@dataclass(frozen=True)
class LearningRate:
    """The SGD step-size hyper-parameter η.

    Categorical role: scalar that scales the endomorphism on ModelState.
    """
    v: float


@dataclass
class LayerParams:
    """The parameter space P for one fully-connected layer.

    Attributes
    ----------
    W : list[list[float]]
        Weight matrix stored as a list of row vectors. Shape: (fan_out, fan_in).
    b : list[float]
        Bias vector of length fan_out.

    Categorical role: an object in the ParameterSpace category.
    """
    W: list[list[float]]
    b: list[float]


@dataclass
class LayerGrads:
    """Gradients with respect to one layer's parameters.

    Shape mirrors LayerParams - both live in the same tangent space.

    Attributes
    ----------
    dW : list[list[float]]
        Gradient of the loss w.r.t. W. Same shape as W.
    db : list[float]
        Gradient of the loss w.r.t. b. Same shape as b.
    """
    dW: list[list[float]]
    db: list[float]


@dataclass
class Model:
    """The full network - a product of N LayerParams objects.

    Categorical role: an object in the ModelState category (the product
    ParameterSpace₁ × … × ParameterSpaceₙ).

    Attributes
    ----------
    layers : list[LayerParams]
        Ordered list of layer parameter objects, from input to output.
    """
    layers: list[LayerParams]


# ---------------------------------------------------------------------------
# I.2  Math utilities
# ---------------------------------------------------------------------------

def sigmoid(z: float) -> float:
    """The logistic sigmoid activation function σ(z) = 1 / (1 + e^{-z})."""
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_deriv(a: float) -> float:
    """Derivative of sigmoid given its *output* value: σ'(z) = a(1 − a)."""
    return a * (1.0 - a)


def relu(z: float) -> float:
    """Rectified Linear Unit activation: ReLU(z) = max(0, z)."""
    return max(0.0, z)


def relu_deriv(z: float) -> float:
    """Derivative of ReLU: 1 if z > 0 else 0."""
    return 1.0 if z > 0.0 else 0.0


def dot(ws: list[float], xs: list[float]) -> float:
    """Standard vector dot product Σ wᵢ · xᵢ."""
    return sum(w * x for w, x in zip(ws, xs))


def matvec(M: list[list[float]], v: list[float]) -> list[float]:
    """Matrix-vector product: matvec(M, v)[i] = dot(M[i], v)."""
    return [dot(row, v) for row in M]


def vec_add(u: list[float], v: list[float]) -> list[float]:
    """Elementwise vector addition."""
    return [a + b for a, b in zip(u, v)]


def scalar_vec_mul(s: float, v: list[float]) -> list[float]:
    """Multiply every element of vector *v* by scalar *s*."""
    return [s * x for x in v]


def outer(delta: list[float], x: list[float]) -> list[list[float]]:
    """Outer product: outer(δ, x)[i][j] = δᵢ · xⱼ."""
    return [[d * xi for xi in x] for d in delta]


def transpose(M: list[list[float]]) -> list[list[float]]:
    """Transpose a matrix M of shape (m, n) to shape (n, m)."""
    return [list(row) for row in zip(*M)]


def matT_vec(M: list[list[float]], v: list[float]) -> list[float]:
    """Compute Mᵀ · v (transpose-multiply)."""
    return matvec(transpose(M), v)


# ---------------------------------------------------------------------------
# I.3  Para morphism (tracked forward lens)
# ---------------------------------------------------------------------------
#
# forward_para : LayerParams × activation_fn × list[float]
#              → (list[float], pullback_closure)
#
# The pullback closure is the "put-back" half of the lens:
#   pullback : ∇output → (∇params, ∇input)
#
# This directly implements the Para category composition rule (survey §2.1):
#   composition of two Para morphisms threads parameters and residuals.

# Type alias for pullback closures.
PullbackFn = Callable[[list[float]], tuple[LayerGrads, list[float]]]


def forward_para(
    params: LayerParams,
    act: Callable[[float], float],
    act_deriv: Callable[[float], float],
    inputs: list[float],
) -> tuple[list[float], PullbackFn]:
    """Para morphism: forward pass + pullback closure for one layer.

    Category-theory reading
    -----------------------
    This function realises the arrow::

        f : P × X → Y × Ctx

    where
      • P     = LayerParams (W, b)
      • X     = input activations from the previous layer
      • Y     = output activations of this layer
      • Ctx   = the pullback closure (captures W, inputs, zs, acts)

    The returned pullback closure implements::

        f* : ∇Y → ∇P × ∇X

    Parameters
    ----------
    params : LayerParams
        Weight matrix W and bias vector b for this layer.
    act : Callable[[float], float]
        Activation function (e.g. sigmoid, relu).
    act_deriv : Callable[[float], float]
        Derivative of the activation (accepts the pre-activation z).
    inputs : list[float]
        Activation vector arriving from the previous layer.

    Returns
    -------
    acts : list[float]
        Post-activation output of this layer.
    pullback : PullbackFn
        Closure ``upstream_grad → (LayerGrads, input_grad)`` for backprop.
    """
    W = params.W
    b = params.b
    zs = vec_add(matvec(W, inputs), b)   # pre-activations
    acts = [act(z) for z in zs]           # post-activations

    # Pullback closure - the "put-back" of the categorical lens
    def pullback(upstream_grad: list[float]) -> tuple[LayerGrads, list[float]]:
        """Backward morphism: ∇Y → ∇P × ∇X.

        δ = upstream ⊙ act'(z)   (local gradient, Hadamard product)
        ∇W = δ ⊗ xᵀ              (outer product)
        ∇b = δ
        ∇x = Wᵀ · δ              (propagate to previous layer)
        """
        delta = [u * act_deriv(z) for u, z in zip(upstream_grad, zs)]
        dW = outer(delta, inputs)
        db = list(delta)
        dx = matT_vec(W, delta)
        return LayerGrads(dW=dW, db=db), dx

    return acts, pullback


# ---------------------------------------------------------------------------
# I.4  Sequential composition of Para morphisms (the network)
# ---------------------------------------------------------------------------
#
# network_forward : Model × list[float] → (Prediction, list[PullbackFn], list[float])
#
# This implements the functor  F : Para → Para  that maps a sequence of layers
# to their composed forward-pass + stacked pullback closures (survey §2.2).

def network_forward(
    m: Model,
    xs: list[float],
) -> tuple[Prediction, list[PullbackFn], list[float]]:
    """Compose Para morphisms sequentially to produce the network's prediction.

    Category-theory reading
    -----------------------
    Implements the sequential composition::

        F = fₙ ⊚ … ⊚ f₁  :  (P₁ × … × Pₙ) × X → Ŷ × (Ctx₁ × … × Ctxₙ)

    Parameters
    ----------
    m : Model
        Current network weights.
    xs : list[float]
        Raw input features.

    Returns
    -------
    pred : Prediction
        The scalar network output ŷ.
    pullbacks : list[PullbackFn]
        The pullback closures in forward order (reversed during backprop).
    final_acts : list[float]
        The final layer's activation vector.
    """
    current_input = xs
    pullbacks: list[PullbackFn] = []
    params_list = m.layers

    for i, p in enumerate(params_list):
        is_last = (i == len(params_list) - 1)
        # Use sigmoid throughout (identity would also work for the output layer)
        layer_act = sigmoid
        layer_act_deriv = lambda z: sigmoid_deriv(sigmoid(z))  # noqa: E731
        acts, pb = forward_para(p, layer_act, layer_act_deriv, current_input)
        pullbacks.append(pb)
        current_input = acts

    # The final activation is the scalar prediction
    return Prediction(v=current_input[0]), pullbacks, current_input


# ---------------------------------------------------------------------------
# I.5  Loss - typed morphism: Prediction × Target → SquaredError
# ---------------------------------------------------------------------------
#
# Survey §2.3: loss is a natural transformation from the prediction functor
# to the real-number functor.  MSE is the simplest instance.

def mse_loss(pred: Prediction, tgt: TargetVal) -> float:
    """Mean-squared error loss for one sample: L = (ŷ − y)².

    Categorical role: a morphism  Prediction × Target → ℝ  (a scalar).
    """
    d = pred.v - tgt.v
    return d * d


def mse_loss_grad(pred: Prediction, tgt: TargetVal) -> float:
    """Gradient of MSE loss w.r.t. the prediction: dL/dŷ = 2(ŷ − y).

    This is the "seed" upstream gradient that starts the backward pass.
    """
    return 2.0 * (pred.v - tgt.v)


# ---------------------------------------------------------------------------
# I.6  Backward pass - pullback composition (covariant functor on ∇)
# ---------------------------------------------------------------------------
#
# model_backward : list[PullbackFn] × float → list[LayerGrads]
#
# The survey (§2.2) notes that backpropagation is the composite of the
# pullback morphisms in reverse order - a covariant functor on the gradient
# category.

def model_backward(pullbacks: list[PullbackFn], dl_dy_hat: float) -> list[LayerGrads]:
    """Run the backward pass by composing pullback morphisms in reverse order.

    Category-theory reading
    -----------------------
    Implements::

        f₁* ∘ f₂* ∘ … ∘ fₙ*  :  ∇Ŷ → (∇P₁, … , ∇Pₙ, ∇X)

    where each fᵢ* is the pullback closure produced by ``forward_para``.

    Parameters
    ----------
    pullbacks : list[PullbackFn]
        Pullback closures in *forward* order (produced by network_forward).
    dl_dy_hat : float
        dL/dŷ - the seed gradient from the loss function.

    Returns
    -------
    list[LayerGrads]
        Parameter gradients for each layer, in *forward* order.
    """
    upstream: list[float] = [dl_dy_hat]
    grads_acc: list[LayerGrads] = []

    for pb in reversed(pullbacks):
        grads, upstream = pb(upstream)
        grads_acc.append(grads)

    # Return grads in forward layer order
    return list(reversed(grads_acc))


# ---------------------------------------------------------------------------
# I.7  SGD update - endomorphism on ModelState
# ---------------------------------------------------------------------------
#
# Survey §2.1: the gradient-descent update  θ ← θ − η∇θ  is an endomorphism
# u_η : Model → Model on the parameter object of Para.

def update_layer(params: LayerParams, grads: LayerGrads, eta: float) -> LayerParams:
    """Apply one SGD step to a single layer's parameters.

    Implements the elementwise update::

        W := W − η · ∇W
        b := b − η · ∇b

    Returns a *new* LayerParams (the original is not mutated).
    """
    new_W = [
        [w - eta * dw for w, dw in zip(wi, dwi)]
        for wi, dwi in zip(params.W, grads.dW)
    ]
    new_b = [bi - eta * dbi for bi, dbi in zip(params.b, grads.db)]
    return LayerParams(W=new_W, b=new_b)


def model_update(m: Model, grads_list: list[LayerGrads], eta: float) -> Model:
    """Apply one SGD step to the full model - the endomorphism u_η : Model → Model.

    Category-theory reading
    -----------------------
    This is the endomorphism on ModelState::

        u_η : Model → Model,   u_η(θ) = θ − η · ∇θ

    Training is iterated application of ``model_update``.

    Returns a *new* Model (the original is not mutated).
    """
    return Model(
        layers=[update_layer(p, g, eta) for p, g in zip(m.layers, grads_list)]
    )


# ---------------------------------------------------------------------------
# I.8  One training step - the composed morphism
# ---------------------------------------------------------------------------

def train_step(
    m: Model,
    xs: list[float],
    y: TargetVal,
    eta: float,
) -> tuple[Model, float]:
    """One complete training step (forward → loss → backward → update).

    Category-theory reading
    -----------------------
    Realises the composite morphism::

        train_step = update ∘ backward ∘ loss ∘ forward

    which maps  (Model, X, Y, η)  to  (Model', loss_value).

    Returns
    -------
    m_new : Model
        Updated model after one gradient-descent step.
    loss : float
        Scalar loss value for this example before the update.
    """
    pred, pullbacks, _ = network_forward(m, xs)
    loss = mse_loss(pred, y)
    dl_dy = mse_loss_grad(pred, y)
    grads = model_backward(pullbacks, dl_dy)
    return model_update(m, grads, eta), loss


# ---------------------------------------------------------------------------
# I.9  Initialisation
# ---------------------------------------------------------------------------

def glorot_rand(fan_in: int, fan_out: int) -> float:
    """Sample one weight using Glorot-uniform initialisation.

    Draws from U(−limit, limit) where limit = sqrt(6 / (fan_in + fan_out)).
    """
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return random.uniform(-limit, limit)


def make_layer(fan_in: int, fan_out: int) -> LayerParams:
    """Create a randomly initialised LayerParams with Glorot weights and zero biases."""
    W = [
        [glorot_rand(fan_in, fan_out) for _ in range(fan_in)]
        for _ in range(fan_out)
    ]
    b = [0.0] * fan_out
    return LayerParams(W=W, b=b)


def make_network(arch: list[tuple[int, int]]) -> Model:
    """Build a multi-layer network from an architecture specification.

    Parameters
    ----------
    arch : list[tuple[int, int]]
        Each tuple is (fan_in, fan_out) for one layer.
        Example: [(2, 4), (4, 4), (4, 1)] for input(2)→hidden(4)→hidden(4)→output(1).

    Returns
    -------
    Model
        Freshly initialised model with Glorot weights.
    """
    return Model(layers=[make_layer(fi, fo) for fi, fo in arch])


# ---------------------------------------------------------------------------
# I.10  Training loop
# ---------------------------------------------------------------------------

def train(
    m: Model,
    dataset: list[tuple[InputVec, TargetVal]],
    eta: float,
    epochs: int,
    print_every: int = 2000,
) -> Model:
    """Run the full training loop (iterated endomorphism application).

    Category-theory reading
    -----------------------
    Each epoch is one application of the composed endomorphism over the
    dataset::

        m_{t+1} = u_η^{|D|}(m_t)

    Parameters
    ----------
    m : Model
        Initial (untrained) model.
    dataset : list[tuple[InputVec, TargetVal]]
        List of (input, target) pairs.
    eta : float
        Learning-rate scalar η.
    epochs : int
        Number of full passes over the dataset.
    print_every : int
        Print training loss every this many epochs.

    Returns
    -------
    Model
        The trained model after ``epochs`` epochs.
    """
    for epoch in range(epochs):
        total_loss = 0.0
        for inp, tgt in dataset:
            m, loss = train_step(m, list(inp.vals), tgt, eta)
            total_loss += loss
        if epoch % print_every == 0:
            print(f"  Epoch {epoch}  loss: {total_loss:.6f}")
    return m


# ---------------------------------------------------------------------------
# I.11  Inference
# ---------------------------------------------------------------------------

def predict(m: Model, xs: list[float]) -> float:
    """Run a forward pass and return the scalar prediction ŷ.

    Parameters
    ----------
    m : Model
        The (trained) model.
    xs : list[float]
        Input feature vector.

    Returns
    -------
    float
        The network's scalar output ŷ ∈ (0, 1).
    """
    pred, _, _ = network_forward(m, xs)
    return pred.v


# =============================================================================
# SECTION II - PROBABILITY-BASED LEARNING: MARKOV CATEGORIES
#              (survey §3 - stochastic morphisms, dropout, Bayesian sketch)
# =============================================================================
#
# A Markov category is a symmetric monoidal category where every object X
# carries a commutative comonoid structure:
#   copy  : X → X ⊗ X    (diagonal / duplication)
#   delete: X → I         (marginalisation / discarding)
#
# Morphisms are "stochastic kernels" - they map objects to probability
# distributions over objects.  In practice we represent a stochastic morphism
# as a function that, given an input, *samples* an output.
#
# Two instances we implement:
#   A) Dropout   - a stochastic lens (forward stochasticity)
#   B) Bayesian  - weight-space priors / posterior sampling sketch


# ---------------------------------------------------------------------------
# II.1  Stochastic morphism composition (Kleisli composition)
# ---------------------------------------------------------------------------

def stochastic_compose(
    f: Callable[[list[float]], list[float]],
    g: Callable[[list[float]], list[float]],
) -> Callable[[list[float]], list[float]]:
    """Kleisli composition of two stochastic morphisms: (f ∘_s g)(x) = f(g(x))."""
    return lambda x: f(g(x))


# ---------------------------------------------------------------------------
# II.2  Dropout as a stochastic lens
# ---------------------------------------------------------------------------
#
# Dropout (Srivastava 2014) can be formalised as a stochastic lens (survey §3.2):
#   forward  : X →_s X ⊗ Mask    where Mask ~ Bernoulli(p)^n
#   backward : X ⊗ Mask → X      (apply the same mask to gradient)
#
# The stochastic forward pass samples a mask once; the backward pass reuses it
# (this is the "closed" lens / optic requirement that both passes share state).

def make_dropout_lens(
    keep_prob: float,
) -> Callable[[list[float]], tuple[list[float], Callable[[list[float]], list[float]]]]:
    """Create a stochastic dropout lens.

    Returns a callable that, given an input vector, samples a Bernoulli mask,
    applies inverted dropout, and returns (masked_output, pullback_fn).

    The same mask is reused in the pullback (the "closed" stochastic lens
    requirement from survey §3.2).

    Parameters
    ----------
    keep_prob : float
        Probability of keeping each neuron active (p ∈ (0, 1]).

    Returns
    -------
    Callable
        A stochastic Para morphism: inputs → (masked_output, pullback_fn).
    """
    def dropout_forward(
        inputs: list[float],
    ) -> tuple[list[float], Callable[[list[float]], list[float]]]:
        # Sample binary mask from Bernoulli(keep_prob)
        mask = [1.0 if random.random() < keep_prob else 0.0 for _ in inputs]
        # Apply mask and scale (inverted dropout - keeps expectation constant)
        scale = 1.0 / keep_prob
        masked = [x * m * scale for x, m in zip(inputs, mask)]

        # The pullback reuses the same mask (stochastic lens requirement)
        def pullback(upstream: list[float]) -> list[float]:
            return [u * m * scale for u, m in zip(upstream, mask)]

        return masked, pullback

    return dropout_forward


# ---------------------------------------------------------------------------
# II.3  Bayesian weight uncertainty (sketch)
# ---------------------------------------------------------------------------
#
# In Bayesian deep learning (survey §3.3), weights are random variables.
# A Bayesian layer samples weights from a distribution W ~ N(μ, σ²) at each
# forward pass - this is a stochastic morphism in the Markov category Stoch.
#
# We implement a single-layer Bayesian linear model with Gaussian weights.

def gaussian_sample(mu: float, sigma: float) -> float:
    """Sample from N(μ, σ) using the Box-Muller transform."""
    import math as _math
    u1 = max(random.random(), 1e-10)
    u2 = random.random()
    z = _math.sqrt(-2.0 * _math.log(u1)) * _math.cos(2.0 * _math.pi * u2)
    return mu + sigma * z


@dataclass
class BayesianLayer:
    """Parameters of a Bayesian linear layer.

    Weights are treated as random variables W ~ N(μ, σ²I).

    Attributes
    ----------
    mu : list[list[float]]
        Mean weight matrix (fan_out × fan_in).
    sigma : float
        Standard deviation shared across all weights (could be learned).
    fan_in : int
    fan_out : int
    """
    mu: list[list[float]]
    sigma: float
    fan_in: int
    fan_out: int


def make_bayesian_layer(fan_in: int, fan_out: int, sigma: float = 0.1) -> BayesianLayer:
    """Create a Bayesian linear layer with Glorot mean and fixed sigma.

    Parameters
    ----------
    fan_in : int
        Number of input features.
    fan_out : int
        Number of output features.
    sigma : float
        Standard deviation of weight prior (default 0.1).

    Returns
    -------
    BayesianLayer
    """
    mu = [
        [glorot_rand(fan_in, fan_out) for _ in range(fan_in)]
        for _ in range(fan_out)
    ]
    return BayesianLayer(mu=mu, sigma=sigma, fan_in=fan_in, fan_out=fan_out)


def bayesian_forward(bl: BayesianLayer, inputs: list[float]) -> list[float]:
    """Stochastic forward pass: sample W ~ N(μ, σI) and apply.

    This is a stochastic morphism in the Markov category Stoch:
    each call samples a different weight matrix.

    Parameters
    ----------
    bl : BayesianLayer
        Bayesian layer parameters (μ, σ).
    inputs : list[float]
        Input activation vector.

    Returns
    -------
    list[float]
        Output activations: sigmoid(W_s · x) for sampled W_s.
    """
    W_sample = [
        [gaussian_sample(w, bl.sigma) for w in row]
        for row in bl.mu
    ]
    bias = [0.0] * bl.fan_out
    return [sigmoid(z) for z in vec_add(matvec(W_sample, inputs), bias)]


def bayesian_predict_mc(
    bl: BayesianLayer,
    inputs: list[float],
    n_samples: int,
) -> tuple[float, float]:
    """Uncertainty estimation via Monte Carlo sampling (survey §3.3).

    Performs multiple stochastic forward passes and returns the
    empirical mean and standard deviation of the output.

    Parameters
    ----------
    bl : BayesianLayer
        Bayesian layer.
    inputs : list[float]
        Input vector.
    n_samples : int
        Number of Monte Carlo forward-pass samples.

    Returns
    -------
    mean : float
        Empirical mean prediction.
    std : float
        Empirical standard deviation (epistemic uncertainty estimate).
    """
    samples = [bayesian_forward(bl, inputs)[0] for _ in range(n_samples)]
    mean = sum(samples) / n_samples
    variance = sum((s - mean) ** 2 for s in samples) / n_samples
    return mean, math.sqrt(variance)


# =============================================================================
# SECTION III - INVARIANCE / EQUIVARIANCE
#               (survey §4 - functors that respect symmetry structure)
# =============================================================================
#
# A layer  f : X → Y  is *equivariant* with respect to a group G if:
#   f(g · x) = g · f(x)    for all g ∈ G
#
# In categorical language: there exists a functor  F : BG → Vect  such that
# f is a natural transformation between F and another G-representation.
#
# We demonstrate permutation-equivariance (the simplest case), which is the
# categorical basis of set-valued neural networks (DeepSets, §4.1 of survey).


def permutation_invariant_pool(xs: list[float]) -> float:
    """Sum-pool over a set - invariant to any permutation.

    Categorical reading: this is the colimit (coproduct) over the set diagram
    Σᵢ xᵢ - invariant to any permutation because addition is commutative.

    This is the basis of the DeepSets architecture (Zaheer et al. 2017).
    """
    return sum(xs)


def permutation_equivariant_map(
    f: Callable[[float], float],
    xs: list[float],
) -> list[float]:
    """Apply f elementwise - equivariant to any permutation.

    f ∘ π = π ∘ f for any permutation π, by construction.
    """
    return [f(x) for x in xs]


# ---------------------------------------------------------------------------
# III.B  Categorical clustering (colimit interpretation)
# ---------------------------------------------------------------------------
#
# K-means can be viewed as computing colimits in a category of metric spaces
# (survey §4.2): the cluster centre is the colimit / limit of points in the
# cluster, and assignment is the universal morphism.

def euclidean_dist(u: list[float], v: list[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def nearest_centroid(point: list[float], centroids: list[list[float]]) -> int:
    """Return the index of the nearest centroid to *point*."""
    dists = [euclidean_dist(point, c) for c in centroids]
    return dists.index(min(dists))


def update_centroids(
    data: list[list[float]],
    labels: list[int],
    k: int,
) -> list[list[float]]:
    """Recompute centroids as the average (colimit) of each cluster.

    Parameters
    ----------
    data : list[list[float]]
        Data points.
    labels : list[int]
        Cluster assignment for each point.
    k : int
        Number of clusters.

    Returns
    -------
    list[list[float]]
        New centroid positions (one per cluster).
    """
    new_centroids = []
    for c in range(k):
        cluster = [data[i] for i, lbl in enumerate(labels) if lbl == c]
        if not cluster:
            # Degenerate cluster: keep at origin
            new_centroids.append([0.0] * len(data[0]))
        else:
            n = len(cluster)
            # Centroid = colimit (average) of the cluster
            centroid = [
                sum(pt[j] for pt in cluster) / n
                for j in range(len(cluster[0]))
            ]
            new_centroids.append(centroid)
    return new_centroids


def k_means(
    data: list[list[float]],
    k: int,
    max_iter: int,
) -> tuple[list[list[float]], list[int]]:
    """K-means clustering viewed as iterative colimit computation.

    Parameters
    ----------
    data : list[list[float]]
        Data points to cluster.
    k : int
        Number of clusters.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    centroids : list[list[float]]
        Final centroid positions (colimits of each cluster).
    labels : list[int]
        Cluster assignment for each data point.
    """
    # Initialise centroids by random selection
    centroids = random.sample(data, k)

    for _ in range(max_iter):
        labels = [nearest_centroid(p, centroids) for p in data]
        new_centroids = update_centroids(data, labels, k)
        if new_centroids == centroids:
            break
        centroids = new_centroids

    labels = [nearest_centroid(p, centroids) for p in data]
    return centroids, labels


# =============================================================================
# SECTION IV - TOPOS-BASED LEARNING
#              (survey §5 - subobject classifiers, sheaves, internal logic)
# =============================================================================
#
# A topos E is a category that has:
#   • finite limits and colimits
#   • a subobject classifier Ω  with a "true" morphism  ⊤ : 1 → Ω
#
# Every subobject (subset) S ⊆ X is classified by a unique morphism
#   χ_S : X → Ω   such that S = χ_S⁻¹(⊤)
#
# In machine learning (survey §5.1), the output neuron of a binary classifier
# IS the characteristic morphism χ : InputSpace → {0,1}.  Sigmoid squashes
# the activation into [0,1] ≅ Ω, and the decision boundary is χ⁻¹(0.5).
#
# Sheaf composition (survey §5.2): local predictions on overlapping contexts
# are "glued" into a global consistent assignment - exactly the sheaf
# condition.  We implement a simple ensemble that enforces global consistency.


# ---------------------------------------------------------------------------
# IV.1  Subobject classifier - binary decision morphism
# ---------------------------------------------------------------------------
#
# χ : InputSpace → Ω ≅ [0,1]   (the sigmoid output IS the characteristic map)

def subobject_classify(m: Model, xs: list[float]) -> tuple[float, int]:
    """Apply the subobject classifier χ : X → Ω.

    The prediction probability is the value of the characteristic morphism χ.
    The class decision is χ⁻¹(0.5) - the decision boundary (survey §5.1).

    Returns
    -------
    prob : float
        The probability value χ(x) ∈ [0, 1].
    cls : int
        Binary class label (0 or 1).
    """
    prob = predict(m, xs)
    cls = 1 if prob >= 0.5 else 0
    return prob, cls


# ---------------------------------------------------------------------------
# IV.2  Sheaf composition - local-to-global consistency
# ---------------------------------------------------------------------------
#
# A sheaf F on a space X assigns to each open set U ⊆ X a set of "sections"
# F(U), with restriction maps that are compatible on overlaps.
#
# For ML: each "expert" (sub-model) covers a local context (a feature subset).
# The sheaf condition requires that predictions on overlapping contexts agree.
#
# We implement a two-expert ensemble with a consistency check ("gluing lemma").

@dataclass(frozen=True)
class SheafSection:
    """A local section F(U) of a sheaf - one expert's prediction on its context.

    Attributes
    ----------
    context : str
        Identifier for the local open set / feature subset.
    prediction : float
        The expert's scalar prediction on this context.
    """
    context: str
    prediction: float


def sheaf_consistent(s1: SheafSection, s2: SheafSection, tol: float) -> bool:
    """Check whether two sections are compatible (agree within tolerance ε).

    The sheaf condition requires that sections on overlapping open sets
    restrict to the same value on the overlap (survey §5.2).
    """
    return abs(s1.prediction - s2.prediction) < tol


def sheaf_glue(sections: list[SheafSection], tol: float) -> Optional[float]:
    """Attempt to glue local sections into a global section (survey §5.2).

    Returns the average prediction if all sections are pairwise consistent,
    or None if the sheaf condition fails (inconsistent experts).

    Parameters
    ----------
    sections : list[SheafSection]
        Local expert predictions.
    tol : float
        Consistency tolerance ε.

    Returns
    -------
    float or None
        Global prediction (average) if consistent, else None.
    """
    # Check all pairwise section consistencies (the gluing lemma)
    for i in range(len(sections)):
        for j in range(i + 1, len(sections)):
            if not sheaf_consistent(sections[i], sections[j], tol):
                return None  # sheaf condition fails

    # Global section: average of local predictions
    return sum(s.prediction for s in sections) / len(sections)


# ---------------------------------------------------------------------------
# IV.3  Internal logic - the Heyting algebra of propositions
# ---------------------------------------------------------------------------
#
# The subobject classifier Ω in a topos carries a Heyting algebra structure
# (intuitionistic logic).  In a Boolean topos (Set), this collapses to
# classical logic.  We represent propositions as probability thresholds.

def heyting_and(p: float, q: float) -> float:
    """Conjunction: p ∧ q = min(p, q)."""
    return min(p, q)


def heyting_or(p: float, q: float) -> float:
    """Disjunction: p ∨ q = max(p, q)."""
    return max(p, q)


def heyting_not(p: float) -> float:
    """Pseudo-complement: ¬p = 1 − p  (collapses to Boolean NOT on {0, 1})."""
    return 1.0 - p


def heyting_implies(p: float, q: float) -> float:
    """Implication: p ⇒ q = ¬p ∨ q."""
    return heyting_or(heyting_not(p), q)


# =============================================================================
# SECTION V - FUNCTOR COMPOSITION DEMO
#             Natural transformation between two trained models
# =============================================================================
#
# A natural transformation  η : F ⇒ G  between two functors witnesses that
# G's predictions can be derived from F's predictions in a coherent, functorial
# way.  We implement a simple "knowledge distillation" adapter - a linear map
# from F-outputs to G-outputs - as a concrete natural transformation.

@dataclass
class NatTransform:
    """Components of a natural transformation η : F ⇒ G.

    Represented as a linear adapter::

        η_X : F(X) → G(X)   with   η_X(v) = sigmoid(W · v + b)

    Attributes
    ----------
    adapter_W : list[list[float]]
        Weight matrix of the linear adapter (target_size × source_size).
    adapter_b : list[float]
        Bias vector of length target_size.
    """
    adapter_W: list[list[float]]
    adapter_b: list[float]


def make_nat_transform(source_size: int, target_size: int) -> NatTransform:
    """Create a randomly initialised natural transformation adapter.

    Parameters
    ----------
    source_size : int
        Dimension of the teacher (source functor) representation.
    target_size : int
        Dimension of the student (target functor) representation.

    Returns
    -------
    NatTransform
        A Glorot-initialised linear adapter.
    """
    W = [
        [glorot_rand(source_size, target_size) for _ in range(source_size)]
        for _ in range(target_size)
    ]
    b = [0.0] * target_size
    return NatTransform(adapter_W=W, adapter_b=b)


def apply_nat_transform(nt: NatTransform, v: list[float]) -> list[float]:
    """Apply the natural transformation η_X to a vector v.

    Computes: sigmoid(W · v + b), mapping teacher rep → student rep.
    """
    return [sigmoid(z) for z in vec_add(matvec(nt.adapter_W, v), nt.adapter_b)]


# =============================================================================
# DEMO - runs all five perspectives with concrete examples
# =============================================================================

if __name__ == "__main__":
    random.seed(42)

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   Category-Theory Deep Learning Framework (Python)              ║")
    print("║   Reference: Jia et al. (2025) Axioms 14(3):204                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    #  DEMO I: Para category - compositional backprop on XOR
    # ──────────────────────────────────────────────────────────────────────────
    print("━━━━  I. Para Category + Lens Composition  (XOR problem)  ━━━━")
    print("  Architecture: input(2) → hidden(4) → hidden(4) → output(1)")
    print("  Para morphisms composed sequentially; pullbacks in reverse.")
    print()

    xor_data: list[tuple[InputVec, TargetVal]] = [
        (InputVec(vals=(0.0, 0.0)), TargetVal(v=0.0)),
        (InputVec(vals=(0.0, 1.0)), TargetVal(v=1.0)),
        (InputVec(vals=(1.0, 0.0)), TargetVal(v=1.0)),
        (InputVec(vals=(1.0, 1.0)), TargetVal(v=0.0)),
    ]

    # Architecture: input(2) → hidden1(4) → hidden2(4) → output(1)
    xor_model = make_network([(2, 4), (4, 4), (4, 1)])

    trained_xor = train(xor_model, xor_data, eta=0.5, epochs=6000, print_every=2000)

    print("  → Predictions after 6000 epochs:")
    for inp, tgt in xor_data:
        xs = list(inp.vals)
        y_hat = predict(trained_xor, xs)
        cls = 1 if y_hat > 0.5 else 0
        print(f"      Input: {xs}  Target: {tgt.v:.1f}  Pred: {y_hat:.4f}  Class: {cls}")

    print()
    print("  Category-theory reading:")
    print("  • Each layer is a morphism in Para(Euc): f : P × X → Y")
    print("  • forward_para returns (output, pullback_closure) - the lens")
    print("  • Backprop = pullback composition  f* ∘ g* ∘ h*  in reverse")
    print("  • SGD update = endomorphism u_η : Model → Model")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    #  DEMO II: Markov Categories - Dropout + Bayesian uncertainty
    # ──────────────────────────────────────────────────────────────────────────
    print("━━━━  II. Markov Categories - Stochastic Morphisms  ━━━━")
    print()

    # II-A: Dropout as stochastic lens
    print("  II-A  Dropout (stochastic lens, Bernoulli(0.7) mask)")
    dropout_lens = make_dropout_lens(0.7)
    test_vec = [1.0, 2.0, 3.0, 4.0, 5.0]
    masked_vec, pb_fn = dropout_lens(test_vec)
    print(f"    Input:       {test_vec}")
    print(f"    After drop:  {[f'{x:.3f}' for x in masked_vec]}")
    masked_grad = pb_fn([1.0, 1.0, 1.0, 1.0, 1.0])
    print(f"    Grad (back): {[f'{x:.3f}' for x in masked_grad]}")
    print()
    print("    The same mask is reused in both passes - the 'closed'")
    print("    stochastic lens condition (survey §3.2).")
    print()

    # II-B: Bayesian uncertainty
    print("  II-B  Bayesian layer - Monte Carlo uncertainty estimation")
    bayes_layer = make_bayesian_layer(3, 1, sigma=0.3)
    bayes_input = [1.0, 0.5, -0.5]
    mu_est, sigma_est = bayesian_predict_mc(bayes_layer, bayes_input, n_samples=200)
    print(f"    Input:  {bayes_input}")
    print(f"    MC mean (200 samples):  {mu_est:.4f}")
    print(f"    MC std  (uncertainty):  {sigma_est:.4f}")
    print()
    print("    W ~ N(μ,σ) at each forward pass = stochastic morphism in Stoch.")
    print("    Variance encodes epistemic uncertainty (survey §3.3).")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    #  DEMO III: Invariance / Equivariance - DeepSets + K-means
    # ──────────────────────────────────────────────────────────────────────────
    print("━━━━  III. Invariance & Equivariance  ━━━━")
    print()

    print("  III-A  Permutation-Invariant Pooling (colimit / coproduct)")
    set1 = [1.0, 3.0, 5.0, 2.0]
    set2 = [3.0, 1.0, 2.0, 5.0]   # permutation of set1
    print(f"    Set 1 pool: {permutation_invariant_pool(set1)}  (order: {set1})")
    print(f"    Set 2 pool: {permutation_invariant_pool(set2)}  (order: {set2})")
    print("    → Same result for both permutations ✓")
    print()

    print("  III-B  K-means as categorical colimit computation")
    cluster_data = [
        [1.0, 1.0], [1.2, 0.8], [0.9, 1.1],
        [5.0, 5.0], [5.1, 4.9], [4.8, 5.2],
        [1.0, 5.0], [0.9, 4.8], [1.1, 5.1],
    ]
    centroids, labels = k_means(cluster_data, k=3, max_iter=50)
    print(f"    Data: {len(cluster_data)} points in 2D")
    print("    Centroids (colimits of each cluster):")
    for i, c in enumerate(centroids):
        print(f"      Cluster {i}: ({c[0]:.2f}, {c[1]:.2f})")
    print(f"    Labels: {labels}")
    print()
    print("    Each centroid = colimit of its cluster (survey §4.2).")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    #  DEMO IV: Topos Framework - subobject classifier + sheaf gluing
    # ──────────────────────────────────────────────────────────────────────────
    print("━━━━  IV. Topos Framework  ━━━━")
    print()

    # Use the trained XOR model as our binary classifier
    print("  IV-A  Subobject Classifier χ : X → Ω")
    print("        (sigmoid output = characteristic morphism)")
    for inp, _ in xor_data:
        xs = list(inp.vals)
        prob, cls = subobject_classify(trained_xor, xs)
        print(f"    χ({xs}) = {prob:.4f}  → class {cls}")
    print()
    print("    The sigmoid IS the characteristic morphism of the classifier's")
    print("    subobject S ⊆ InputSpace (survey §5.1).")
    print()

    # Sheaf gluing
    print("  IV-B  Sheaf Gluing - local-to-global consistency")
    s1 = SheafSection(context="feature-subset-A", prediction=0.72)
    s2 = SheafSection(context="feature-subset-B", prediction=0.68)
    s3 = SheafSection(context="feature-subset-C", prediction=0.91)  # inconsistent
    glued_12 = sheaf_glue([s1, s2], tol=0.1)
    glued_13 = sheaf_glue([s1, s3], tol=0.1)
    print(f"    Expert A pred: {s1.prediction},  Expert B pred: {s2.prediction}")
    glue_str = f"{glued_12:.4f}" if glued_12 is not None else "INCONSISTENT - cannot glue"
    print(f"    Glue A+B (tol=0.1): {glue_str}")
    print(f"    Expert C pred: {s3.prediction} (conflict with A)")
    glue_str2 = f"{glued_13:.4f}" if glued_13 is not None else "INCONSISTENT - cannot glue"
    print(f"    Glue A+C (tol=0.1): {glue_str2}")
    print()
    print("    Consistent sections glue to a global prediction; inconsistent")
    print("    sections reveal model disagreement (survey §5.2).")
    print()

    # Heyting algebra
    print("  IV-C  Internal Logic - Heyting Algebra on Ω = [0,1]")
    p1, p2 = 0.8, 0.3
    print(f"    p='high confidence'={p1},  q='low confidence'={p2}")
    print(f"    p ∧ q  = min(p,q)       = {heyting_and(p1, p2)}")
    print(f"    p ∨ q  = max(p,q)       = {heyting_or(p1, p2)}")
    print(f"    ¬p     = 1-p            = {heyting_not(p1)}")
    print(f"    p ⇒ q  = ¬p ∨ q        = {heyting_implies(p1, p2)}")
    print()
    print("    Ω carries Heyting algebra structure - the internal logic of")
    print("    the topos (Boolean collapse in Set, survey §5.1).")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    #  DEMO V: Natural Transformation - knowledge distillation adapter
    # ──────────────────────────────────────────────────────────────────────────
    print("━━━━  V. Natural Transformation - Knowledge Distillation Adapter  ━━━━")
    print()
    print("  A natural transformation η : F ⇒ G is a coherent family of")
    print("  morphisms η_X : F(X) → G(X) that commutes with all arrows.")
    print("  Here we use a linear adapter as η, mapping the 4-dim penultimate")
    print("  hidden layer of the teacher to a 2-dim student representation.")
    print()

    # Retrieve the 4-dim hidden representation from layers 0 and 1 (penultimate)
    # by running a partial forward pass.
    layers = trained_xor.layers
    l1, l2 = layers[0], layers[1]
    a1, _pb1 = forward_para(l1, sigmoid, lambda z: sigmoid_deriv(sigmoid(z)), [1.0, 0.0])
    a2, _pb2 = forward_para(l2, sigmoid, lambda z: sigmoid_deriv(sigmoid(z)), a1)
    teacher_hidden = a2   # 4-dim hidden representation

    adapter = make_nat_transform(source_size=4, target_size=2)
    student_rep = apply_nat_transform(adapter, teacher_hidden)
    print(f"    Teacher hidden (4-dim): {[f'{x:.4f}' for x in teacher_hidden]}")
    print(f"    Student rep   (2-dim):  {[f'{x:.4f}' for x in student_rep]}")
    print()
    print("    The adapter is the natural transformation η_X that lets the")
    print("    student functor G access the teacher functor F's information.")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    #  Summary
    # ──────────────────────────────────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Framework Summary                                              ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  I.   Para category    Layers as Para morphisms;               ║")
    print("║       + Lens/Optics    pullbacks compose backprop              ║")
    print("║  II.  Markov cats      Dropout & Bayesian layers as            ║")
    print("║                        stochastic morphisms in Stoch           ║")
    print("║  III. Invariance       Permutation-invariant pooling =         ║")
    print("║                        colimit; K-means = colimit of clusters  ║")
    print("║  IV.  Topos            Sigmoid = subobject classifier χ;       ║")
    print("║                        ensemble gluing = sheaf condition       ║")
    print("║  V.   Nat. transform   Knowledge distillation = η : F ⇒ G     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
