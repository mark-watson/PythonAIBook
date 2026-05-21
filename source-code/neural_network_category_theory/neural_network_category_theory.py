"""
Category-Theory-Inspired 2-Hidden-Layer Neural Network in Python
=================================================================

Reference: https://rust-ml.com/materials/one-training-step-end-to-end.html

Category-theory lens: every layer is a morphism that carries BOTH
  • a forward output  (the "get")
  • a backward context (the "put-back" / pullback)

The four training-step stages, viewed through two lenses:

  Stage         | Algebra lens           | Category theory lens
  --------------|------------------------|-----------------------------
  Forward       | z = Wx + b; σ(z)       | ParameterSpace × InputSpace
                |                        |   → Prediction × Context
  Loss          | L = (ŷ - y)²           | Prediction × Target
                |                        |   → SquaredError
  Backward      | ∇w = upstream · x      | Context × ∇ŷ → ∇P × ∇X
                | chain rule all layers  | (pullback / covariant functor)
  Update        | w := w - η · ∇w        | ModelState → ModelState
                |                        | (endomorphism on parameter space)

Architecture:  input(2) → hidden1(3) → hidden2(3) → output(1)
Activation:    sigmoid on every layer
Loss:          mean-squared error
Optimiser:     vanilla SGD

Usage::

    uv run neural_network_category_theory.py

"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable


# =============================================================================
# 1.  TYPED WRAPPERS  (newtypes that encode categorical roles)
# =============================================================================
# We use dataclasses so that Prediction, Target, LearningRate etc. cannot be
# accidentally confused — the type carries the semantic role of the value.
# In Racket these were plain structs; Python dataclasses give the same
# transparency (repr) and lightweight construction.


@dataclass(frozen=True)
class InputVec:
    """An element of the InputSpace — carries the raw feature vector.

    Categorical role: object in the InputSpace category.
    """
    vals: tuple[float, ...]


@dataclass(frozen=True)
class TargetVal:
    """A supervised label — the target scalar y.

    Categorical role: element of the Target object.
    """
    v: float


@dataclass(frozen=True)
class Prediction:
    """The network's scalar output ŷ after the forward pass.

    Categorical role: element of the Prediction object — the codomain of the
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
        Weight matrix stored as a list of row vectors.
        Shape: (fan_out, fan_in).
    b : list[float]
        Bias vector of length fan_out.

    Categorical role: an object in the ParameterSpace category.
    """
    W: list[list[float]]
    b: list[float]


@dataclass
class LayerGrads:
    """Gradients with respect to one layer's parameters.

    Shape mirrors LayerParams — both live in the same tangent space.

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
    """The full network — a product of three LayerParams.

    Categorical role: an object in the ModelState category (the product
    ParameterSpace₁ × ParameterSpace₂ × ParameterSpace₃).

    Attributes
    ----------
    l1 : LayerParams  — input(2) → hidden1(3)
    l2 : LayerParams  — hidden1(3) → hidden2(3)
    l3 : LayerParams  — hidden2(3) → output(1)
    """
    l1: LayerParams
    l2: LayerParams
    l3: LayerParams


# =============================================================================
# 2.  UTILITY MATH
# =============================================================================

def sigmoid(z: float) -> float:
    """The logistic sigmoid activation function σ(z) = 1 / (1 + e^{-z})."""
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_deriv(sig_val: float) -> float:
    """Derivative of sigmoid given its *output* value: σ'(z) = σ(z)(1 − σ(z)).

    Accepts the already-computed sigmoid value to avoid recomputation.
    """
    return sig_val * (1.0 - sig_val)


def dot(ws: list[float], xs: list[float]) -> float:
    """Standard vector dot product Σ wᵢ · xᵢ."""
    return sum(w * x for w, x in zip(ws, xs))


def vec_mul(xs: list[float], ys: list[float]) -> list[float]:
    """Hadamard (elementwise) product of two equal-length vectors."""
    return [x * y for x, y in zip(xs, ys)]


def scalar_vec_mul(s: float, xs: list[float]) -> list[float]:
    """Multiply every element of vector *xs* by scalar *s*."""
    return [s * x for x in xs]


# =============================================================================
# 3.  LAYER LENS
# =============================================================================
#
# A Lens morphism for one fully-connected sigmoid layer:
#
#   forward_lens_tracked : LayerParams × input → (activations, pullback_fn)
#
# The pullback_fn closes over the forward-pass state (W, inputs, zs, acts)
# and implements:
#
#   pullback : upstream_grad → (LayerGrads, input_grad)
#
# Category-theory reading:
#   The layer is an arrow  f : P × X → Y × Ctx
#   The pullback  f* : Ctx × ∇Y → ∇P × ∇X  is captured in the closure.
#   Returning (output, closure) is exactly the Lens pattern:
#       get  = forward evaluation
#       put  = pullback / backward pass

# Type alias for the pullback closure.
PullbackFn = Callable[[list[float]], tuple[LayerGrads, list[float]]]


def forward_lens_tracked(
    params: LayerParams,
    inputs: list[float],
) -> tuple[list[float], PullbackFn]:
    """Lens morphism for a single fully-connected sigmoid layer.

    Performs the forward pass and returns a pullback closure that can later
    compute the parameter and input gradients given an upstream gradient.

    Category-theory reading
    -----------------------
    This function realises the arrow::

        f : P × X → Y × Ctx

    where
      • P     = LayerParams (W, b)
      • X     = input activations from the previous layer
      • Y     = output activations of this layer
      • Ctx   = the pullback closure (captures W, inputs, acts)

    The returned pullback closure implements::

        f* : ∇Y → ∇P × ∇X

    Parameters
    ----------
    params : LayerParams
        Weight matrix W and bias vector b for this layer.
    inputs : list[float]
        Activation vector arriving from the previous layer (or raw input).

    Returns
    -------
    acts : list[float]
        Post-activation output of this layer: aᵢ = σ(Wᵢ · x + bᵢ).
    pullback : PullbackFn
        Closure  ``upstream_grad → (LayerGrads, input_grad)``  that computes
        gradients during the backward pass.
    """
    W = params.W
    b = params.b
    # Pre-activations: zᵢ = dot(Wᵢ, inputs) + bᵢ
    zs = [dot(wi, inputs) + bi for wi, bi in zip(W, b)]
    # Post-activations: aᵢ = σ(zᵢ)
    acts = [sigmoid(z) for z in zs]

    def pullback(upstream_grad: list[float]) -> tuple[LayerGrads, list[float]]:
        """Pullback (backward) morphism for this layer.

        Implements the chain rule for a sigmoid fully-connected layer.

        Category-theory reading
        -----------------------
        This is the "put-back" of the Lens::

            f* (Ctx, ∇Y) = (∇P, ∇X)

        where ∇P = (dW, db) and ∇X is propagated to the previous layer.

        Parameters
        ----------
        upstream_grad : list[float]
            Gradient of the loss w.r.t. the output activations of THIS layer,
            i.e. dL/daᵢ.

        Returns
        -------
        grads : LayerGrads
            Gradients w.r.t. W and b.
        dX : list[float]
            Gradient to propagate to the previous layer (dL/dxⱼ).
        """
        # δᵢ = upstream_gradᵢ · σ'(aᵢ)   — local error signal
        delta = [u * sigmoid_deriv(a) for u, a in zip(upstream_grad, acts)]

        # ∇Wᵢⱼ = δᵢ · xⱼ   (outer product δ ⊗ inputs)
        dW = [[d * x for x in inputs] for d in delta]

        # ∇bᵢ = δᵢ
        db = list(delta)

        # ∇xⱼ = Σᵢ δᵢ · Wᵢⱼ   (Wᵀ · δ, transpose-multiply)
        n_inputs = len(inputs)
        dX = [
            sum(delta[i] * W[i][j] for i in range(len(delta)))
            for j in range(n_inputs)
        ]

        return LayerGrads(dW=dW, db=db), dX

    return acts, pullback


# =============================================================================
# 4.  FULL NETWORK — FORWARD PASS  (lens composition)
# =============================================================================
#
# The network is the sequential composition of three lenses:
#
#   model_forward : Model × InputVec → (Prediction × NetworkContext)
#
# NetworkContext is a triple of pullback closures (one per layer) stored
# in the order they were produced; the backward pass traverses them in
# reverse.  This is the categorical composition of lenses:
#
#   (f ⊚ g ⊚ h) where ⊚ denotes sequential lens composition.


@dataclass
class NetworkContext:
    """Product of the three per-layer pullback closures from the forward pass.

    Categorical role: the "residual" of the composed lens — everything needed
    to run the pullbacks in reverse order during backprop.

    Attributes
    ----------
    pb1, pb2, pb3 : PullbackFn
        Pullback closures for layers 1, 2, 3 respectively.
    final_act : float
        The scalar output of the network ŷ (cached for convenience).
    """
    pb1: PullbackFn
    pb2: PullbackFn
    pb3: PullbackFn
    final_act: float


def model_forward(m: Model, xs: list[float]) -> tuple[Prediction, NetworkContext]:
    """Compose three lens morphisms to produce the network's prediction.

    Category-theory reading
    -----------------------
    This is the sequential composition of three lenses::

        F = f₃ ⊚ f₂ ⊚ f₁  :  (P₁ × P₂ × P₃) × X → Ŷ × (Ctx₁ × Ctx₂ × Ctx₃)

    Each fᵢ = ``forward_lens_tracked(lᵢ, ·)`` and the contexts are the
    pullback closures stored in :class:`NetworkContext`.

    Parameters
    ----------
    m : Model
        The current network weights (P₁, P₂, P₃).
    xs : list[float]
        Raw input features.

    Returns
    -------
    pred : Prediction
        The scalar network output ŷ.
    ctx : NetworkContext
        The three pullback closures needed for backprop.
    """
    a1, pb1 = forward_lens_tracked(m.l1, xs)
    a2, pb2 = forward_lens_tracked(m.l2, a1)
    a3, pb3 = forward_lens_tracked(m.l3, a2)
    # Output layer has one neuron → a3 is a list of one element.
    y_hat = a3[0]
    return Prediction(v=y_hat), NetworkContext(pb1=pb1, pb2=pb2, pb3=pb3, final_act=y_hat)


# =============================================================================
# 5.  LOSS  (typed morphism: Prediction × Target → SquaredError)
# =============================================================================

def mse_loss(pred: Prediction, tgt: TargetVal) -> float:
    """Mean-squared error loss (for one sample): L = (ŷ − y)².

    Categorical role: a morphism  Prediction × Target → ℝ  (a scalar).
    """
    diff = pred.v - tgt.v
    return diff * diff


def mse_loss_grad(pred: Prediction, tgt: TargetVal) -> float:
    """Gradient of MSE loss w.r.t. the prediction: dL/dŷ = 2(ŷ − y).

    This is the "seed" upstream gradient that starts the backward pass.
    """
    return 2.0 * (pred.v - tgt.v)


# =============================================================================
# 6.  BACKWARD PASS  (pullback composition in reverse)
# =============================================================================
#
# Category-theory reading:
#   The backward pass is the composition of the three pullback morphisms
#   in *reverse* order, threading the upstream gradient through each:
#
#     f₁* ∘ f₂* ∘ f₃*  :  ∇Ŷ → ∇P₃ × ∇P₂ × ∇P₁ × ∇X
#
# Each pullback is the closure returned by forward_lens_tracked.

def model_backward(
    ctx: NetworkContext,
    dl_dy_hat: float,
) -> tuple[LayerGrads, LayerGrads, LayerGrads]:
    """Run the backward pass by composing pullback morphisms in reverse order.

    Category-theory reading
    -----------------------
    Implements::

        f₁* ∘ f₂* ∘ f₃*  :  ∇Ŷ → (∇P₁, ∇P₂, ∇P₃, ∇X)

    where each fᵢ* is the pullback closure stored in ``ctx``.

    Parameters
    ----------
    ctx : NetworkContext
        The three pullback closures produced by :func:`model_forward`.
    dl_dy_hat : float
        dL/dŷ — the seed gradient from the loss function.

    Returns
    -------
    grads1, grads2, grads3 : LayerGrads
        Parameter gradients for layers 1, 2, and 3.
    """
    # Layer 3 pullback — upstream gradient is dL/da₃ = dL/dŷ (a list of 1)
    grads3, dX3 = ctx.pb3([dl_dy_hat])
    # Layer 2 pullback — upstream gradient arrives from layer 3
    grads2, dX2 = ctx.pb2(dX3)
    # Layer 1 pullback — upstream gradient arrives from layer 2
    grads1, _dX1 = ctx.pb1(dX2)
    return grads1, grads2, grads3


# =============================================================================
# 7.  UPDATE STEP  (endomorphism on ModelState)
# =============================================================================
#
# Category-theory reading:
#   SGD update is an endomorphism  u_η : Model → Model.
#   It is *closed* on the parameter space — it maps a model to a new model
#   of the same type.  Training is iterated application of this endomorphism.
#
#   u_η(θ) = θ − η · ∇θ

def update_layer(params: LayerParams, grads: LayerGrads, lr: LearningRate) -> LayerParams:
    """Apply one SGD step to a single layer's parameters.

    Implements the elementwise update::

        W := W − η · ∇W
        b := b − η · ∇b

    Parameters
    ----------
    params : LayerParams
        Current weights and biases for this layer.
    grads : LayerGrads
        Corresponding gradients computed during backprop.
    lr : LearningRate
        The learning-rate hyper-parameter η.

    Returns
    -------
    LayerParams
        Updated parameters (a *new* object — the old one is not mutated).
    """
    eta = lr.v
    new_W = [
        [w - eta * dw for w, dw in zip(wi, dwi)]
        for wi, dwi in zip(params.W, grads.dW)
    ]
    new_b = [bi - eta * dbi for bi, dbi in zip(params.b, grads.db)]
    return LayerParams(W=new_W, b=new_b)


def model_update(
    m: Model,
    g1: LayerGrads,
    g2: LayerGrads,
    g3: LayerGrads,
    lr: LearningRate,
) -> Model:
    """Apply one SGD step to the full model — the endomorphism u_η : Model → Model.

    Category-theory reading
    -----------------------
    This is the endomorphism on ModelState::

        u_η : Model → Model,   u_η(θ) = θ − η · ∇θ

    Training is iterated application of ``model_update``:
    each call produces a *new* :class:`Model` with lower (expected) loss.

    Parameters
    ----------
    m : Model
        Current model parameters.
    g1, g2, g3 : LayerGrads
        Gradients for layers 1, 2, 3 from the backward pass.
    lr : LearningRate
        The learning-rate hyper-parameter η.

    Returns
    -------
    Model
        Updated model (new object, the original is not mutated).
    """
    return Model(
        l1=update_layer(m.l1, g1, lr),
        l2=update_layer(m.l2, g2, lr),
        l3=update_layer(m.l3, g3, lr),
    )


# =============================================================================
# 8.  ONE TRAINING STEP  (the composed morphism)
# =============================================================================
#
# train_step : Model × InputVec × Target × LearningRate → Model × Loss
#
# This is the single end-to-end arrow that composes:
#   forward_lens ∘ loss ∘ backward_pullback ∘ update

def train_step(
    m: Model,
    xs: list[float],
    y: TargetVal,
    lr: LearningRate,
) -> tuple[Model, float]:
    """One complete training step (forward → loss → backward → update).

    Category-theory reading
    -----------------------
    This realises the composite morphism::

        train_step = update ∘ backward ∘ loss ∘ forward

    which maps  (Model, X, Y, η)  to  (Model', loss_value).

    Parameters
    ----------
    m : Model
        Current model parameters.
    xs : list[float]
        Input feature vector for this training example.
    y : TargetVal
        Supervised target scalar.
    lr : LearningRate
        Learning-rate hyper-parameter.

    Returns
    -------
    m_new : Model
        Updated model after one gradient-descent step.
    loss : float
        Scalar loss value for this example before the update.
    """
    # --- Forward pass (lens) ---
    pred, ctx = model_forward(m, xs)
    # --- Loss and seed gradient ---
    loss = mse_loss(pred, y)
    dl_dy = mse_loss_grad(pred, y)
    # --- Backward pass (pullback composition) ---
    g1, g2, g3 = model_backward(ctx, dl_dy)
    # --- Update (endomorphism) ---
    m_new = model_update(m, g1, g2, g3, lr)
    return m_new, loss


# =============================================================================
# 9.  PARAMETER INITIALISATION
# =============================================================================

def glorot_rand(fan_in: int, fan_out: int) -> float:
    """Sample one weight using Glorot-uniform initialisation.

    Draws from U(−limit, limit) where::

        limit = sqrt(6 / (fan_in + fan_out))

    This keeps activations and gradients in a reasonable range at the start
    of training.
    """
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return random.uniform(-limit, limit)


def make_layer(fan_in: int, fan_out: int) -> LayerParams:
    """Create a randomly initialised :class:`LayerParams`.

    Parameters
    ----------
    fan_in : int
        Number of input features (width of the incoming activation vector).
    fan_out : int
        Number of neurons in this layer.

    Returns
    -------
    LayerParams
        Weight matrix of shape (fan_out, fan_in) with Glorot-uniform values,
        and a zero bias vector of length fan_out.
    """
    W = [
        [glorot_rand(fan_in, fan_out) for _ in range(fan_in)]
        for _ in range(fan_out)
    ]
    b = [0.0] * fan_out
    return LayerParams(W=W, b=b)


def make_model() -> Model:
    """Build the full 2-hidden-layer network with random Glorot-uniform weights.

    Architecture::

        input(2) → hidden1(3) → hidden2(3) → output(1)

    Returns
    -------
    Model
        A freshly initialised :class:`Model`.
    """
    return Model(
        l1=make_layer(2, 3),   # Layer 1: 2 inputs  → 3 neurons
        l2=make_layer(3, 3),   # Layer 2: 3 inputs  → 3 neurons
        l3=make_layer(3, 1),   # Layer 3: 3 inputs  → 1 output
    )


# =============================================================================
# 10.  TRAINING LOOP  (iterated endomorphism)
# =============================================================================

# XOR dataset — the classic non-linearly-separable binary problem.
# Inputs and targets live in their typed wrappers.
XOR_DATA: list[tuple[InputVec, TargetVal]] = [
    (InputVec(vals=(0.0, 0.0)), TargetVal(v=0.0)),
    (InputVec(vals=(0.0, 1.0)), TargetVal(v=1.0)),
    (InputVec(vals=(1.0, 0.0)), TargetVal(v=1.0)),
    (InputVec(vals=(1.0, 1.0)), TargetVal(v=0.0)),
]


def train(
    model: Model,
    dataset: list[tuple[InputVec, TargetVal]],
    lr: float,
    epochs: int,
) -> Model:
    """Run the full training loop (iterated endomorphism application).

    Category-theory reading
    -----------------------
    Each epoch is one application of the composed endomorphism over the
    dataset::

        m_{t+1} = u_η^{|D|}(m_t)

    where ``u_η^{|D|}`` denotes ``|D|`` sequential applications of
    :func:`train_step`, one per training example.

    Parameters
    ----------
    model : Model
        Initial (untrained) model.
    dataset : list[tuple[InputVec, TargetVal]]
        List of (input, target) pairs.
    lr : float
        Learning-rate scalar η.
    epochs : int
        Number of full passes over the dataset.

    Returns
    -------
    Model
        The trained model after ``epochs`` epochs.
    """
    learning_rate = LearningRate(v=lr)
    m = model

    for epoch in range(epochs):
        total_loss = 0.0
        for inp, tgt in dataset:
            m, loss = train_step(m, list(inp.vals), tgt, learning_rate)
            total_loss += loss
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d}  total-loss: {total_loss:.6f}")

    return m


# =============================================================================
# 11.  INFERENCE
# =============================================================================

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
    pred, _ctx = model_forward(m, xs)
    return pred.v


# =============================================================================
# 12.  DEMO
# =============================================================================

if __name__ == "__main__":
    random.seed(42)

    print("=== Category-Theory Neural Network (Python) ===")
    print("Architecture: input(2) → hidden1(3) → hidden2(3) → output(1)")
    print("Problem:      XOR")
    print()

    # Build model
    init_model = make_model()

    # Train
    trained_model = train(init_model, XOR_DATA, lr=0.5, epochs=5001)

    # Evaluate
    print()
    print("=== Predictions after training ===")
    for inp, tgt in XOR_DATA:
        xs = list(inp.vals)
        y_hat = predict(trained_model, xs)
        predicted_class = 1 if y_hat > 0.5 else 0
        print(
            f"  Input: {xs}  Target: {tgt.v:.1f}  "
            f"Prediction: {y_hat:.4f}  Class: {predicted_class}"
        )

    print()
    print("=== Category-theory concepts demonstrated ===")
    print("  • Typed wrappers (InputVec, TargetVal, Prediction, LearningRate)")
    print("    prevent mixing up roles — the type IS the semantic tag.")
    print("  • forward_lens_tracked returns (output, pullback_closure)")
    print("    modelling the categorical Lens pattern: P × X → Y × Ctx")
    print("  • backward pullback composition threads ∇Y back through")
    print("    each layer in reverse, implementing f₁* ∘ f₂* ∘ f₃*.")
    print("  • model_update is an endomorphism u_η : Model → Model;")
    print("    training is iterated application of this arrow.")
