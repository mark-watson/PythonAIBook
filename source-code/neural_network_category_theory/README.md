# Category-Theory-Inspired Neural Network (Python)

A pure-Python implementation of a 2-hidden-layer neural network whose
architecture is explained through the lens of category theory.

Ported from the Racket original in
[`Racket-AI-book/source-code/neural_category_theory`](../../../../Racket-AI-book/source-code/neural_category_theory).

Reference: <https://rust-ml.com/materials/one-training-step-end-to-end.html>

## Category-Theory Concepts Demonstrated

| Concept | Where it appears |
|---|---|
| **Typed newtypes** — `InputVec`, `TargetVal`, `Prediction`, `LearningRate` | §1 — prevent accidental role confusion; the *type* is the semantic tag |
| **Lens morphism** `f : P × X → Y × Ctx` | `forward_lens_tracked` returns `(activations, pullback_closure)` |
| **Pullback / put-back** `f* : ∇Y → ∇P × ∇X` | Closure returned by `forward_lens_tracked`; captures `W`, `inputs`, `acts` |
| **Lens composition** | `model_forward` sequences three lenses: `f₃ ⊚ f₂ ⊚ f₁` |
| **Pullback composition** (reverse) | `model_backward` threads `∇Ŷ` through `f₁* ∘ f₂* ∘ f₃*` |
| **Endomorphism** `u_η : Model → Model` | `model_update` — SGD step; training = iterated application |

## Architecture

```
input(2)  →  hidden1(3)  →  hidden2(3)  →  output(1)
```

- **Activation:** sigmoid on every layer  
- **Loss:** mean-squared error  
- **Optimiser:** vanilla SGD  
- **Demo task:** XOR (non-linearly separable binary classification)

## Running

```bash
uv run neural_network_category_theory.py
```

Expected output after ~5 000 epochs (learning rate 0.5):

```
Epoch     0  total-loss: 1.103494
Epoch  1000  total-loss: 1.066174
Epoch  2000  total-loss: 1.064834
Epoch  3000  total-loss: 1.056367
Epoch  4000  total-loss: 0.008396
Epoch  5000  total-loss: 0.002651

=== Predictions after training ===
  Input: [0.0, 0.0]  Target: 0.0  Prediction: 0.0269  Class: 0
  Input: [0.0, 1.0]  Target: 1.0  Prediction: 0.9731  Class: 1
  Input: [1.0, 0.0]  Target: 1.0  Prediction: 0.9726  Class: 1
  Input: [1.0, 1.0]  Target: 0.0  Prediction: 0.0211  Class: 0
```

## File Layout

| File | Description |
|---|---|
| `neural_network_category_theory.py` | Main implementation — all 12 sections |
| `pyproject.toml` | `uv` project metadata |
| `NOT_IN_BOOK.md` | Note about book inclusion status |
| `README.md` | This file |

## Requirements

- Python ≥ 3.10 (uses `match`-free dataclasses + `from __future__ import annotations`)
- No third-party libraries — stdlib only (`math`, `random`, `dataclasses`)
- [`uv`](https://github.com/astral-sh/uv) for running the script
