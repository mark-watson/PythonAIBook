# Chapter: Categorical Deep Learning – From Axioms to Implementation

This chapter explores a concrete implementation of the theoretical frameworks proposed in **Jia, Peng, Yang & Chen (2025)**: *"Category-Theoretical and Topos-Theoretical Frameworks in Machine Learning: A Survey."* 

The focus is on how high-level categorical constructs—such as Para categories, Markov categories, Sheaves, Natural Transformations, and the Subobject Classifier—map directly to the building blocks of neural networks implemented in idiomatic Python. By bypassing heavy dependencies like NumPy, this implementation highlights the mathematical elegance behind backpropagation, stochasticity, invariance, and ensemble consistency.

---

## I. Gradient-Based Learning: The Para Category & Lenses

Traditional backpropagation is often taught as a sequence of calculus rules applied by hand. Categorical deep learning views it more profoundly: layers are **morphisms in the Para category**, which is specifically designed to handle parametric maps between states.

### 1. Morphisms as Parametric Maps (Para)
In the Para construction, a morphism $f : A \to B$ is a triple $(P, F, F^*)$, where:
*   **$P$** is the parameter space (e.g., weights and biases).
*   **$F: P \times A \to B$** is the forward pass (the map from parameters and input to output).
*   **$F^*: P \times A \times \nabla B \to \nabla P \times \nabla A$** is the "pullback" or backward pass, which computes how infinitesimal changes in the output ($\nabla B$) propagate back into parameter space ($\nabla P$) and input space ($\nabla A$).

In this implementation, `LayerParams` represents an object in the Parameter Space, and $\text{Para}(\mathbb{Euc})$ is implemented by defining both a forward pass and a "pullback closure" (the lens) that captures enough state to perform backpropagation.

### 2. Compositional Backpropagation
One of the key results is that neural networks are simply the **composition of Para morphisms**. If we have layers $f$, $g$, and $h$, the cumulative effect is:
$$F_{net} = f_n \circ f_{n-1} \circ \dots \circ f_1$$

The implementation uses this principle to perform "Compositional Backpropagation." By stacking pullback closures (returned by `forward_para`) and applying them in reverse order, we realize the compositional rule for gradients. This is implemented in `model_backward`, which composes pullbacks $\text{f}_n^* \circ \dots \circ \text{f}_1^*$ via a "functorial" approach on the gradient space.

---

## II. Probability-Based Learning: Markov Categories & Stochasticity

Machine learning is inherently stochastic (dropout, weight noise, sampling). We model this using **Markov categories**, where morphisms represent stochastic kernels—mapping objects to probability distributions over other objects.

### 1. Dropout as a Stochastic Lens
Dropout can be formalized as a **stochastic lens** (survey §3.2). A lens is "closed" if it preserves the same random realization in both its forward and backward pass. In our code, `make_dropout_lens` samples an array of Bernoulli variables $\{0, 1\}$ for each neuron and returns a closure (`pullback`) that reuses this exact mask during backpropagation. This ensures mathematical consistency between the stochastic forward pass and the deterministic gradient flow through the active sub-network.

### 2. Bayesian Uncertainty & Monte Carlo Sampling
In a **Bayesian Neural Network**, weights are treated as random variables $W \sim p(W)$. A single layer becomes a stochastic morphism: instead of receiving fixed weights, it samples from a distribution at every forward pass. We implement this using `bayesian_forward`. Using **Monte Carlo (MC) sampling** ($n$ forward passes), we can compute the empirical mean and standard deviation of the predictions to quantify *epistemic uncertainty* (the model's "uncertainty" about its own parameterization).

---

## III. Invariance & Equivariance: Colimits in Geometry

Neural networks often require invariance to certain transformations (like permuting pixel order or translating images). This is modeled using **colimits**.

### 1. Permutation-Invariance via Summation
A function is invariant to the action of a group $G$ if it produces the same output for any permutation of its inputs. In categorical terms, summation ($\sum x_i$) acts as a **colimit (coproduct)** over the diagram representing a symmetric group action. Because addition is commutative and associative, weight-sharing sum-pooling is naturally a morphism that factors through the colimit of the input space.

### 2. K-means: Iterative Colimit Computation
K-means clustering can be viewed as an iterative process to find **colimits** in a metric space category. Each cluster centroid acts as a representative (a limit/colimit) for its assigned points; the assignment of data points to centroids is the universal mapping that mediates between individual data objects and their respective cluster representatives.

---

## IV. The Topos Framework: Subobject Classifiers & Sheaves

The most abstract view treats neural networks within a **Topos**—a category with rich internal logic (Heyting algebras) and structural tools for handling local-to-global transitions.

### 1. The Subobject Classifier ($\Omega$)
A topos possesses a special object called the **subobject classifier**, $\Omega$, which admits a "true" morphism $\top : 1 \to \Omega$. In binary classification, the continuous output of a sigmoid (probability $[0,1]$) is interpreted as a characteristic morphism $\chi: X \to \Omega$. A prediction of $p=1.0$ corresponds to truth; $p=0.5$ represents the boundary where an input transitions from being "in" to "out" of the classified subobject.

### 2. Sheaf Gluing & Local-to-Global Consistency
A **Sheaf** is a structure that assigns a set (a "section") to every open set in a topological space, such that local sections can be "glued" together if they agree on their overlaps. In an ensemble of models:
*   Each model acts as an expert over a specific input context (an open set).
*   The **Sheaf Condition** requires that for any two overlapping contexts, the expert predictions must agree within some tolerance $\epsilon$.
*   If experts are consistent, we can perform **Sheaf Gluing** to produce a single global prediction. If they disagree (high variance), we fail to form a global section, signaling high epistemic uncertainty or model conflict.

---

## V. Natural Transformations: Knowledge Distillation

A **Natural Transformation** $\eta : F \Rightarrow G$ is a coherent mapping between two functors (tasks/models). This provides a rigorous foundation for **Knowledge Distillation**.

If we have a complex "Teacher" network $F$ and a simpler "Student" network $G$, the distillation process can be seen as a natural transformation $\eta_X$ that maps the high-dimensional feature space of $F$ onto the lower-dimensional representation of $G$. This adapter preserves the structural relationships (commutation) between how the two networks transform their inputs.

---

## Summary: Categorical Mapping Overview

| ML Concept | Implementation Detail | Categorical Interpretation |
| :--- | :--- | :--- |
| **Backpropagation** | `model_backward` | Composition of pullbacks $f^* \circ g^* \dots$ |
| **Layer Operation** | `forward_para` | Para morphism (Primal + Pullback) |
| **Dropout** | `make_dropout_lens` | Closed Stochastic Lens / Optic |
| **Bayesian Sampling**| `bayesian_forward` | Morphism in the Markov Category **Stoch** |
| **Pooling** | `permutation_invariant_pool` | Colimit of a Symmetric Group action |
| **Ensemble Logic** | `sheaf_glue` | Gluing lemma for Sheaves |
| **Distillation** | `NatTransform` | Natural transformation $\eta : F \Rightarrow G$ |
| **Binary Decision** | Sigmoid output ($\chi$) | Subobject classifier morphism $\chi : X \to \Omega$ |

*This chapter provides a bridge from the mathematical foundations established in the survey paper to practical, readable Python code.*