# Papers: LoRA Algebra

Core references for the algebraic operations on LoRA weight spaces project,
organized by relevance category.

---

## Category A: LoRA Generation (Different Goal — We Build ON Their Outputs)

### Text-to-LoRA: Instant LoRA Generation from Natural Language
- **Authors:** Zhong et al.
- **Venue:** ICLR 2025
- **URL:** https://arxiv.org/abs/2412.XXXXX
- **Key idea:** Generate LoRA adapter weights from natural language task description via hypernetwork
- **Relevance:** Produces LoRA adapters that our algebra can then compose/interpolate
- **Distinction:** T2L GENERATES individual LoRAs; we define OPERATIONS on collections of LoRAs

### LoRAGen: LoRA Generation for Efficient Task-Specific Adaptation
- **Authors:** Various
- **Venue:** ICLR 2026
- **URL:** https://arxiv.org/abs/2501.XXXXX
- **Key idea:** Improved LoRA generation with better generalization to unseen tasks
- **Relevance:** Another source of LoRA adapters; complementary to our algebra
- **Distinction:** LoRAGen generates → we compose/manipulate the generated outputs

---

## Category B: Heuristic Model Merging (Direct Baselines)

### TIES-Merging: Resolving Interference When Merging Models
- **Authors:** Yadav et al.
- **Venue:** NeurIPS 2023
- **URL:** https://arxiv.org/abs/2306.01708
- **Key idea:** Trim low-magnitude, resolve sign conflicts via election, merge surviving parameters
- **Relevance:** Primary heuristic baseline for composition
- **Limitation:** No theoretical guarantee; sign election is a heuristic that ignores subspace structure

### DARE: Language Models are Super Mario: Absorbing Abilities from Homologous Models
- **Authors:** Yu et al.
- **Venue:** ICML 2024
- **URL:** https://arxiv.org/abs/2311.03099
- **Key idea:** Randomly drop delta parameters with probability p, rescale by 1/(1-p), then merge
- **Relevance:** Primary heuristic baseline for interference reduction
- **Limitation:** Random dropping destroys important directions; no rank or subspace awareness

### Task Arithmetic: Editing Models with Task Arithmetic
- **Authors:** Ilharco et al.
- **Venue:** ICLR 2023
- **URL:** https://arxiv.org/abs/2212.04089
- **Key idea:** Task vectors τ = θ_ft − θ_base can be added/subtracted for multi-task composition
- **Relevance:** Foundational work on weight-space arithmetic; our Grassmann approach generalizes this
- **Limitation:** Linear approximation fails when task vectors are not in shared subspace

---

## Category C: Theoretical Foundations (Grassmann Manifold & Subspaces)

### Grassmann Manifold Optimization
- **Authors:** Edelman, Arias, Smith
- **Venue:** SIAM Journal on Matrix Analysis (1998)
- **URL:** https://arxiv.org/abs/physics/9806030
- **Key idea:** Geometry, algorithms, and optimization on Grassmann manifolds
- **Relevance:** Mathematical foundation for our algebraic operations

### Riemannian Optimization on the Grassmann Manifold
- **Authors:** Absil, Mahony, Sepulchre
- **Venue:** Optimization Algorithms on Matrix Manifolds (Princeton, 2008)
- **URL:** https://press.princeton.edu/absil
- **Key idea:** Practical algorithms for geodesics, exponential/logarithmic maps on Gr(r,d)
- **Relevance:** Implementation reference for our manifold operations

### Subspace Angles and Distances
- **Authors:** Björck, Golub
- **Venue:** Mathematics of Computation (1973)
- **Key idea:** Principal angles between subspaces define the Grassmann distance
- **Relevance:** Distance metric for comparing LoRA subspaces

---

## Category D: Neural Network Weight Space Geometry

### Git Re-Basin: Merging Models Modulo Permutation Symmetries
- **Authors:** Ainsworth et al.
- **Venue:** ICLR 2023
- **URL:** https://arxiv.org/abs/2209.04836
- **Key idea:** Align neural network weight bases via permutation search before merging
- **Relevance:** Addresses basis alignment but only for discrete permutations, not continuous rotations
- **Limitation:** NP-hard permutation search; doesn't handle LoRA's continuous rotational symmetry

### Model Soups: Averaging Weights of Multiple Fine-tuned Models
- **Authors:** Wortsman et al.
- **Venue:** ICML 2022
- **URL:** https://arxiv.org/abs/2203.05482
- **Key idea:** Simple weight averaging of independently fine-tuned models improves accuracy
- **Relevance:** Motivates weight-space operations; our algebra provides the theoretical framework

### Linear Mode Connectivity and the Lottery Ticket Hypothesis
- **Authors:** Frankle et al.
- **Venue:** ICML 2020
- **URL:** https://arxiv.org/abs/1912.05671
- **Key idea:** Models trained from same initialization are linearly connected in loss landscape
- **Relevance:** Theoretical justification for weight interpolation; we extend to LoRA subspaces

---

## Category E: LoRA Foundations

### LoRA: Low-Rank Adaptation of Large Language Models
- **Authors:** Hu et al.
- **Venue:** ICLR 2022
- **URL:** https://arxiv.org/abs/2106.09685
- **Key idea:** Adapt LLMs via low-rank weight updates W + BA where B ∈ R^{d×r}, A ∈ R^{r×k}
- **Relevance:** Foundational method; our algebra operates on LoRA's (A, B) matrices

### QLoRA: Efficient Finetuning of Quantized LLMs
- **Authors:** Dettmers et al.
- **Venue:** NeurIPS 2023
- **URL:** https://arxiv.org/abs/2305.14314
- **Key idea:** 4-bit quantized base model + LoRA adapters for memory efficiency
- **Relevance:** We may use QLoRA for training domain adapters on A100s

### DoRA: Weight-Decomposed Low-Rank Adaptation
- **Authors:** Liu et al.
- **Venue:** ICML 2024
- **URL:** https://arxiv.org/abs/2402.09353
- **Key idea:** Decompose weight into magnitude and direction components for LoRA
- **Relevance:** Direction-magnitude decomposition relates to our Grassmann (direction) framework

---

## Category F: Multi-Task & Continual Learning with Adapters

### AdapterFusion: Non-Destructive Task Composition for Transfer Learning
- **Authors:** Pfeiffer et al.
- **Venue:** EACL 2021
- **URL:** https://arxiv.org/abs/2005.00247
- **Key idea:** Learn attention-based fusion of task-specific adapters
- **Relevance:** Learned composition baseline; ours is algebraic, not attention-based

### Orthogonal Subspace Learning for Language Model Continual Learning
- **Authors:** Various
- **Venue:** Various (2023-2024)
- **Key idea:** Constrain new task learning to orthogonal subspace to prevent forgetting
- **Relevance:** Orthogonality in adapter subspaces relates to our projection operation

---

## Reading Priority

| Priority | Papers | Reason |
|----------|--------|--------|
| P0 (must read) | TIES, DARE, Task Arithmetic | Direct heuristic baselines we improve upon |
| P0 (must read) | LoRA, Grassmann optimization | Foundations of our method |
| P1 (should read) | T2L, LoRAGen, Git Re-Basin | Related work positioning |
| P1 (should read) | DoRA, Model Soups | Weight-space geometry connections |
| P2 (nice to have) | AdapterFusion, Linear Mode Connectivity | Broader context |
