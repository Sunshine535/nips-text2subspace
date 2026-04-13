# Proof Skeleton — SFC Theoretical Framework

## 1. Dependency DAG

```
[Def: SAE] ── [Def: Feature Support S(ΔW)]
                    │
                    ├── T1: Sparse Feature Decomposition
                    │       │
                    │       └── C5: Scaling Law (uses T1 sparsity + random feature model)
                    │
                    ├── T2: Interference Localization
                    │       │
                    │       └── [Claim: SFC-Exact optimality] (uses T2 + undefined objective)
                    │
                    └── [Def: FDS] (pure definition, no proof needed)
```

**Cycles detected**: NONE
**Forward references**: "SFC-Exact provably optimal" references T2 but never defines the optimality criterion.

## 2. Assumption Ledger

### T1: Sparse Feature Decomposition

| Assumption | Stated? | Where verified? |
|-----------|---------|----------------|
| SAE has D features | YES | Parameter |
| SAE has "coherence μ" | **UNDEFINED** — μ never defined | NOWHERE |
| ΔW is rank-r | YES | LoRA structure |
| "with high probability over probe distribution" | **VAGUE** — no probability bound, no distribution specified | NOWHERE |
| SAE reconstruction is exact or bounded | **MISSING** — no reconstruction error analysis | NOWHERE |
| Feature activations are non-negative (JumpReLU) | IMPLICIT | SAE architecture |
| Probe distribution is representative | **MISSING** — no distributional assumption | NOWHERE |

### T2: Interference Localization

| Assumption | Stated? | Where verified? |
|-----------|---------|----------------|
| SAE decoder is linear: x̂ = Σ_k f_k · d_k + b | YES | SAE definition |
| Feature coefficients are input-independent (averaged) | **HIDDEN** — the theorem uses mean |Δf_k| but interference is per-input | NOWHERE |
| ψ(c₁,c₂) is the max-pool residual | **VAGUELY** — "captures the max-pool residual" but no formula | NOWHERE |
| Dictionary atoms d_k are fixed (not input-dependent) | YES | SAE architecture |
| SAE reconstruction error is negligible | **MISSING** — not stated, not bounded | NOWHERE |
| Composition is exactly: f_composed = max(f₁, f₂) per feature | IMPLICIT | Algorithm definition |

### C5: Scaling Law

| Assumption | Stated? | Where verified? |
|-----------|---------|----------------|
| Random feature model: each adapter activates features i.i.d. with prob p | YES | Stated |
| p ∝ r/D | **UNJUSTIFIED** — claimed but not derived from T1 | NOWHERE |
| Features are independent across adapters | **HIDDEN** — assumed but adapters may share features systematically | NOWHERE |
| D grows with model size | **PLAUSIBLE** — but relationship not specified | NOWHERE |

## 3. Typed Symbol Table

| Symbol | Type | Depends on | Consistent? |
|--------|------|-----------|-------------|
| ΔW | matrix ∈ R^{d_out × d_in} | layer l | YES |
| r | integer > 0, LoRA rank | architecture choice | YES |
| D | integer > 0, SAE dictionary size | SAE choice | YES |
| S(ΔW) | set ⊆ {1,...,D}, feature support | ΔW, SAE, probe distribution, threshold | YES |
| c_k | scalar ≥ 0, mean feature coefficient | E_x[|f_k(h+δh) - f_k(h)|] | YES but input-dependent nature hidden |
| μ | **UNDEFINED** — called "coherence" | **UNKNOWN** | ⚠ UNDEFINED |
| C | **UNDEFINED** — constant in T1 | **UNKNOWN** | ⚠ UNDEFINED |
| ψ | function R×R → R, "max-pool residual" | **UNSPECIFIED** | ⚠ UNDEFINED |
| d_k | vector ∈ R^{d_model}, SAE decoder column | SAE | YES |
| p | scalar ∈ (0,1), feature activation probability | r, D | YES |
| FDS | scalar ∈ [0,1], Jaccard distance | S₁, S₂ | YES |

## 4. Canonical Quantified Statements

### T1 (current, informal):
"For a rank-r LoRA adapter ΔW and an SAE with D features and coherence μ,
the feature support |S(ΔW)| ≤ C · r · μ(SAE) with high probability
over a probe distribution."

### T1 (REQUIRED precise version):
```
∀ rank-r LoRA adapter ΔW ∈ R^{d_out × d_in},
∀ SAE with dictionary W_dec ∈ R^{D × d_model},
∀ threshold ε > 0,
∀ probe distribution P over R^{d_in}:
  Define S_ε(ΔW) = {k : E_{x~P}[|f_k(Wx + ΔWx) - f_k(Wx)|] > ε}
  Then: |S_ε(ΔW)| ≤ ??? 
```
**CANNOT COMPLETE** — the bound depends on the SAE's properties (coherence, conditioning)
and the probe distribution in ways that are never specified.

### T2 (current, informal):
"Interference = Σ_{k∈S₁∩S₂} ψ(c₁ₖ,c₂ₖ) · ||d_k||².
When S₁ ∩ S₂ = ∅, interference = 0 exactly."

### T2 (REQUIRED precise version):
```
Define: for adapter i, feature profile c_i ∈ R^D_≥0 with support S_i
Define: SFC composition c_composed = max(c_1, c_2) element-wise
Define: oracle composition c_oracle = c_1 + c_2
Define: interference = ||W_dec^T(c_composed - c_oracle)||^2

Then: interference = Σ_{k∈S₁∩S₂} (c₁ₖ + c₂ₖ - max(c₁ₖ,c₂ₖ))² · ||d_k||²
                   = Σ_{k∈S₁∩S₂} min(c₁ₖ,c₂ₖ)² · ||d_k||²  [for c ≥ 0]

When S₁ ∩ S₂ = ∅: the sum is empty, interference = 0.
```
**THIS IS FORMALIZABLE** but requires:
1. Define what "oracle" means (why is sum the oracle?)
2. Dictionary atoms d_k are approximately orthogonal (otherwise cross-terms exist)
3. The "interference" metric is in reconstruction space, not task performance space

### C5 (current):
"E[overlap(N)] / E[union(N)] = p^N / (1 - (1-p)^N)"

### C5 (precise):
```
Under i.i.d. Bernoulli(p) feature activation model:
  For N adapters with feature supports S_1,...,S_N drawn i.i.d.:
    E[|∩_i S_i|] / E[|∪_i S_i|] = p^N / (1 - (1-p)^N)
  
  As D → ∞ with p = r·μ/D:
    E[|∩_i S_i|] / E[|∪_i S_i|] → 0 exponentially in N
```
**DERIVABLE** from standard combinatorics, BUT the i.i.d. assumption is unrealistic.

## 5. Micro-Claim Inventory

### MC-1: LoRA activation perturbation is low-rank
Context: ΔW = BA with B ∈ R^{d_out×r}, A ∈ R^{r×d_in}
⊢ Goal: δh(x) = ΔW·x lies in an r-dimensional subspace of R^{d_out}
Rule: image of rank-r matrix is at most r-dimensional
Side-conditions: NONE ✓
Status: TRIVIALLY TRUE

### MC-2: Low-rank perturbation → sparse SAE features
Context: δh ∈ span(B) which is r-dimensional; SAE has D >> d_out features
⊢ Goal: |{k : E[|f_k(h+δh) - f_k(h)|] > ε}| is small
Rule: ??? **NO STANDARD THEOREM APPLIES**
Side-conditions: requires SAE-specific properties (incoherence, RIP, etc.)
Status: **UNJUSTIFIED** — this is the core gap

### MC-3: Disjoint supports → zero interference (in SAE reconstruction space)
Context: c₁, c₂ ≥ 0 with S₁ ∩ S₂ = ∅; composition = max(c₁,c₂)
⊢ Goal: ||W_dec^T max(c₁,c₂) - W_dec^T(c₁+c₂)|| = 0
Rule: when supports disjoint, max(c₁,c₂) = c₁ + c₂ (since one is 0 at each k)
Side-conditions: NONE ✓
Status: **TRUE by construction** (trivial)

### MC-4: SAE reconstruction interference = task performance interference
Context: interference defined in ||W_dec^T · Δc|| space
⊢ Goal: small reconstruction interference → small downstream accuracy loss
Rule: ??? **NO THEOREM ESTABLISHES THIS**
Side-conditions: requires bounding how SAE reconstruction error maps to task metrics
Status: **UNJUSTIFIED** — major gap

### MC-5: p ∝ r/D in real adapters
Context: T1 claims |S| ≤ C·r·μ
⊢ Goal: practical adapters have |S|/D ∝ r/D
Rule: T1 (if proven)
Side-conditions: μ must be bounded, C must be small
Status: **DEPENDS ON T1** which is itself unjustified

## 6. Limit-Order Map

| Statement | Limit | Uniform in? | Verified? |
|-----------|-------|------------|-----------|
| |S|/D < 0.05 | fixed SAE, fixed adapter | nothing | **EMPIRICAL CLAIM, NOT THEOREM** |
| E[overlap]/E[union] = p^N/(1-(1-p)^N) | D → ∞ implied but not stated | N, p | **NOT STATED** |
| "scales inversely with D" | D → ∞ | r, μ | **VAGUE** |
