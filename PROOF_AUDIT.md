# Proof Audit — SFC Theoretical Framework

**Difficulty**: nightmare
**Date**: 2026-04-14
**Status**: Round 1

## Round 1: Self-Audit (Pre-Codex)

### Issue List

---

#### ISSUE-1: T1 has no proof and is likely FALSE as stated
- **Status**: INVALID
- **Impact**: GLOBAL
- **Category**: UNJUSTIFIED_ASSERTION + SCOPE_OVERCLAIM
- **Severity**: **FATAL**
- **Location**: PROPOSAL.md, Theorem 1

**Statement**: |S(ΔW)| ≤ C · r · μ(SAE) with high probability

**Why INVALID**:

1. **μ (coherence) is undefined.** The theorem uses a symbol that never receives a definition. Without knowing what μ measures, the bound is meaningless.

2. **The claim that low-rank → sparse in SAE space is NOT generally true.** A rank-r perturbation δh = ΔW·x lives in an r-dimensional subspace of R^{d_out}. But SAE features are NOT aligned with any particular low-dimensional subspace. If the SAE dictionary atoms {d_k} are spread across the full d_model-dimensional space (as they should be for good reconstruction), then a rank-r perturbation can activate up to O(D) features — specifically, ALL features whose dictionary atom has nonzero projection onto the r-dimensional subspace.

   **Counterexample**: Let d_model = 1000, r = 1, D = 16384. The perturbation δh is a single direction v ∈ R^{1000}. The SAE encodes via z_k = w_k^T · δh = (w_k^T v). For random dictionary atoms w_k, approximately ALL D features will have nonzero z_k. The threshold determines sparsity, not the rank.

3. **The sparsity is an artifact of the threshold, not the theory.** In the synthetic verification (verify_sfc_synthetic.py), the "sparsity = 5%" result comes from using the 95th percentile threshold — by definition, 5% of features are above the 95th percentile. This is tautological.

**Counterexample confirmed**: The synthetic experiment uses `threshold_percentile=95.0`, which means |S| = 0.05·D BY CONSTRUCTION. The "theorem" is verifying the definition of a percentile, not a structural property.

**Affects**: T1 → C5 (scaling law depends on T1), entire SFC narrative ("sparse in feature space")

**Minimal fix**: 
- Option A: Restate as empirical observation, not theorem
- Option B: Define coherence properly (e.g., maximum correlation between any SAE atom and the perturbation subspace) and derive a proper bound
- Option C: Use RIP (Restricted Isometry Property) or incoherence from compressed sensing literature

---

#### ISSUE-2: T2 interference formula ignores SAE reconstruction error
- **Status**: UNDERSTATED
- **Impact**: GLOBAL
- **Category**: HIDDEN_ASSUMPTION
- **Severity**: **CRITICAL**
- **Location**: PROPOSAL.md, Theorem 2

**Statement**: Interference(SFC) = Σ_{k∈S₁∩S₂} ψ(c₁ₖ,c₂ₖ) · ||d_k||²

**Why UNDERSTATED**:

The theorem assumes SAE is perfect: encode(decode(f)) = f. In reality:
1. SAE has nonzero reconstruction error: h ≠ decode(encode(h))
2. The composition operates on ENCODED features, not true activations
3. There is an additional error term: ||SAE_error(h₁+h₂) - SAE_error(h₁) - SAE_error(h₂)||

The "interference = 0 when disjoint" claim is only true IN THE SAE'S LATENT SPACE, not in the model's actual activation space. The SAE reconstruction error provides a floor on true interference that may dominate the compositional interference.

**Affects**: Core claim "provably interference-free"

**Minimal fix**: Add explicit reconstruction error bound. Theorem becomes:
  True_interference ≤ Feature_interference + 2·SAE_reconstruction_error

---

#### ISSUE-3: T2's "oracle" is undefined and unjustified
- **Status**: UNJUSTIFIED
- **Impact**: LOCAL
- **Category**: MISSING_DERIVATION
- **Severity**: **MAJOR**
- **Location**: PROPOSAL.md, Theorem 2

**Why**: The theorem measures interference as distance from an "oracle" composition c_oracle = c₁ + c₂. But:
1. Why is summing coefficients the oracle? If two adapters both amplify feature k, the correct composition might be max, not sum.
2. The "oracle" is never defined in terms of downstream task performance.
3. Different definitions of "oracle" give different interference formulas.

**Minimal fix**: Define oracle precisely. E.g., "the single-adapter activation effect applied independently" = c₁ + c₂ when features don't interact. Acknowledge this is additive-interaction assumption.

---

#### ISSUE-4: ψ is never defined
- **Status**: UNCLEAR
- **Impact**: LOCAL  
- **Category**: MISSING_DERIVATION
- **Severity**: **MAJOR**
- **Location**: PROPOSAL.md, Theorem 2

**Statement**: "ψ captures the max-pool residual"

**Why**: For non-negative coefficients with oracle = sum:
  ψ(a,b) = (a + b - max(a,b))² = min(a,b)²

This should be stated explicitly. The current "captures" language hides the actual formula.

**Minimal fix**: Write ψ(a,b) = min(a,b)² explicitly.

---

#### ISSUE-5: "SFC-Exact provably optimal" has no theorem or proof
- **Status**: UNJUSTIFIED
- **Impact**: GLOBAL
- **Category**: UNJUSTIFIED_ASSERTION + SCOPE_OVERCLAIM
- **Severity**: **CRITICAL**
- **Location**: PROPOSAL.md, C3

**Statement**: "SFC-Exact: Hook-based inference, applies composed features at each layer. Provably optimal."

**Why**: 
1. "Optimal" with respect to WHAT objective? Never defined.
2. Max-pool is NOT optimal for all objectives. If the goal is to minimize ||composed_effect - oracle_effect||², sum is optimal (error = 0), not max-pool.
3. The synthetic verification confirms: sum has error = 0.0000, maxpool has error = 1.2605. Max-pool is STRICTLY WORSE than sum.
4. The "optimality" claim in the synthetic test is defined as "better than mean and random," which is a trivially low bar.

**Counterexample from our own experiments**:
```
maxpool: error=1.2605
mean:    error=2.9167  
sum:     error=0.0000  ← THIS IS THE OPTIMAL METHOD
random:  error=3.7925
```

**Affects**: Core method claim. SFC's max-pool is an ad-hoc choice, not provably optimal.

**Minimal fix**: 
- Remove "provably optimal" claim
- Explain why max-pool instead of sum: sum would double-count shared features, so max-pool is appropriate when features are NOT additive (e.g., binary activation vs. magnitude)
- Or: change composition rule to sum (which IS optimal for additive interference model)

---

#### ISSUE-6: C5 Scaling Law uses i.i.d. assumption that is empirically false
- **Status**: OVERSTATED
- **Impact**: LOCAL
- **Category**: HIDDEN_ASSUMPTION + SCOPE_OVERCLAIM
- **Severity**: **MAJOR**
- **Location**: PROPOSAL.md, C5

**Statement**: Under random feature model with i.i.d. Bernoulli(p) activation

**Why OVERSTATED**:
1. Different task domains share features systematically (e.g., math and science overlap more than math and history)
2. The i.i.d. model predicts overlap ∝ p² which doesn't account for semantic similarity
3. The scaling law E[overlap]/E[union] = p^N/(1-(1-p)^N) is correct UNDER the model but the model doesn't match reality

**Minimal fix**: Present as "baseline prediction under independence" and compare with actual observed overlap patterns. The deviation from i.i.d. prediction is itself informative.

---

#### ISSUE-7: No theorem connects SAE feature interference to downstream task performance
- **Status**: UNJUSTIFIED
- **Impact**: GLOBAL
- **Category**: LOGICAL_GAP
- **Severity**: **CRITICAL**
- **Location**: Entire framework

**Why**: The entire SFC framework operates in SAE feature space. But:
1. Features ≠ task performance
2. Two adapters with zero feature overlap could still interfere through nonlinear model dynamics
3. Two adapters with high feature overlap might not interfere if the shared features are task-agnostic
4. SAE reconstruction error provides a floor on interference regardless of feature overlap

**This is the deepest theoretical gap in the paper.**

**Minimal fix**: Add a theorem: "If SAE reconstruction error ≤ ε_sae per token, and feature interference ≤ ε_feat, then downstream task error ≤ f(ε_sae, ε_feat, model_depth, ...)"

---

## Summary Table

| ID | Category | Status | Impact | Severity | Fixable? |
|----|----------|--------|--------|----------|----------|
| 1 | UNJUSTIFIED_ASSERTION | INVALID | GLOBAL | **FATAL** | Yes — restate as empirical + add proper bound |
| 2 | HIDDEN_ASSUMPTION | UNDERSTATED | GLOBAL | **CRITICAL** | Yes — add reconstruction error term |
| 3 | MISSING_DERIVATION | UNJUSTIFIED | LOCAL | MAJOR | Yes — define oracle precisely |
| 4 | MISSING_DERIVATION | UNCLEAR | LOCAL | MAJOR | Yes — write ψ explicitly |
| 5 | SCOPE_OVERCLAIM | UNJUSTIFIED | GLOBAL | **CRITICAL** | Yes — remove "optimal" or change to sum |
| 6 | HIDDEN_ASSUMPTION | OVERSTATED | LOCAL | MAJOR | Yes — reframe as baseline model |
| 7 | LOGICAL_GAP | UNJUSTIFIED | GLOBAL | **CRITICAL** | Hard — requires new theorem |

### Acceptance Gate: **FAIL**
- FATAL issues: 1 (ISSUE-1)
- CRITICAL issues: 3 (ISSUE-2, 5, 7)
- MAJOR issues: 3 (ISSUE-3, 4, 6)

### Most Devastating Finding

**ISSUE-1 + ISSUE-5 combined**: The "sparsity" result in our synthetic verification is tautological (95th percentile → 5% by definition), and max-pool is provably worse than sum in our own experiments. The two core claims — "adapters are sparse in SAE space" and "max-pool is optimal" — are both unsupported.
