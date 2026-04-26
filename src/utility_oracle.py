"""Utility oracle for UCAR: compute per-item best candidate over base.

For each evaluation item, runs every candidate (base, single adapters,
static merges, CARR variants) and records logprob MCQ predictions.
The oracle selects the best candidate per item — an upper bound on what
any router can achieve with this candidate pool.

If oracle <= base, no routing method can help; candidate pool must be improved.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

CHOICE_LABELS = ["A", "B", "C", "D"]


@dataclass
class CandidateResult:
    name: str
    item_idx: int
    domain: str
    gold_idx: int
    gold_label: str
    option_logprobs: List[float]   # (n_options,)
    predicted_idx: int
    predicted_label: str
    correct: bool
    gold_logprob: float            # logprob of gold option


@dataclass
class UtilityRow:
    item_idx: int
    domain: str
    question_hash: str
    gold_label: str
    gold_idx: int
    candidates: Dict[str, CandidateResult]
    oracle_candidate: str
    oracle_correct: bool
    base_correct: bool
    oracle_gain_over_base: float   # gold_logprob(oracle) - gold_logprob(base)


def compute_option_logprobs_for_item(
    model, tokenizer, prompt: str, device: str = "cuda",
) -> torch.Tensor:
    """Score each of A/B/C/D by length-normalized sum logprob. Returns (4,)."""
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=1024).input_ids.to(device)
    scores = []
    with torch.no_grad():
        for opt in CHOICE_LABELS:
            opt_ids = tokenizer(" " + opt, add_special_tokens=False).input_ids
            if len(opt_ids) == 0:
                opt_ids = tokenizer(opt, add_special_tokens=False).input_ids
            opt_tensor = torch.tensor([opt_ids], device=device, dtype=prompt_ids.dtype)
            full = torch.cat([prompt_ids, opt_tensor], dim=-1)
            logits = model(full).logits
            target = full[:, prompt_ids.shape[1]:]
            log_probs = F.log_softmax(
                logits[:, prompt_ids.shape[1] - 1:-1, :].float(), dim=-1)
            s = log_probs.gather(-1, target.unsqueeze(-1)).sum().item() / max(len(opt_ids), 1)
            scores.append(s)
    return torch.tensor(scores)


def evaluate_candidate_on_items(
    model, tokenizer, items: List[dict], candidate_name: str,
    device: str = "cuda",
) -> List[CandidateResult]:
    """Run a single candidate model on all items. items are dicts with prompt/gold_idx/gold_label/domain."""
    results = []
    model.eval()
    for item in items:
        lps = compute_option_logprobs_for_item(model, tokenizer, item["prompt"], device)
        pred_idx = int(lps.argmax().item())
        results.append(CandidateResult(
            name=candidate_name,
            item_idx=item["item_idx"],
            domain=item["domain"],
            gold_idx=item["gold_idx"],
            gold_label=item["gold_label"],
            option_logprobs=lps.tolist(),
            predicted_idx=pred_idx,
            predicted_label=CHOICE_LABELS[pred_idx],
            correct=pred_idx == item["gold_idx"],
            gold_logprob=lps[item["gold_idx"]].item(),
        ))
    return results


def build_utility_table(
    candidate_results: Dict[str, List[CandidateResult]],
) -> List[UtilityRow]:
    """Combine per-candidate results into per-item oracle analysis."""
    base_results = candidate_results.get("base", [])
    if not base_results:
        raise ValueError("'base' candidate is required")

    n_items = len(base_results)
    rows = []
    for i in range(n_items):
        base_r = base_results[i]
        item_candidates = {}
        for cname, cresults in candidate_results.items():
            if i < len(cresults):
                item_candidates[cname] = cresults[i]

        oracle_name = max(
            item_candidates.keys(),
            key=lambda k: item_candidates[k].gold_logprob,
        )
        oracle_r = item_candidates[oracle_name]

        q_hash = hashlib.sha256(str(base_r.item_idx).encode()).hexdigest()[:16]
        rows.append(UtilityRow(
            item_idx=base_r.item_idx,
            domain=base_r.domain,
            question_hash=q_hash,
            gold_label=base_r.gold_label,
            gold_idx=base_r.gold_idx,
            candidates=item_candidates,
            oracle_candidate=oracle_name,
            oracle_correct=oracle_r.correct,
            base_correct=base_r.correct,
            oracle_gain_over_base=oracle_r.gold_logprob - base_r.gold_logprob,
        ))
    return rows


def summarize_oracle(rows: List[UtilityRow], candidate_names: List[str]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"error": "no items"}

    base_acc = sum(1 for r in rows if r.base_correct) / n
    oracle_acc = sum(1 for r in rows if r.oracle_correct) / n
    oracle_gain = sum(r.oracle_gain_over_base for r in rows) / n

    per_candidate_acc = {}
    for cname in candidate_names:
        correct = sum(
            1 for r in rows
            if cname in r.candidates and r.candidates[cname].correct
        )
        total = sum(1 for r in rows if cname in r.candidates)
        per_candidate_acc[cname] = correct / max(total, 1)

    oracle_selection_counts = {}
    for r in rows:
        oracle_selection_counts[r.oracle_candidate] = oracle_selection_counts.get(
            r.oracle_candidate, 0) + 1

    complementarity_items = sum(
        1 for r in rows
        if r.oracle_correct and not r.base_correct
    )

    return {
        "n_items": n,
        "base_accuracy": base_acc,
        "oracle_accuracy": oracle_acc,
        "oracle_gain_over_base_logprob": oracle_gain,
        "oracle_accuracy_lift": oracle_acc - base_acc,
        "complementarity_items": complementarity_items,
        "complementarity_rate": complementarity_items / n,
        "per_candidate_accuracy": per_candidate_acc,
        "oracle_selection_counts": oracle_selection_counts,
        "oracle_beats_base": oracle_acc > base_acc,
    }


def save_utility_table(rows: List[UtilityRow], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            entry = {
                "item_idx": r.item_idx,
                "domain": r.domain,
                "question_hash": r.question_hash,
                "gold_label": r.gold_label,
                "gold_idx": r.gold_idx,
                "oracle_candidate": r.oracle_candidate,
                "oracle_correct": r.oracle_correct,
                "base_correct": r.base_correct,
                "oracle_gain_over_base": r.oracle_gain_over_base,
                "candidate_correctness": {
                    k: v.correct for k, v in r.candidates.items()
                },
                "candidate_gold_logprobs": {
                    k: v.gold_logprob for k, v in r.candidates.items()
                },
            }
            f.write(json.dumps(entry) + "\n")
