"""CARR multi-term training objective + MCQ calibration utilities.

Round 3 Task 4+5: moves CARR training off LM loss onto a downstream-aligned
objective with calibrated reliability labels.

Loss terms (all weighted by configs/carr_minimal.yaml):
  L_task    : CE on option logprobs against gold option (MCQ calib items).
  L_base_KL : KL(CARR_option_probs || base_option_probs) — preserve base on calib.
  L_conf    : sum_module  g_0·g_1·interference_ratio — penalizes co-activation on
              conflicting modules (module-level conflict prior).
  L_sparse  : mean gate entropy across hooked modules.
  L_cal     : BCE(reliability_head_out, adapter_correctness_label) on calib items.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCQ calibration items
# ---------------------------------------------------------------------------

_MCQ_CALIB_MAP = {
    # domain -> (local_name, train_split, type)
    "science": ("arc_challenge", "train", "arc"),
    "medical": ("medmcqa", "train", "medmcqa"),
}


@dataclass
class CalibItem:
    domain: str
    row_idx: int
    prompt: str
    prompt_ids: torch.LongTensor  # (L,)
    option_labels: List[str]      # e.g. ["A","B","C","D"]
    option_token_ids: List[List[int]]  # tokens-to-append for each option
    gold_idx: int                 # 0..n_options-1


def _format_arc(ex) -> Tuple[str, List[str], List[str], int]:
    q = ex["question"]
    labels = list(ex["choices"]["label"])
    texts = list(ex["choices"]["text"])
    gold_label = ex["answerKey"]
    try:
        gold_idx = labels.index(gold_label)
    except ValueError:
        # arc uses 1..5 sometimes; coerce
        gold_idx = 0
    option_labels = ["A", "B", "C", "D"][: len(labels)]
    options_str = "\n".join(f"{l}) {t}" for l, t in zip(option_labels, texts))
    prompt = f"Question: {q}\n{options_str}\nAnswer:"
    return prompt, option_labels, [f" {l}" for l in option_labels], gold_idx


def _format_medmcqa(ex) -> Tuple[str, List[str], List[str], int]:
    q = ex["question"]
    opts = [ex.get(f"op{c}", "") for c in "abcd"]
    option_labels = ["A", "B", "C", "D"]
    options_str = "\n".join(f"{l}) {o}" for l, o in zip(option_labels, opts))
    gold_idx = int(ex.get("cop", 0))
    gold_idx = max(0, min(gold_idx, 3))
    prompt = f"Question: {q}\n{options_str}\nAnswer:"
    return prompt, option_labels, [f" {l}" for l in option_labels], gold_idx


_FORMATTERS = {"arc": _format_arc, "medmcqa": _format_medmcqa}


def build_mcq_calib_items(
    domains: Sequence[str],
    dataset_dir: str,
    n_per_domain: int,
    tokenizer,
    sample_seed: int = 42,
    max_length: int = 384,
) -> List[CalibItem]:
    from datasets import load_from_disk

    items: List[CalibItem] = []
    for d in domains:
        if d not in _MCQ_CALIB_MAP:
            logger.warning("No MCQ calib formatter for domain=%s; skipping L_task/L_cal", d)
            continue
        local, split, etype = _MCQ_CALIB_MAP[d]
        ds = load_from_disk(os.path.join(dataset_dir, local))
        split = split if split in ds else list(ds.keys())[0]
        sub = ds[split].shuffle(seed=sample_seed).select(
            range(min(n_per_domain, len(ds[split])))
        )
        fmt = _FORMATTERS[etype]
        for i, ex in enumerate(sub):
            try:
                prompt, opt_labels, opt_strs, gold_idx = fmt(ex)
            except Exception as e:
                logger.debug("Skip row %d %s: %s", i, d, e)
                continue
            prompt_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=max_length
            ).input_ids[0]
            opt_token_ids = [
                tokenizer(s, add_special_tokens=False).input_ids for s in opt_strs
            ]
            items.append(CalibItem(
                domain=d, row_idx=i, prompt=prompt, prompt_ids=prompt_ids,
                option_labels=opt_labels, option_token_ids=opt_token_ids,
                gold_idx=gold_idx,
            ))
    logger.info("Built %d MCQ calib items across domains=%s", len(items), list(domains))
    return items


# ---------------------------------------------------------------------------
# Option logprob computation
# ---------------------------------------------------------------------------


def compute_option_logprobs(
    model, item: CalibItem, device: str = "cuda",
    require_grad: bool = False,
) -> torch.Tensor:
    """Return (n_options,) length-normalized sum log-probs.

    When require_grad=True, gradient flows through model (used for CARR training).
    """
    prompt_ids = item.prompt_ids.unsqueeze(0).to(device)
    scores = []
    ctx = torch.enable_grad() if require_grad else torch.no_grad()
    with ctx:
        for opt_ids in item.option_token_ids:
            if len(opt_ids) == 0:
                scores.append(torch.tensor(0.0, device=device))
                continue
            opt_tensor = torch.tensor([opt_ids], device=device, dtype=prompt_ids.dtype)
            full = torch.cat([prompt_ids, opt_tensor], dim=-1)
            logits = model(full).logits
            # predict tokens at positions [L-1, L, ..., L+K-2] for targets [L, ..., L+K-1]
            target = full[:, prompt_ids.shape[1]:]
            log_probs = F.log_softmax(
                logits[:, prompt_ids.shape[1] - 1:-1, :].float(), dim=-1
            )
            s = log_probs.gather(-1, target.unsqueeze(-1)).sum() / max(len(opt_ids), 1)
            scores.append(s)
    return torch.stack(scores)


# ---------------------------------------------------------------------------
# Reliability labels
# ---------------------------------------------------------------------------


@dataclass
class ReliabilityLabels:
    n_items: int
    n_adapters: int
    # (n_items, n_adapters) binary correctness under single-adapter inference
    correctness: torch.Tensor
    # (n_items, n_adapters) continuous advantage over base (adapter - base) on gold logprob
    advantage: torch.Tensor
    # (n_items,) base correctness, for reference
    base_correct: torch.Tensor

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "n_items": self.n_items,
            "n_adapters": self.n_adapters,
            "correctness": self.correctness,
            "advantage": self.advantage,
            "base_correct": self.base_correct,
        }, path)

    @classmethod
    def load(cls, path: str) -> "ReliabilityLabels":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            n_items=int(d["n_items"]), n_adapters=int(d["n_adapters"]),
            correctness=d["correctness"], advantage=d["advantage"],
            base_correct=d["base_correct"],
        )


def precompute_reliability_labels(
    base_model_path: str,
    adapter_paths: Sequence[str],
    tokenizer,
    calib_items: List[CalibItem],
    device: str = "cuda",
    save_path: Optional[str] = None,
) -> ReliabilityLabels:
    """Run base + each PEFT adapter on each calib item, build (n_items, n_adapters) labels."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    n_items = len(calib_items)
    n_adapters = len(adapter_paths)

    logger.info("Loading base model for reliability precompute...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa", trust_remote_code=True)
    base.eval()

    base_gold_lp = torch.zeros(n_items)
    base_correct = torch.zeros(n_items, dtype=torch.bool)

    logger.info("  base pass ...")
    for i, item in enumerate(calib_items):
        lp = compute_option_logprobs(base, item, device=device, require_grad=False)
        base_gold_lp[i] = lp[item.gold_idx].item()
        base_correct[i] = bool(int(lp.argmax().item()) == item.gold_idx)

    correctness = torch.zeros(n_items, n_adapters, dtype=torch.bool)
    advantage = torch.zeros(n_items, n_adapters)

    for k, ap in enumerate(adapter_paths):
        logger.info("  adapter %d / %d : %s", k + 1, n_adapters, ap)
        peft = PeftModel.from_pretrained(base, ap)
        peft.eval()
        for i, item in enumerate(calib_items):
            lp = compute_option_logprobs(peft, item, device=device, require_grad=False)
            correctness[i, k] = bool(int(lp.argmax().item()) == item.gold_idx)
            advantage[i, k] = lp[item.gold_idx].item() - base_gold_lp[i].item()
        try:
            base = peft.unload()
        except Exception:
            base = peft.merge_and_unload()
        del peft
        torch.cuda.empty_cache()

    logger.info("Reliability labels: base_acc=%.3f  adapter_acc=%s",
                base_correct.float().mean().item(),
                correctness.float().mean(0).tolist())
    labels = ReliabilityLabels(
        n_items=n_items, n_adapters=n_adapters,
        correctness=correctness, advantage=advantage, base_correct=base_correct,
    )
    if save_path:
        labels.save(save_path)
        logger.info("Saved reliability labels -> %s", save_path)
    return labels


# ---------------------------------------------------------------------------
# Multi-term loss
# ---------------------------------------------------------------------------


@dataclass
class LossWeights:
    task: float = 1.0
    base_kl: float = 0.1
    conflict: float = 0.05
    sparse: float = 0.01
    calibration: float = 0.1


def carr_multi_term_loss(
    carr_option_logprobs: torch.Tensor,      # (n_options,) with grad
    base_option_logprobs: torch.Tensor,      # (n_options,) no grad, precomputed
    gold_idx: int,
    last_gates: Optional[Dict[str, torch.Tensor]],  # per-module (B,S,C) with grad
    module_conflict_scores: Optional[Dict[str, torch.Tensor]],
    reliability_pred: Optional[torch.Tensor],  # (B, n_adapters) with grad
    reliability_label: Optional[torch.Tensor], # (n_adapters,) binary
    weights: LossWeights,
    adapter_start_idx: int = 2,  # depends on use_base_fallback
    static_idx: int = 1,
) -> Dict[str, torch.Tensor]:
    device = carr_option_logprobs.device
    terms: Dict[str, torch.Tensor] = {}

    # L_task: -log P(gold) where P is softmax over options
    logp_options = F.log_softmax(carr_option_logprobs, dim=-1)
    terms["L_task"] = -logp_options[gold_idx]

    # L_base_KL: KL(carr || base) on option distribution
    carr_p = F.log_softmax(carr_option_logprobs, dim=-1)
    base_p = F.log_softmax(base_option_logprobs.to(device), dim=-1)
    terms["L_base_KL"] = F.kl_div(carr_p, base_p, reduction="batchmean", log_target=True)

    # L_conf: sum_module mean(g_adapter_0)·mean(g_adapter_1)·interference_ratio
    # differentiable through gates
    if module_conflict_scores and last_gates:
        per_mod = []
        for mod, gates in last_gates.items():
            if mod not in module_conflict_scores:
                continue
            # gates (B, S, n_choices). adapter_0 at adapter_start_idx, adapter_1 at +1
            g0 = gates[:, :, adapter_start_idx].mean()
            g1 = gates[:, :, adapter_start_idx + 1].mean() if gates.shape[-1] > adapter_start_idx + 1 else torch.zeros((), device=device)
            interf = module_conflict_scores[mod][0].to(device).float()
            per_mod.append(g0 * g1 * interf)
        if per_mod:
            terms["L_conf"] = torch.stack(per_mod).mean()
        else:
            terms["L_conf"] = torch.tensor(0.0, device=device)
    else:
        terms["L_conf"] = torch.tensor(0.0, device=device)

    # L_sparse: mean gate entropy across modules (differentiable)
    if last_gates:
        ent = []
        for gates in last_gates.values():
            e = -(gates * (gates + 1e-10).log()).sum(-1).mean()
            ent.append(e)
        if ent:
            terms["L_sparse"] = torch.stack(ent).mean()
        else:
            terms["L_sparse"] = torch.tensor(0.0, device=device)
    else:
        terms["L_sparse"] = torch.tensor(0.0, device=device)

    # L_cal: BCE between reliability head output and binary correctness label
    if reliability_pred is not None and reliability_label is not None:
        pred = reliability_pred.clamp(1e-4, 1 - 1e-4)
        # reliability_pred: (B, n_adapters), label: (n_adapters,) — broadcast B
        label = reliability_label.float().to(device)
        while label.dim() < pred.dim():
            label = label.unsqueeze(0).expand_as(pred) if label.numel() == pred.shape[-1] else label.unsqueeze(0)
        terms["L_cal"] = F.binary_cross_entropy(pred, label, reduction="mean")
    else:
        terms["L_cal"] = torch.tensor(0.0, device=device)

    total = (
        weights.task * terms["L_task"]
        + weights.base_kl * terms["L_base_KL"]
        + weights.conflict * terms["L_conf"]
        + weights.sparse * terms["L_sparse"]
        + weights.calibration * terms["L_cal"]
    )
    terms["total"] = total
    return terms


def compute_ece(predictions: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """Expected Calibration Error for binary predictions."""
    preds = predictions.flatten().float()
    tgts = targets.flatten().float()
    ece = 0.0
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1] + (1e-6 if i == n_bins - 1 else 0))
        if mask.sum() == 0:
            continue
        acc = tgts[mask].mean().item()
        conf = preds[mask].mean().item()
        ece += (mask.float().mean().item()) * abs(acc - conf)
    return float(ece)


def compute_brier(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return float(((predictions - targets.float()) ** 2).mean().item())


# ---------------------------------------------------------------------------
# Shared multi-term training loop (used by train_carr_router.py and eval_carr.py)
# ---------------------------------------------------------------------------


def train_router_multiterm(
    model,
    hook,           # CARRHook (already attached)
    router,         # ConflictAwareResidualRouter
    calib_items: List[CalibItem],
    base_option_lps: torch.Tensor,         # (n_items, n_options)
    rel_labels: ReliabilityLabels,
    module_conflict_scores: Dict[str, torch.Tensor],
    weights: LossWeights,
    max_steps: int,
    lr: float,
    device: str = "cuda",
    overfit_one_batch: bool = False,
    log_every: int = 10,
    log_fn=None,
) -> List[Dict]:
    """One shared training loop. Returns loss history."""
    if log_fn is None:
        log_fn = lambda *a, **kw: None  # noqa: E731

    optimizer = torch.optim.Adam(router.parameters(), lr=lr)
    router.train()
    history: List[Dict] = []

    n_items = len(calib_items)
    train_order = list(range(n_items))
    if overfit_one_batch:
        train_order = train_order[: max(1, min(n_items, 4))]

    for step in range(max_steps):
        item_idx = train_order[step % len(train_order)]
        item = calib_items[item_idx]

        option_lps = []
        for opt_ids in item.option_token_ids:
            hook.clear_step_buffers()
            prompt_ids = item.prompt_ids.unsqueeze(0).to(device)
            opt_tensor = torch.tensor(
                [opt_ids], device=device, dtype=prompt_ids.dtype)
            full = torch.cat([prompt_ids, opt_tensor], dim=-1)
            logits = model(full).logits
            target = full[:, prompt_ids.shape[1]:]
            log_probs = F.log_softmax(
                logits[:, prompt_ids.shape[1] - 1:-1, :].float(), dim=-1)
            lp = log_probs.gather(-1, target.unsqueeze(-1)).sum() / max(len(opt_ids), 1)
            option_lps.append(lp)
        carr_lps = torch.stack(option_lps)

        last_gates = {mod: g for mod, g in hook.last_gates.items()}
        rel_pred = hook.mean_reliability_across_modules()

        losses = carr_multi_term_loss(
            carr_option_logprobs=carr_lps,
            base_option_logprobs=base_option_lps[item_idx],
            gold_idx=item.gold_idx,
            last_gates=last_gates,
            module_conflict_scores=module_conflict_scores,
            reliability_pred=rel_pred,
            reliability_label=rel_labels.correctness[item_idx]
            if rel_labels is not None else None,
            weights=weights,
            adapter_start_idx=router.adapter_start_idx,
            static_idx=router.static_idx,
        )
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        rec = {
            "step": step,
            "total": float(losses["total"].item()),
            "L_task": float(losses["L_task"].item()),
            "L_base_KL": float(losses["L_base_KL"].item()),
            "L_conf": float(losses["L_conf"].item()),
            "L_sparse": float(losses["L_sparse"].item()),
            "L_cal": float(losses["L_cal"].item()),
            "item_idx": item_idx, "domain": item.domain,
        }
        history.append(rec)
        if step % log_every == 0:
            log_fn("    step=%d total=%.4f task=%.4f kl=%.4f conf=%.4f sparse=%.4f cal=%.4f",
                   step, rec["total"], rec["L_task"], rec["L_base_KL"],
                   rec["L_conf"], rec["L_sparse"], rec["L_cal"])
        if overfit_one_batch and step >= 49:
            break
    return history


def precompute_base_option_logprobs(
    model, calib_items: List[CalibItem], device: str = "cuda"
) -> torch.Tensor:
    """Compute (n_items, n_options) length-normalized sum logprobs under base model.

    Caller must detach any CARR hooks before calling.
    """
    model.eval()
    base_lps = []
    with torch.no_grad():
        for it in calib_items:
            lp = compute_option_logprobs(model, it, device=device, require_grad=False)
            base_lps.append(lp.cpu())
    return torch.stack(base_lps)

