#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from datetime import datetime, timezone


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
DEFAULT_INPUT = "methods/01_adathink/results/per_sample_Qwen3_8B_20260227_140410.csv"
ACTIONS = [64, 128, 256]


def to_int(v):
    try:
        return int(float(v))
    except Exception:
        return 0


def to_float(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def utility(row, budget, lambda_cost):
    c = to_int(row.get(f"fixed_{budget}_correct", 0))
    t = to_float(row.get(f"fixed_{budget}_tokens", 0.0))
    return c - lambda_cost * (t / 256.0)


def best_action_label(row, lambda_cost):
    best_b = ACTIONS[0]
    best_u = utility(row, best_b, lambda_cost)
    for b in ACTIONS[1:]:
        u = utility(row, b, lambda_cost)
        if u > best_u:
            best_u = u
            best_b = b
    return ACTIONS.index(best_b)


def text_features(text, dim):
    vec = [0.0] * dim
    toks = TOKEN_RE.findall((text or "").lower())
    if not toks:
        vec[0] = 1.0
        return vec
    for tok in toks:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % dim
        vec[h] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def softmax(scores):
    m = max(scores)
    ex = [math.exp(s - m) for s in scores]
    z = sum(ex)
    return [v / z for v in ex]


def train_full_rank(xs, ys, dim, k, epochs, lr, l2, seed):
    rnd = random.Random(seed)
    w = [[0.0 for _ in range(k)] for _ in range(dim)]
    idx = list(range(len(xs)))
    for _ in range(epochs):
        rnd.shuffle(idx)
        for i in idx:
            x = xs[i]
            y = ys[i]
            scores = [sum(w[d][j] * x[d] for d in range(dim)) for j in range(k)]
            p = softmax(scores)
            for j in range(k):
                g = p[j] - (1.0 if j == y else 0.0)
                for d in range(dim):
                    w[d][j] -= lr * (g * x[d] + l2 * w[d][j])
    return w


def predict_full_rank(w, x, dim, k):
    scores = [sum(w[d][j] * x[d] for d in range(dim)) for j in range(k)]
    p = softmax(scores)
    return max(range(k), key=lambda j: p[j])


def train_low_rank(xs, ys, dim, k, rank, epochs, lr, l2, seed):
    rnd = random.Random(seed)
    u = [[(rnd.random() - 0.5) * 0.01 for _ in range(rank)] for _ in range(dim)]
    v = [[(rnd.random() - 0.5) * 0.01 for _ in range(k)] for _ in range(rank)]
    idx = list(range(len(xs)))

    for _ in range(epochs):
        rnd.shuffle(idx)
        for i in idx:
            x = xs[i]
            y = ys[i]

            hidden = [sum(u[d][r] * x[d] for d in range(dim)) for r in range(rank)]
            scores = [sum(hidden[r] * v[r][j] for r in range(rank)) for j in range(k)]
            p = softmax(scores)
            grad_scores = [p[j] - (1.0 if j == y else 0.0) for j in range(k)]

            grad_hidden = [sum(grad_scores[j] * v[r][j] for j in range(k)) for r in range(rank)]

            for r in range(rank):
                for j in range(k):
                    v[r][j] -= lr * (hidden[r] * grad_scores[j] + l2 * v[r][j])
            for d in range(dim):
                xd = x[d]
                if xd == 0.0:
                    continue
                for r in range(rank):
                    u[d][r] -= lr * (xd * grad_hidden[r] + l2 * u[d][r])
    return u, v


def predict_low_rank(u, v, x, dim, k, rank):
    hidden = [sum(u[d][r] * x[d] for d in range(dim)) for r in range(rank)]
    scores = [sum(hidden[r] * v[r][j] for r in range(rank)) for j in range(k)]
    p = softmax(scores)
    return max(range(k), key=lambda j: p[j])


def evaluate(rows, pred_fn, lambda_cost):
    n = max(1, len(rows))
    action_match = 0
    total_correct = 0
    total_tokens = 0.0
    total_utility = 0.0
    for r in rows:
        y = best_action_label(r, lambda_cost)
        pred = pred_fn(r)
        if pred == y:
            action_match += 1
        b = ACTIONS[pred]
        total_correct += to_int(r.get(f"fixed_{b}_correct", 0))
        t = to_float(r.get(f"fixed_{b}_tokens", 0.0))
        total_tokens += t
        total_utility += to_int(r.get(f"fixed_{b}_correct", 0)) - lambda_cost * (t / 256.0)
    return {
        "action_match_rate": action_match / n,
        "accuracy": total_correct / n,
        "avg_tokens": total_tokens / n,
        "avg_utility": total_utility / n,
    }


def main():
    ap = argparse.ArgumentParser(description="Text2Subspace pilot with low-rank policy head")
    ap.add_argument("--input_csv", type=str, default=DEFAULT_INPUT)
    ap.add_argument("--output_dir", type=str, default="methods/05_text2subspace/results")
    ap.add_argument("--lambda_cost", type=float, default=0.15)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    with open(args.input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_csv}")

    train = [r for r in rows if (to_int(r.get("idx", 0)) % 5) != 0]
    test = [r for r in rows if (to_int(r.get("idx", 0)) % 5) == 0]
    if not test:
        test = rows[-max(1, len(rows) // 5) :]
        train = rows[: len(rows) - len(test)]

    x_train = [text_features(r.get("question", ""), args.dim) for r in train]
    y_train = [best_action_label(r, args.lambda_cost) for r in train]

    full = train_full_rank(
        x_train, y_train, dim=args.dim, k=3, epochs=args.epochs, lr=args.lr, l2=1e-4, seed=args.seed
    )
    low_u, low_v = train_low_rank(
        x_train,
        y_train,
        dim=args.dim,
        k=3,
        rank=args.rank,
        epochs=args.epochs,
        lr=args.lr,
        l2=1e-4,
        seed=args.seed + 101,
    )

    full_eval = evaluate(
        test,
        pred_fn=lambda r: predict_full_rank(full, text_features(r.get("question", ""), args.dim), args.dim, 3),
        lambda_cost=args.lambda_cost,
    )
    low_eval = evaluate(
        test,
        pred_fn=lambda r: predict_low_rank(
            low_u, low_v, text_features(r.get("question", ""), args.dim), args.dim, 3, args.rank
        ),
        lambda_cost=args.lambda_cost,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, f"text2subspace_pilot_{ts}.json")
    result = {
        "meta": {
            "timestamp_utc": ts,
            "input_csv": args.input_csv,
            "train_size": len(train),
            "test_size": len(test),
            "lambda_cost": args.lambda_cost,
            "dim": args.dim,
            "rank": args.rank,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
        },
        "full_rank_test": full_eval,
        "low_rank_test": low_eval,
        "delta_low_minus_full": {
            "action_match_rate": low_eval["action_match_rate"] - full_eval["action_match_rate"],
            "accuracy": low_eval["accuracy"] - full_eval["accuracy"],
            "avg_tokens": low_eval["avg_tokens"] - full_eval["avg_tokens"],
            "avg_utility": low_eval["avg_utility"] - full_eval["avg_utility"],
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
