"""Token-level influence quality analysis.

Usage: python scripts/analyze_token_influence.py <trace_dir> --tokenizer <model_path>

Analyzes at the TOKEN level (not response level):
1. Position-dependent patterns: do early tokens just reflect response-level signal?
2. Same-token-different-response: for identical tokens at same position across
   responses to the same prompt, how much does influence vary?
3. Token type discrimination: are math tokens, reasoning words, filler tokens
   scored differently?
4. Within-response ranking: ignoring absolute values, does the ranking of tokens
   by influence correlate with meaningful patterns?
5. Late-response tokens: tokens near the final answer should have cleaner signal
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_trace(trace_dir, step=None):
    steps_dir = Path(trace_dir) / "steps"
    if step:
        return np.load(steps_dir / f"step_{step:06d}.npz", allow_pickle=True)
    npz_files = sorted(steps_dir.glob("step_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No step files in {steps_dir}")
    return np.load(npz_files[-1], allow_pickle=True)


def position_analysis(infl, acc, gids, responses, tokenizer):
    """Check if influence at early positions is just response-level signal."""
    print("=" * 80)
    print("POSITION-DEPENDENT ANALYSIS")
    print("=" * 80)
    print("If early tokens just reflect response-level correctness, their")
    print("mean influence should correlate perfectly with accepted/rejected label.")
    print()

    # Split tokens into position buckets
    buckets = [(0, 20, "pos 0-20 (start)"),
               (20, 50, "pos 20-50"),
               (50, 100, "pos 50-100"),
               (100, 300, "pos 100-300 (mid)"),
               (300, 1000, "pos 300-1000"),
               (1000, 99999, "pos 1000+ (late)")]

    for lo, hi, label in buckets:
        acc_scores = []
        rej_scores = []
        for i in range(len(infl)):
            row = infl[i].astype(np.float32)
            valid = ~np.isnan(row)
            # Tokens in this position range
            mask = np.zeros(len(row), dtype=bool)
            mask[lo:min(hi, len(row))] = True
            mask &= valid
            if mask.sum() == 0:
                continue
            mean_s = row[mask].mean()
            if acc[i]:
                acc_scores.append(mean_s)
            else:
                rej_scores.append(mean_s)

        if not acc_scores or not rej_scores:
            continue
        acc_m = np.mean(acc_scores)
        rej_m = np.mean(rej_scores)
        # AUC
        n_correct = 0
        n_total = 0
        for a in acc_scores:
            for r in rej_scores:
                n_total += 1
                if a > r:
                    n_correct += 1
                elif a == r:
                    n_correct += 0.5
        auc = n_correct / max(n_total, 1)
        print(f"  {label:25s}: acc_mean={acc_m:+.4f}, rej_mean={rej_m:+.4f}, "
              f"AUC={auc:.3f}, gap={acc_m - rej_m:+.4f}")


def same_token_analysis(infl, acc, gids, responses, tokenizer):
    """For same prompt, same token at same position: how much does influence vary?"""
    print()
    print("=" * 80)
    print("SAME-TOKEN-SAME-POSITION ANALYSIS (within prompt group)")
    print("=" * 80)
    print("For tokens at the start of responses (shared prompt continuation),")
    print("influence should be similar if it reflects token-level credit.")
    print("Large variance → signal is trajectory-level, not token-level.")
    print()

    unique_gids = np.unique(gids)
    all_within_var = []
    all_between_var = []

    for gid in unique_gids:
        mask = gids == gid
        g_infl = infl[mask]
        g_acc = acc[mask]
        g_resp = responses[mask]
        n = mask.sum()
        if n < 4:
            continue

        # Find common prefix length (same tokens across all responses)
        min_len = min(len(r) for r in g_resp)
        common_prefix_len = 0
        for pos in range(min(50, min_len)):
            tokens_at_pos = set(int(g_resp[j][pos]) for j in range(n))
            if len(tokens_at_pos) == 1:
                common_prefix_len = pos + 1
            else:
                break

        if common_prefix_len < 5:
            continue

        # For common prefix tokens: compute within-group variance of influence
        prefix_scores = []
        for j in range(n):
            row = g_infl[j].astype(np.float32)
            valid = ~np.isnan(row[:common_prefix_len])
            if valid.sum() < 3:
                continue
            prefix_scores.append(row[:common_prefix_len][valid].mean())

        if len(prefix_scores) < 4:
            continue

        prefix_scores = np.array(prefix_scores)
        acc_prefix = prefix_scores[g_acc[:len(prefix_scores)]]
        rej_prefix = prefix_scores[~g_acc[:len(prefix_scores)]]

        within_var = prefix_scores.var()
        total_mean = prefix_scores.mean()

        print(f"  Group {gid} (prefix_len={common_prefix_len}, {n} resp): "
              f"mean={total_mean:+.4f}, var={within_var:.6f}, "
              f"acc_mean={acc_prefix.mean():+.4f}, rej_mean={rej_prefix.mean():+.4f}"
              if len(acc_prefix) > 0 and len(rej_prefix) > 0 else
              f"  Group {gid} (prefix_len={common_prefix_len}, {n} resp): "
              f"mean={total_mean:+.4f}, var={within_var:.6f}")


def token_type_analysis(infl, acc, responses, tokenizer, n_sample=100):
    """Analyze influence by token type."""
    print()
    print("=" * 80)
    print("TOKEN TYPE ANALYSIS")
    print("=" * 80)

    # Define token categories
    math_tokens = set()
    reasoning_tokens = set()
    filler_tokens = set()
    number_tokens = set()

    # Build category sets from tokenizer vocab
    vocab = tokenizer.get_vocab()
    for token_str, token_id in vocab.items():
        t = token_str.lower().replace("▁", " ").replace("Ġ", " ").strip()
        if t in ("+", "-", "*", "/", "=", "^", "(", ")", "{", "}", "[", "]",
                 "\\", "frac", "sqrt", "sum", "int", "lim", "pi", "times",
                 "div", "cdot", "leq", "geq", "neq", "approx"):
            math_tokens.add(token_id)
        elif t in ("therefore", "because", "since", "thus", "hence", "so",
                    "if", "then", "let", "suppose", "assume", "given",
                    "step", "first", "next", "finally", "note", "recall"):
            reasoning_tokens.add(token_id)
        elif t in ("the", "a", "an", "is", "are", "was", "were", "be",
                    "to", "of", "and", "in", "that", "it", "for", "on"):
            filler_tokens.add(token_id)
        if t.isdigit():
            number_tokens.add(token_id)

    categories = {
        "math_symbols": math_tokens,
        "reasoning_words": reasoning_tokens,
        "filler_words": filler_tokens,
        "numbers": number_tokens,
    }

    n_sample = min(n_sample, len(infl))
    cat_scores = {cat: {"acc": [], "rej": []} for cat in categories}
    cat_scores["other"] = {"acc": [], "rej": []}

    for i in range(n_sample):
        row = infl[i].astype(np.float32)
        resp = responses[i]
        is_acc = bool(acc[i])
        valid = ~np.isnan(row)
        label = "acc" if is_acc else "rej"

        for j in range(min(len(resp), len(row))):
            if not valid[j]:
                continue
            tid = int(resp[j])
            score = float(row[j])
            categorized = False
            for cat, token_set in categories.items():
                if tid in token_set:
                    cat_scores[cat][label].append(score)
                    categorized = True
                    break
            if not categorized:
                cat_scores["other"][label].append(score)

    print(f"{'Category':>20s} | {'N(acc)':>8s} {'Mean(acc)':>10s} | {'N(rej)':>8s} {'Mean(rej)':>10s} | {'Gap':>8s}")
    print("-" * 80)
    for cat in list(categories.keys()) + ["other"]:
        acc_s = cat_scores[cat]["acc"]
        rej_s = cat_scores[cat]["rej"]
        if not acc_s or not rej_s:
            continue
        acc_m = np.mean(acc_s)
        rej_m = np.mean(rej_s)
        print(f"{cat:>20s} | {len(acc_s):>8d} {acc_m:>+10.4f} | {len(rej_s):>8d} {rej_m:>+10.4f} | {acc_m-rej_m:>+8.4f}")


def within_response_ranking(infl, acc, gids, responses, tokenizer, n_sample=30):
    """Within each response, do high-influence tokens have meaningful patterns?"""
    print()
    print("=" * 80)
    print("WITHIN-RESPONSE TOKEN RANKING")
    print("=" * 80)
    print("For each response: show top-5 highest and lowest influence tokens.")
    print("Look for: do high-influence tokens correspond to key reasoning steps?")
    print()

    n_sample = min(n_sample, len(infl))
    # Pick a few interesting responses: one high-reward accepted, one rejected
    for label, indices in [("ACCEPTED", np.where(acc)[0][:3]),
                           ("REJECTED", np.where(~acc)[0][:3])]:
        for idx in indices:
            row = infl[idx].astype(np.float32)
            resp = responses[idx]
            valid = ~np.isnan(row)
            n_valid = valid.sum()
            if n_valid < 20:
                continue

            scores = row[:n_valid]
            tokens = resp[:n_valid]
            sorted_idx = np.argsort(scores)

            mean_s = scores.mean()
            print(f"[{label}] idx={idx}, gid={gids[idx]}, n_tok={n_valid}, mean_infl={mean_s:+.4f}")

            print(f"  TOP-5 (highest influence):")
            for k in sorted_idx[-5:][::-1]:
                tok_str = tokenizer.decode([tokens[k]])
                print(f"    pos={k:4d} score={scores[k]:+7.3f} {repr(tok_str)}")

            print(f"  BOTTOM-5 (lowest influence):")
            for k in sorted_idx[:5]:
                tok_str = tokenizer.decode([tokens[k]])
                print(f"    pos={k:4d} score={scores[k]:+7.3f} {repr(tok_str)}")

            # Show first 30 tokens inline
            print(f"  First 30 tokens:")
            for j in range(min(30, n_valid)):
                tok_str = tokenizer.decode([tokens[j]])
                s = scores[j]
                print(f"    [{s:+6.2f}] {repr(tok_str)}")
            print()


def late_token_analysis(infl, acc, gids, responses, tokenizer):
    """Analyze tokens near the end of responses (closer to final answer)."""
    print()
    print("=" * 80)
    print("LATE TOKEN ANALYSIS (last 50 tokens of each response)")
    print("=" * 80)
    print("Tokens near the end should have cleaner per-token signal")
    print("(less cross-position gradient contamination).")
    print()

    acc_late = []
    rej_late = []
    for i in range(len(infl)):
        row = infl[i].astype(np.float32)
        valid = ~np.isnan(row)
        n_valid = valid.sum()
        if n_valid < 50:
            continue
        # Last 50 valid tokens
        valid_indices = np.where(valid)[0]
        late_indices = valid_indices[-50:]
        late_scores = row[late_indices]
        mean_late = late_scores.mean()
        if acc[i]:
            acc_late.append(mean_late)
        else:
            rej_late.append(mean_late)

    if not acc_late or not rej_late:
        print("  Not enough data")
        return

    acc_late = np.array(acc_late)
    rej_late = np.array(rej_late)

    # AUC
    n_correct = sum(1 for a in acc_late for r in rej_late if a > r)
    n_tie = sum(0.5 for a in acc_late for r in rej_late if a == r)
    auc = (n_correct + n_tie) / (len(acc_late) * len(rej_late))

    print(f"  Late-token mean (acc): {acc_late.mean():+.4f} ± {acc_late.std():.4f}")
    print(f"  Late-token mean (rej): {rej_late.mean():+.4f} ± {rej_late.std():.4f}")
    print(f"  Late-token AUC: {auc:.4f}")
    print(f"  (Compare with full-response AUC to see if late tokens are cleaner)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()

    npz = load_trace(args.trace_dir, args.step)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    infl = npz["influence"]
    acc = npz["accepted"]
    gids = npz["group_id"]
    responses = npz["responses"]

    print(f"Loaded: {len(infl)} rows, {acc.sum()} acc, {(~acc).sum()} rej, "
          f"{len(np.unique(gids))} groups")
    print()

    position_analysis(infl, acc, gids, responses, tokenizer)
    same_token_analysis(infl, acc, gids, responses, tokenizer)
    token_type_analysis(infl, acc, responses, tokenizer)
    within_response_ranking(infl, acc, gids, responses, tokenizer)
    late_token_analysis(infl, acc, gids, responses, tokenizer)


if __name__ == "__main__":
    main()
