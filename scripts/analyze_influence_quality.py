"""Comprehensive influence trace quality analysis.

Usage: python scripts/analyze_influence_quality.py <trace_dir> [--tokenizer <model_path>]

Analyzes:
1. Response-level discrimination: Can influence scores distinguish accepted from rejected?
2. Per-prompt sign consistency: Does accepted have higher mean influence than rejected within each group?
3. Signal-to-noise ratio per response
4. Inter-response correlation within prompt groups
5. Token-level patterns: Which token types get high/low influence?
6. Score distribution and normality
7. AUC for response-level binary classification
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_trace(trace_dir: str, step: int | None = None):
    """Load the latest (or specified) step's NPZ trace."""
    steps_dir = Path(trace_dir) / "steps"
    if step is not None:
        npz_path = steps_dir / f"step_{step:06d}.npz"
    else:
        npz_files = sorted(steps_dir.glob("step_*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No step files in {steps_dir}")
        npz_path = npz_files[-1]
    print(f"Loading: {npz_path}")
    return np.load(npz_path, allow_pickle=True)


def response_level_stats(infl, acc, gids):
    """Compute per-response mean influence and analyze discrimination."""
    n = len(infl)
    means = np.zeros(n)
    stds = np.zeros(n)
    snrs = np.zeros(n)
    n_valid = np.zeros(n, dtype=int)

    for i in range(n):
        row = infl[i].astype(np.float32)
        valid = ~np.isnan(row)
        nv = valid.sum()
        n_valid[i] = nv
        if nv == 0:
            means[i] = stds[i] = snrs[i] = np.nan
            continue
        vals = row[valid]
        means[i] = vals.mean()
        stds[i] = vals.std()
        snrs[i] = abs(means[i]) / max(stds[i], 1e-8)

    return means, stds, snrs, n_valid


def compute_auc(labels, scores):
    """Simple AUC computation without sklearn."""
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Mann-Whitney U statistic
    n_pos, n_neg = len(pos), len(neg)
    correct = 0
    for p in pos:
        correct += (p > neg).sum() + 0.5 * (p == neg).sum()
    return correct / (n_pos * n_neg)


def per_prompt_analysis(infl, acc, gids, means):
    """Per-prompt group analysis."""
    unique_gids = np.unique(gids)
    results = []

    for gid in unique_gids:
        mask = gids == gid
        g_acc = acc[mask]
        g_means = means[mask]

        n_acc = g_acc.sum()
        n_rej = (~g_acc).sum()
        if n_acc == 0 or n_rej == 0:
            continue

        acc_m = g_means[g_acc].mean()
        rej_m = g_means[~g_acc].mean()
        sign_ok = acc_m > rej_m

        # Within-group AUC
        auc = compute_auc(g_acc, g_means)

        results.append({
            "gid": int(gid),
            "n_acc": int(n_acc),
            "n_rej": int(n_rej),
            "acc_mean": float(acc_m),
            "rej_mean": float(rej_m),
            "sign_ok": bool(sign_ok),
            "auc": float(auc),
            "gap": float(acc_m - rej_m),
        })

    return results


def inter_response_correlation(infl, acc, gids, max_groups=15, max_pairs=28):
    """Compute pairwise token-level correlation within prompt groups."""
    unique_gids = np.unique(gids)
    corr_results = []

    for gid in unique_gids[:max_groups]:
        mask = gids == gid
        g_infl = infl[mask]
        g_acc = acc[mask]
        n = mask.sum()
        if n < 2:
            continue

        valid_lens = [(~np.isnan(row.astype(np.float32))).sum() for row in g_infl]
        min_len = min(valid_lens)
        if min_len < 50:
            continue

        corrs = []
        pairs_checked = 0
        for i in range(min(8, n)):
            for j in range(i + 1, min(8, n)):
                if pairs_checked >= max_pairs:
                    break
                a = g_infl[i][:min_len].astype(np.float32)
                b = g_infl[j][:min_len].astype(np.float32)
                both = (~np.isnan(a)) & (~np.isnan(b))
                if both.sum() < 50:
                    continue
                av, bv = a[both], b[both]
                am, bm = av - av.mean(), bv - bv.mean()
                denom = np.sqrt((am**2).sum() * (bm**2).sum())
                if denom < 1e-8:
                    continue
                r = float((am * bm).sum() / denom)
                corrs.append(r)
                pairs_checked += 1

        if corrs:
            corr_results.append({
                "gid": int(gid),
                "mean_corr": float(np.mean(corrs)),
                "n_pairs": len(corrs),
                "n_acc": int(g_acc.sum()),
                "n_rej": int((~g_acc).sum()),
            })

    return corr_results


def token_type_analysis(infl, responses, acc, tokenizer=None):
    """Analyze influence by token type (requires tokenizer)."""
    if tokenizer is None:
        return None

    # Collect scores by token id
    token_scores = {}  # token_id -> list of (score, is_accepted)
    n_rows = min(50, len(infl))  # sample for speed

    for i in range(n_rows):
        row = infl[i].astype(np.float32)
        resp = responses[i]
        is_acc = bool(acc[i])
        valid = ~np.isnan(row)

        for j in range(min(len(resp), len(row))):
            if not valid[j]:
                continue
            tid = int(resp[j])
            if tid not in token_scores:
                token_scores[tid] = []
            token_scores[tid].append((float(row[j]), is_acc))

    # Aggregate: mean score per token type, split by accepted/rejected
    results = []
    for tid, pairs in token_scores.items():
        if len(pairs) < 5:
            continue
        scores = [p[0] for p in pairs]
        acc_scores = [p[0] for p in pairs if p[1]]
        rej_scores = [p[0] for p in pairs if not p[1]]

        token_str = tokenizer.decode([tid])
        results.append({
            "token_id": tid,
            "token": token_str,
            "count": len(pairs),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "acc_mean": float(np.mean(acc_scores)) if acc_scores else float("nan"),
            "rej_mean": float(np.mean(rej_scores)) if rej_scores else float("nan"),
        })

    results.sort(key=lambda x: abs(x["mean"]), reverse=True)
    return results


def print_report(npz, tokenizer=None):
    infl = npz["influence"]  # (n_rows, seq_len) float16
    acc = npz["accepted"]  # (n_rows,) bool
    gids = npz["group_id"]
    responses = npz["responses"]
    rewards = npz["reward"]

    n_rows = len(infl)
    n_acc = int(acc.sum())
    n_rej = int((~acc).sum())
    n_groups = len(np.unique(gids))

    print("=" * 80)
    print("INFLUENCE TRACE QUALITY REPORT")
    print("=" * 80)
    print(f"Rows: {n_rows} ({n_acc} accepted, {n_rej} rejected)")
    print(f"Prompt groups: {n_groups}")
    print(f"Mean reward: {rewards.mean():.3f}")

    # 1. Response-level stats
    means, stds, snrs, n_valid = response_level_stats(infl, acc, gids)

    print(f"\n--- RESPONSE-LEVEL STATISTICS ---")
    valid_mask = ~np.isnan(means)
    print(f"Valid responses: {valid_mask.sum()}/{n_rows}")
    print(f"Mean influence (acc):  {means[acc & valid_mask].mean():.4f} ± {means[acc & valid_mask].std():.4f}")
    print(f"Mean influence (rej):  {means[~acc & valid_mask].mean():.4f} ± {means[~acc & valid_mask].std():.4f}")
    print(f"SNR (acc): mean={snrs[acc & valid_mask].mean():.4f}, median={np.median(snrs[acc & valid_mask]):.4f}")
    print(f"SNR (rej): mean={snrs[~acc & valid_mask].mean():.4f}, median={np.median(snrs[~acc & valid_mask]):.4f}")

    # 2. Global AUC
    global_auc = compute_auc(acc[valid_mask], means[valid_mask])
    print(f"\n--- GLOBAL DISCRIMINATION ---")
    print(f"AUC (mean_influence predicts accepted): {global_auc:.4f}")
    print(f"  (0.5 = random, >0.6 = weak signal, >0.7 = moderate, >0.8 = strong)")

    # 3. Per-prompt analysis
    pp_results = per_prompt_analysis(infl, acc, gids, means)
    n_sign_ok = sum(1 for r in pp_results if r["sign_ok"])
    n_mixed = len(pp_results)
    per_prompt_aucs = [r["auc"] for r in pp_results]

    print(f"\n--- PER-PROMPT SIGN CONSISTENCY ---")
    print(f"Groups with correct sign (acc > rej): {n_sign_ok}/{n_mixed} = {n_sign_ok/max(n_mixed,1):.1%}")
    print(f"Per-prompt AUC: mean={np.mean(per_prompt_aucs):.4f}, median={np.median(per_prompt_aucs):.4f}")
    print(f"  Per-prompt AUC distribution:")
    for p in [10, 25, 50, 75, 90]:
        print(f"    P{p}: {np.percentile(per_prompt_aucs, p):.4f}")

    print(f"\n  Top 5 by AUC:")
    pp_sorted = sorted(pp_results, key=lambda x: x["auc"], reverse=True)
    for r in pp_sorted[:5]:
        print(f"    Group {r['gid']}: AUC={r['auc']:.3f}, gap={r['gap']:.3f} ({r['n_acc']}a/{r['n_rej']}r)")
    print(f"  Bottom 5 by AUC:")
    for r in pp_sorted[-5:]:
        print(f"    Group {r['gid']}: AUC={r['auc']:.3f}, gap={r['gap']:.3f} ({r['n_acc']}a/{r['n_rej']}r)")

    # 4. Inter-response correlation
    corr_results = inter_response_correlation(infl, acc, gids)
    if corr_results:
        all_corrs = [r["mean_corr"] for r in corr_results]
        print(f"\n--- INTER-RESPONSE CORRELATION (within group) ---")
        print(f"Mean correlation: {np.mean(all_corrs):.4f}")
        print(f"Range: [{min(all_corrs):.4f}, {max(all_corrs):.4f}]")
        print(f"  (>0.1 = influence captures prompt-specific patterns)")

    # 5. Score distribution
    all_valid = []
    for row in infl:
        valid = ~np.isnan(row.astype(np.float32))
        if valid.sum() > 0:
            all_valid.append(row[valid].astype(np.float32))
    all_scores = np.concatenate(all_valid)

    print(f"\n--- SCORE DISTRIBUTION ---")
    print(f"Total tokens: {len(all_scores)}")
    print(f"Mean: {all_scores.mean():.4f}, Std: {all_scores.std():.4f}")
    print(f"Positive fraction: {(all_scores > 0).sum() / len(all_scores):.4f}")
    print(f"Skewness: {((all_scores - all_scores.mean())**3).mean() / all_scores.std()**3:.4f}")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"  P{p}: {np.percentile(all_scores, p):.4f}")

    # 6. Token-type analysis
    if tokenizer is not None:
        print(f"\n--- TOKEN TYPE ANALYSIS ---")
        tt_results = token_type_analysis(infl, responses, acc, tokenizer)
        if tt_results:
            print(f"Top 20 tokens by |mean influence|:")
            for r in tt_results[:20]:
                print(f"  {repr(r['token']):>20s} (id={r['token_id']:>6d}): "
                      f"mean={r['mean']:+.3f}, std={r['std']:.3f}, count={r['count']}, "
                      f"acc={r['acc_mean']:+.3f}, rej={r['rej_mean']:+.3f}")

    # 7. Balanced vs unbalanced groups
    print(f"\n--- GROUP BALANCE EFFECT ---")
    balanced = [r for r in pp_results if min(r["n_acc"], r["n_rej"]) >= 3]
    unbalanced = [r for r in pp_results if min(r["n_acc"], r["n_rej"]) < 3]
    if balanced:
        print(f"Balanced groups (min 3 acc & 3 rej): {len(balanced)}")
        print(f"  Mean AUC: {np.mean([r['auc'] for r in balanced]):.4f}")
        print(f"  Sign correct: {sum(1 for r in balanced if r['sign_ok'])}/{len(balanced)}")
    if unbalanced:
        print(f"Unbalanced groups: {len(unbalanced)}")
        print(f"  Mean AUC: {np.mean([r['auc'] for r in unbalanced]):.4f}")
        print(f"  Sign correct: {sum(1 for r in unbalanced if r['sign_ok'])}/{len(unbalanced)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", help="Path to influence_trace directory")
    parser.add_argument("--step", type=int, default=None, help="Specific step to analyze")
    parser.add_argument("--tokenizer", type=str, default=None, help="Model path for tokenizer")
    args = parser.parse_args()

    npz = load_trace(args.trace_dir, args.step)

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print_report(npz, tokenizer)


if __name__ == "__main__":
    main()
