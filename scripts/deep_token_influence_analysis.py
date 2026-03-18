"""Deep token-level influence analysis across configurations.

Produces insights on:
1. Cross-config agreement: do different configs agree on which tokens matter?
2. Entropy-influence correlation: are high-entropy tokens scored differently?
3. Error localization: can influence identify where rejected responses go wrong?
4. Answer-critical tokens: do tokens near boxed answers get high influence?
5. Influence distribution shape: heavy-tailed? bimodal?
6. Within-group contrastive patterns: how influence separates acc/rej at token level
7. Token-level influence vs response-level: decomposition analysis
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_traces(configs):
    """Load multiple config traces."""
    data = {}
    for name, path in configs.items():
        steps_dir = Path(path) / "steps"
        npz_files = sorted(steps_dir.glob("step_*.npz"))
        if not npz_files:
            print(f"  SKIP {name}: no files in {steps_dir}")
            continue
        npz = np.load(npz_files[0], allow_pickle=True)
        data[name] = npz
        n = len(npz["influence"])
        n_acc = npz["accepted"].sum()
        print(f"  Loaded {name}: {n} rows, {n_acc} acc, {n - n_acc} rej")
    return data


def compute_response_score(infl_row):
    """Compute response-level influence score (mean of valid tokens)."""
    row = infl_row.astype(np.float32)
    valid = ~np.isnan(row)
    if valid.sum() == 0:
        return 0.0
    return row[valid].mean()


def compute_auc(pos_scores, neg_scores):
    """Compute AUC between positive and negative score arrays."""
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    n_correct = 0
    n_total = len(pos_scores) * len(neg_scores)
    for a in pos_scores:
        for r in neg_scores:
            if a > r:
                n_correct += 1
            elif a == r:
                n_correct += 0.5
    return n_correct / n_total


# ============================================================
# Analysis 1: Cross-config token ranking agreement
# ============================================================
def cross_config_agreement(data, tokenizer):
    """Compare token rankings between configs for the same responses."""
    print("\n" + "=" * 80)
    print("CROSS-CONFIG TOKEN RANKING AGREEMENT")
    print("=" * 80)
    print("Do different configs agree on which tokens are most/least influential?")
    print("High Spearman ρ → configs capture similar signal. Low ρ → different info.\n")

    config_names = list(data.keys())
    if len(config_names) < 2:
        print("  Need at least 2 configs for comparison.")
        return

    def spearmanr(a, b):
        """Numpy-only Spearman rank correlation."""
        def _rank(x):
            order = np.argsort(x)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(x), dtype=float)
            return ranks
        ra, rb = _rank(a), _rank(b)
        return np.corrcoef(ra, rb)[0, 1], None

    # Compare all pairs
    for i in range(len(config_names)):
        for j in range(i + 1, len(config_names)):
            c1, c2 = config_names[i], config_names[j]
            d1, d2 = data[c1], data[c2]
            n = min(len(d1["influence"]), len(d2["influence"]))

            rhos = []
            rhos_top = []  # correlation among top-20% tokens only
            for k in range(n):
                row1 = d1["influence"][k].astype(np.float32)
                row2 = d2["influence"][k].astype(np.float32)
                valid = (~np.isnan(row1)) & (~np.isnan(row2))
                nv = valid.sum()
                if nv < 20:
                    continue
                r1 = row1[valid]
                r2 = row2[valid]
                rho, _ = spearmanr(r1, r2)
                if not np.isnan(rho):
                    rhos.append(rho)

                # Top-20% agreement
                thresh1 = np.percentile(r1, 80)
                thresh2 = np.percentile(r2, 80)
                top1 = set(np.where(r1 >= thresh1)[0])
                top2 = set(np.where(r2 >= thresh2)[0])
                if len(top1) > 0 and len(top2) > 0:
                    jaccard = len(top1 & top2) / len(top1 | top2)
                    rhos_top.append(jaccard)

            if rhos:
                print(f"  {c1} vs {c2}:")
                print(f"    Spearman ρ: mean={np.mean(rhos):.4f}, "
                      f"median={np.median(rhos):.4f}, std={np.std(rhos):.4f}")
                if rhos_top:
                    print(f"    Top-20% Jaccard: mean={np.mean(rhos_top):.4f}")
                # Split by acc/rej
                acc = d1["accepted"][:n]
                rhos_acc = [rhos[k] for k in range(len(rhos)) if k < len(acc) and acc[k]]
                rhos_rej = [rhos[k] for k in range(len(rhos)) if k < len(acc) and not acc[k]]
                if rhos_acc and rhos_rej:
                    print(f"    ρ (accepted): {np.mean(rhos_acc):.4f}, "
                          f"ρ (rejected): {np.mean(rhos_rej):.4f}")


# ============================================================
# Analysis 2: Entropy-influence correlation
# ============================================================
def entropy_influence_analysis(data, tokenizer):
    """Correlate token entropy with influence score."""
    print("\n" + "=" * 80)
    print("ENTROPY-INFLUENCE CORRELATION")
    print("=" * 80)
    print("High-entropy tokens = model is uncertain. Do they get different influence?\n")

    for config_name, npz in data.items():
        infl = npz["influence"]
        ent = npz["entropies"]
        acc = npz["accepted"]
        resp_mask = npz["response_mask"]

        # Bucketize entropy
        ent_buckets = [
            (0.0, 0.1, "very_low (0-0.1)"),
            (0.1, 0.5, "low (0.1-0.5)"),
            (0.5, 1.0, "medium (0.5-1.0)"),
            (1.0, 2.0, "high (1.0-2.0)"),
            (2.0, 5.0, "very_high (2.0-5.0)"),
            (5.0, 100.0, "extreme (5.0+)"),
        ]

        print(f"  Config: {config_name}")
        print(f"  {'Entropy bucket':>25s} | {'N(acc)':>7s} {'Mean(acc)':>10s} | "
              f"{'N(rej)':>7s} {'Mean(rej)':>10s} | {'Gap':>8s} {'Var(infl)':>10s}")
        print("  " + "-" * 95)

        for lo, hi, label in ent_buckets:
            acc_scores = []
            rej_scores = []
            all_scores = []
            for i in range(len(infl)):
                row_infl = infl[i].astype(np.float32)
                row_ent = ent[i].astype(np.float32)
                row_mask = resp_mask[i]
                valid = (~np.isnan(row_infl)) & row_mask
                ent_mask = (row_ent >= lo) & (row_ent < hi) & valid
                if ent_mask.sum() == 0:
                    continue
                scores = row_infl[ent_mask]
                all_scores.extend(scores.tolist())
                if acc[i]:
                    acc_scores.extend(scores.tolist())
                else:
                    rej_scores.extend(scores.tolist())

            if not acc_scores or not rej_scores:
                continue
            acc_m = np.mean(acc_scores)
            rej_m = np.mean(rej_scores)
            var_all = np.var(all_scores)
            print(f"  {label:>25s} | {len(acc_scores):>7d} {acc_m:>+10.4f} | "
                  f"{len(rej_scores):>7d} {rej_m:>+10.4f} | {acc_m - rej_m:>+8.4f} {var_all:>10.6f}")
        print()


# ============================================================
# Analysis 3: Error localization in rejected responses
# ============================================================
def error_localization_analysis(data, tokenizer):
    """For rejected responses, find where influence drops most sharply.
    Hypothesis: influence should become more negative near the error point."""
    print("\n" + "=" * 80)
    print("ERROR LOCALIZATION IN REJECTED RESPONSES")
    print("=" * 80)
    print("If influence captures token-level credit, it should identify error points.")
    print("We look for sharp influence drops in rejected responses.\n")

    for config_name, npz in data.items():
        infl = npz["influence"]
        acc = npz["accepted"]
        gids = npz["group_id"]
        responses = npz["responses"]

        print(f"  Config: {config_name}")

        # For each rejected response, compute running influence and find
        # the position of maximum negative change
        drop_positions_rel = []  # relative position (0-1) of biggest drop
        n_analyzed = 0

        for i in range(len(infl)):
            if acc[i]:
                continue
            row = infl[i].astype(np.float32)
            valid = ~np.isnan(row)
            n_valid = valid.sum()
            if n_valid < 50:
                continue

            scores = row[valid]
            # Compute running average with window
            window = 20
            if len(scores) < window * 2:
                continue
            running = np.convolve(scores, np.ones(window) / window, mode="valid")
            # Find position of minimum running average (most negative region)
            min_pos = np.argmin(running)
            # Find position of maximum drop (biggest decrease in running avg)
            diffs = np.diff(running)
            if len(diffs) > 0:
                max_drop_pos = np.argmin(diffs)
                rel_pos = max_drop_pos / len(running)
                drop_positions_rel.append(rel_pos)
                n_analyzed += 1

        if drop_positions_rel:
            drop_positions_rel = np.array(drop_positions_rel)
            # Histogram of where biggest drops occur
            bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            print(f"    Analyzed {n_analyzed} rejected responses")
            print(f"    Distribution of maximum-drop position (relative):")
            for lo, hi in bins:
                count = ((drop_positions_rel >= lo) & (drop_positions_rel < hi)).sum()
                pct = count / len(drop_positions_rel) * 100
                bar = "#" * int(pct / 2)
                print(f"      [{lo:.1f}-{hi:.1f}]: {count:3d} ({pct:5.1f}%) {bar}")
            print(f"    Mean drop position: {drop_positions_rel.mean():.3f}")
            print(f"    Median drop position: {np.median(drop_positions_rel):.3f}")
        print()


# ============================================================
# Analysis 4: Answer-token influence (boxed answer detection)
# ============================================================
def answer_token_analysis(data, tokenizer):
    """Check if tokens near \\boxed{...} answers get distinctive influence."""
    print("\n" + "=" * 80)
    print("ANSWER-TOKEN INFLUENCE ANALYSIS")
    print("=" * 80)
    print("Tokens near \\boxed{answer} should be critical for correctness.\n")

    # Find token IDs for boxed-related tokens
    vocab = tokenizer.get_vocab()
    boxed_tokens = set()
    for tok_str, tok_id in vocab.items():
        if "boxed" in tok_str.lower():
            boxed_tokens.add(tok_id)

    if not boxed_tokens:
        print("  No 'boxed' tokens found in vocabulary.")
        return

    print(f"  Found {len(boxed_tokens)} boxed-related token IDs")

    for config_name, npz in data.items():
        infl = npz["influence"]
        acc = npz["accepted"]
        responses = npz["responses"]

        print(f"\n  Config: {config_name}")

        # For each response, find boxed token positions and compare influence
        boxed_infl_acc = []
        boxed_infl_rej = []
        non_boxed_infl_acc = []
        non_boxed_infl_rej = []
        near_boxed_infl_acc = []  # within ±10 of boxed token
        near_boxed_infl_rej = []
        n_with_boxed = 0

        for i in range(len(infl)):
            row = infl[i].astype(np.float32)
            resp = responses[i]
            valid = ~np.isnan(row)

            # Find boxed positions
            boxed_positions = set()
            for j in range(len(resp)):
                if int(resp[j]) in boxed_tokens:
                    boxed_positions.add(j)

            if not boxed_positions:
                continue
            n_with_boxed += 1

            # Near-boxed: within ±10 tokens
            near_positions = set()
            for bp in boxed_positions:
                for offset in range(-10, 11):
                    near_positions.add(bp + offset)
            near_positions -= boxed_positions

            for j in range(min(len(resp), len(row))):
                if not valid[j]:
                    continue
                score = float(row[j])
                if j in boxed_positions:
                    if acc[i]:
                        boxed_infl_acc.append(score)
                    else:
                        boxed_infl_rej.append(score)
                elif j in near_positions:
                    if acc[i]:
                        near_boxed_infl_acc.append(score)
                    else:
                        near_boxed_infl_rej.append(score)
                else:
                    if acc[i]:
                        non_boxed_infl_acc.append(score)
                    else:
                        non_boxed_infl_rej.append(score)

        print(f"    {n_with_boxed} responses contain boxed tokens")
        regions = [
            ("boxed tokens", boxed_infl_acc, boxed_infl_rej),
            ("near-boxed (±10)", near_boxed_infl_acc, near_boxed_infl_rej),
            ("other tokens", non_boxed_infl_acc, non_boxed_infl_rej),
        ]
        print(f"    {'Region':>20s} | {'N(acc)':>7s} {'Mean(acc)':>10s} | "
              f"{'N(rej)':>7s} {'Mean(rej)':>10s} | {'Gap':>8s}")
        print("    " + "-" * 80)
        for label, acc_s, rej_s in regions:
            if not acc_s or not rej_s:
                continue
            acc_m = np.mean(acc_s)
            rej_m = np.mean(rej_s)
            print(f"    {label:>20s} | {len(acc_s):>7d} {acc_m:>+10.4f} | "
                  f"{len(rej_s):>7d} {rej_m:>+10.4f} | {acc_m - rej_m:>+8.4f}")


# ============================================================
# Analysis 5: Influence distribution shape
# ============================================================
def distribution_analysis(data, tokenizer):
    """Analyze the shape of influence score distributions."""
    print("\n" + "=" * 80)
    print("INFLUENCE DISTRIBUTION SHAPE")
    print("=" * 80)
    print("Is the distribution heavy-tailed? Bimodal? How concentrated?\n")

    for config_name, npz in data.items():
        infl = npz["influence"]
        acc = npz["accepted"]

        all_acc = []
        all_rej = []
        for i in range(len(infl)):
            row = infl[i].astype(np.float32)
            valid = ~np.isnan(row)
            scores = row[valid]
            if acc[i]:
                all_acc.extend(scores.tolist())
            else:
                all_rej.extend(scores.tolist())

        all_acc = np.array(all_acc)
        all_rej = np.array(all_rej)

        print(f"  Config: {config_name}")
        for label, scores in [("accepted", all_acc), ("rejected", all_rej)]:
            if len(scores) == 0:
                continue
            pcts = np.percentile(scores, [1, 5, 25, 50, 75, 95, 99])
            kurt = (((scores - scores.mean()) / scores.std()) ** 4).mean() - 3 if scores.std() > 0 else 0
            # Fraction of tokens with |score| > 2*std
            outlier_frac = (np.abs(scores) > 2 * scores.std()).mean()
            print(f"    {label}:")
            print(f"      mean={scores.mean():+.4f}, std={scores.std():.4f}, "
                  f"kurtosis={kurt:+.2f}")
            print(f"      percentiles: 1%={pcts[0]:+.4f}, 5%={pcts[1]:+.4f}, "
                  f"25%={pcts[2]:+.4f}, 50%={pcts[3]:+.4f}, "
                  f"75%={pcts[4]:+.4f}, 95%={pcts[5]:+.4f}, 99%={pcts[6]:+.4f}")
            print(f"      outlier fraction (|s|>2σ): {outlier_frac:.4f}")
        # Overlap: what fraction of rejected tokens have scores above accepted median?
        if len(all_acc) > 0 and len(all_rej) > 0:
            acc_med = np.median(all_acc)
            overlap = (all_rej >= acc_med).mean()
            print(f"    Overlap: {overlap:.4f} of rejected tokens above accepted median")
        print()


# ============================================================
# Analysis 6: Response-level vs token-level decomposition
# ============================================================
def decomposition_analysis(data, tokenizer):
    """How much of token influence is explained by response-level signal?

    For each token: score_t = response_mean + residual_t
    If residual variance is small relative to response_mean variance,
    then token scores are just response-level signal rescaled.
    """
    print("\n" + "=" * 80)
    print("RESPONSE-LEVEL vs TOKEN-LEVEL DECOMPOSITION")
    print("=" * 80)
    print("score_t = response_mean + residual_t")
    print("If residual variance ≈ 0, scores are just response-level signal.\n")

    for config_name, npz in data.items():
        infl = npz["influence"]
        acc = npz["accepted"]

        resp_means = []
        all_residuals = []
        all_raw = []
        resp_means_acc = []
        resp_means_rej = []

        for i in range(len(infl)):
            row = infl[i].astype(np.float32)
            valid = ~np.isnan(row)
            n_valid = valid.sum()
            if n_valid < 10:
                continue
            scores = row[valid]
            resp_mean = scores.mean()
            residuals = scores - resp_mean
            resp_means.append(resp_mean)
            all_residuals.extend(residuals.tolist())
            all_raw.extend(scores.tolist())
            if acc[i]:
                resp_means_acc.append(resp_mean)
            else:
                resp_means_rej.append(resp_mean)

        resp_means = np.array(resp_means)
        all_residuals = np.array(all_residuals)
        all_raw = np.array(all_raw)

        var_total = all_raw.var()
        var_resp = resp_means.var()
        var_resid = all_residuals.var()

        print(f"  Config: {config_name}")
        print(f"    Total variance:    {var_total:.6f}")
        print(f"    Response-mean var: {var_resp:.6f} ({var_resp / var_total * 100:.1f}%)")
        print(f"    Residual var:      {var_resid:.6f} ({var_resid / var_total * 100:.1f}%)")
        print(f"    → Token-level signal is {'WEAK' if var_resid / var_total < 0.3 else 'MODERATE' if var_resid / var_total < 0.6 else 'STRONG'} "
              f"relative to response-level")

        # Can residuals discriminate acc/rej within groups?
        gids = npz["group_id"]
        unique_gids = np.unique(gids)
        within_group_residual_aucs = []
        for gid in unique_gids:
            mask = gids == gid
            g_infl = infl[mask]
            g_acc = acc[mask]
            if g_acc.sum() == 0 or (~g_acc).sum() == 0:
                continue
            # Compute residuals (subtract group mean)
            acc_resid_means = []
            rej_resid_means = []
            for k in range(mask.sum()):
                row = g_infl[k].astype(np.float32)
                valid = ~np.isnan(row)
                if valid.sum() < 10:
                    continue
                scores = row[valid]
                resid = scores - scores.mean()
                # Use std of residuals as a proxy for within-response spread
                if g_acc[k]:
                    acc_resid_means.append(resid.std())
                else:
                    rej_resid_means.append(resid.std())
            if acc_resid_means and rej_resid_means:
                auc = compute_auc(acc_resid_means, rej_resid_means)
                within_group_residual_aucs.append(auc)

        if within_group_residual_aucs:
            print(f"    Residual-std AUC (acc vs rej within group): "
                  f"{np.mean(within_group_residual_aucs):.4f} "
                  f"(random=0.5)")
        print()


# ============================================================
# Analysis 7: Influence gradient (derivative along sequence)
# ============================================================
def influence_gradient_analysis(data, tokenizer):
    """How does influence change along the sequence?
    Compute d(influence)/d(position) — the local trend.
    In accepted responses, does it trend upward toward the answer?
    In rejected responses, does it trend downward at the error?
    """
    print("\n" + "=" * 80)
    print("INFLUENCE GRADIENT (TREND ALONG SEQUENCE)")
    print("=" * 80)
    print("slope > 0 means influence increases toward end of response.\n")

    for config_name, npz in data.items():
        infl = npz["influence"]
        acc = npz["accepted"]

        slopes_acc = []
        slopes_rej = []
        # Also compute slope in last 30% vs first 30%
        late_early_ratio_acc = []
        late_early_ratio_rej = []

        for i in range(len(infl)):
            row = infl[i].astype(np.float32)
            valid = ~np.isnan(row)
            n_valid = valid.sum()
            if n_valid < 50:
                continue
            scores = row[valid]

            # Linear regression: slope
            x = np.arange(len(scores))
            slope = np.polyfit(x, scores, 1)[0]

            # Late vs early
            n30 = max(1, int(len(scores) * 0.3))
            early_mean = scores[:n30].mean()
            late_mean = scores[-n30:].mean()

            if acc[i]:
                slopes_acc.append(slope)
                late_early_ratio_acc.append(late_mean - early_mean)
            else:
                slopes_rej.append(slope)
                late_early_ratio_rej.append(late_mean - early_mean)

        print(f"  Config: {config_name}")
        if slopes_acc and slopes_rej:
            print(f"    Accepted: mean_slope={np.mean(slopes_acc):+.6f}, "
                  f"late-early_gap={np.mean(late_early_ratio_acc):+.4f}")
            print(f"    Rejected: mean_slope={np.mean(slopes_rej):+.6f}, "
                  f"late-early_gap={np.mean(late_early_ratio_rej):+.4f}")
            # Is the slope itself discriminative?
            auc = compute_auc(slopes_acc, slopes_rej)
            print(f"    Slope AUC (acc vs rej): {auc:.4f}")
        print()


# ============================================================
# Analysis 8: Contrastive token examples with full context
# ============================================================
def contrastive_examples(data, tokenizer, n_examples=3):
    """Show matched acc/rej pairs from the same prompt group.
    Highlight tokens where influence differs most between acc and rej.
    """
    print("\n" + "=" * 80)
    print("CONTRASTIVE TOKEN EXAMPLES (acc vs rej from same prompt)")
    print("=" * 80)

    # Use the first config
    config_name = list(data.keys())[0]
    npz = data[config_name]
    infl = npz["influence"]
    acc = npz["accepted"]
    gids = npz["group_id"]
    responses = npz["responses"]

    print(f"  Using config: {config_name}\n")

    unique_gids = np.unique(gids)
    shown = 0
    for gid in unique_gids:
        if shown >= n_examples:
            break
        mask = gids == gid
        indices = np.where(mask)[0]
        g_acc = acc[mask]
        if g_acc.sum() == 0 or (~g_acc).sum() == 0:
            continue

        # Pick first accepted and first rejected
        acc_idx = indices[g_acc][0]
        rej_idx = indices[~g_acc][0]

        # Decode and show first 40 tokens with influence
        print(f"  --- Group {gid} ---")
        for label, idx in [("ACCEPTED", acc_idx), ("REJECTED", rej_idx)]:
            row = infl[idx].astype(np.float32)
            resp = responses[idx]
            valid = ~np.isnan(row)
            n_valid = valid.sum()
            if n_valid < 20:
                continue

            resp_mean = row[valid].mean()
            print(f"  [{label}] idx={idx}, n_tok={n_valid}, resp_mean={resp_mean:+.4f}")

            # Show first 30 tokens
            count = 0
            line = "    "
            for j in range(min(len(resp), len(row))):
                if not valid[j]:
                    continue
                tok_str = tokenizer.decode([resp[j]])
                s = float(row[j])
                # Color coding: positive = good, negative = bad
                marker = "+" if s > 0.05 else ("-" if s < -0.05 else " ")
                line += f"[{marker}{s:+.2f}]{repr(tok_str):10s} "
                count += 1
                if count % 5 == 0:
                    print(line)
                    line = "    "
                if count >= 30:
                    break
            if line.strip():
                print(line)

            # Show last 15 tokens (near answer)
            print(f"    ... last 15 tokens:")
            valid_indices = np.where(valid)[0]
            line = "    "
            for j in valid_indices[-15:]:
                tok_str = tokenizer.decode([resp[j]])
                s = float(row[j])
                marker = "+" if s > 0.05 else ("-" if s < -0.05 else " ")
                line += f"[{marker}{s:+.2f}]{repr(tok_str):10s} "
            print(line)
            print()

        shown += 1


# ============================================================
# Analysis 9: Influence concentration (Gini coefficient)
# ============================================================
def concentration_analysis(data, tokenizer):
    """How concentrated is influence among a few tokens vs spread evenly?
    Gini=0: perfectly equal. Gini=1: all influence in one token."""
    print("\n" + "=" * 80)
    print("INFLUENCE CONCENTRATION (GINI COEFFICIENT)")
    print("=" * 80)
    print("How concentrated is influence? Gini=0: uniform, Gini=1: one token.\n")

    for config_name, npz in data.items():
        infl = npz["influence"]
        acc = npz["accepted"]

        gini_acc = []
        gini_rej = []
        top10_share_acc = []
        top10_share_rej = []

        for i in range(len(infl)):
            row = infl[i].astype(np.float32)
            valid = ~np.isnan(row)
            n_valid = valid.sum()
            if n_valid < 20:
                continue
            scores = np.abs(row[valid])
            # Gini coefficient
            sorted_s = np.sort(scores)
            n = len(sorted_s)
            cumsum = np.cumsum(sorted_s)
            if cumsum[-1] == 0:
                continue
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_s)) / (n * cumsum[-1])) - (n + 1) / n

            # Top-10% share
            top_n = max(1, n // 10)
            top_share = sorted_s[-top_n:].sum() / cumsum[-1]

            if acc[i]:
                gini_acc.append(gini)
                top10_share_acc.append(top_share)
            else:
                gini_rej.append(gini)
                top10_share_rej.append(top_share)

        print(f"  Config: {config_name}")
        if gini_acc and gini_rej:
            print(f"    Gini (accepted): {np.mean(gini_acc):.4f} ± {np.std(gini_acc):.4f}")
            print(f"    Gini (rejected): {np.mean(gini_rej):.4f} ± {np.std(gini_rej):.4f}")
            print(f"    Top-10% share (acc): {np.mean(top10_share_acc):.4f}")
            print(f"    Top-10% share (rej): {np.mean(top10_share_rej):.4f}")
            auc = compute_auc(gini_acc, gini_rej)
            print(f"    Gini AUC (acc vs rej): {auc:.4f}")
        print()


# ============================================================
# Analysis 10: Prompt difficulty vs influence clarity
# ============================================================
def difficulty_analysis(data, tokenizer):
    """Do 'easy' prompts (high solve rate) have cleaner influence signal?"""
    print("\n" + "=" * 80)
    print("PROMPT DIFFICULTY vs INFLUENCE CLARITY")
    print("=" * 80)
    print("Easy prompts (high acc rate) might have cleaner signal.\n")

    for config_name, npz in data.items():
        infl = npz["influence"]
        acc = npz["accepted"]
        gids = npz["group_id"]

        unique_gids = np.unique(gids)
        easy_gap = []
        hard_gap = []
        mid_gap = []

        for gid in unique_gids:
            mask = gids == gid
            g_acc = acc[mask]
            n = mask.sum()
            solve_rate = g_acc.mean()

            # Compute mean influence for acc vs rej
            g_infl = infl[mask]
            acc_means = []
            rej_means = []
            for k in range(n):
                row = g_infl[k].astype(np.float32)
                valid = ~np.isnan(row)
                if valid.sum() < 10:
                    continue
                m = row[valid].mean()
                if g_acc[k]:
                    acc_means.append(m)
                else:
                    rej_means.append(m)
            if not acc_means or not rej_means:
                continue
            gap = np.mean(acc_means) - np.mean(rej_means)

            if solve_rate >= 0.5:
                easy_gap.append(gap)
            elif solve_rate <= 0.2:
                hard_gap.append(gap)
            else:
                mid_gap.append(gap)

        print(f"  Config: {config_name}")
        for label, gaps in [("Easy (≥50% solve)", easy_gap),
                            ("Medium (20-50%)", mid_gap),
                            ("Hard (≤20% solve)", hard_gap)]:
            if gaps:
                print(f"    {label:25s}: mean_gap={np.mean(gaps):+.4f}, "
                      f"n_groups={len(gaps)}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--configs", nargs="+", required=True,
                        help="name:path pairs, e.g., lastmlp:/path/to/trace")
    parser.add_argument("--analyses", nargs="*", default=None,
                        help="Which analyses to run (1-10). Default: all.")
    args = parser.parse_args()

    # Parse configs
    configs = {}
    for c in args.configs:
        name, path = c.split(":", 1)
        configs[name] = path
    print(f"Loading {len(configs)} configs...")
    data = load_traces(configs)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    analyses = {
        "1": ("Cross-config agreement", cross_config_agreement),
        "2": ("Entropy-influence correlation", entropy_influence_analysis),
        "3": ("Error localization", error_localization_analysis),
        "4": ("Answer-token influence", answer_token_analysis),
        "5": ("Distribution shape", distribution_analysis),
        "6": ("Response vs token decomposition", decomposition_analysis),
        "7": ("Influence gradient", influence_gradient_analysis),
        "8": ("Contrastive examples", contrastive_examples),
        "9": ("Concentration (Gini)", concentration_analysis),
        "10": ("Difficulty vs clarity", difficulty_analysis),
    }

    selected = args.analyses or list(analyses.keys())
    for key in selected:
        if key in analyses:
            name, func = analyses[key]
            print(f"\n{'#' * 80}")
            print(f"# ANALYSIS {key}: {name}")
            print(f"{'#' * 80}")
            try:
                func(data, tokenizer)
            except Exception as e:
                print(f"  ERROR in analysis {key}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
