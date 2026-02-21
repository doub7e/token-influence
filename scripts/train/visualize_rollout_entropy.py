#!/usr/bin/env python3
"""Interactive visualizer for rollout token-level entropy traces."""

from __future__ import annotations

import argparse
import html
import json
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize rollout entropy traces.")
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="output/Archer2.0/Archer2.0-Qwen2.5-1.5B-Math-InfluenceTrace/influence_trace",
        help="Directory that contains manifest.jsonl and steps/*.npz",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Tokenizer/model path used for token decoding",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    return parser.parse_args()


def _safe_float(x: float) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _read_index(trace_dir: Path) -> list[dict[str, Any]]:
    manifest_path = trace_dir / "manifest.jsonl"
    records: list[dict[str, Any]] = []
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    else:
        for step_file in sorted((trace_dir / "steps").glob("step_*.npz")):
            step = int(step_file.stem.split("_")[-1])
            records.append(
                {
                    "step": step,
                    "file": str(Path("steps") / step_file.name),
                    "num_responses": None,
                    "response_len": None,
                    "num_valid_tokens": None,
                    "entropy_min": None,
                    "entropy_max": None,
                    "entropy_mean": None,
                    "entropy_std": None,
                }
            )
    records.sort(key=lambda x: int(x["step"]))
    return records


@lru_cache(maxsize=256)
def _load_step_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


@lru_cache(maxsize=2)
def _load_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _decode_token(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    text = tokenizer.decode([int(token_id)])
    if not text:
        return str(token_id)
    return text


def _decode_ids(tokenizer, token_ids: Any) -> str:
    if tokenizer is None:
        return str(token_ids)
    if token_ids is None:
        return ""
    if isinstance(token_ids, np.ndarray):
        ids = token_ids.tolist()
    else:
        ids = token_ids
    if isinstance(ids, str):
        ids = ids.strip()
        if not ids or ids == "[]":
            return "(prompt IDs not saved in this trace)"
        try:
            ids = json.loads(ids)
        except (json.JSONDecodeError, ValueError):
            return ids
    if not isinstance(ids, (list, tuple)):
        return str(ids)
    if len(ids) == 0:
        return "(prompt IDs not saved in this trace)"
    try:
        ids = [int(x) for x in ids]
    except (TypeError, ValueError):
        return str(ids)
    return tokenizer.decode(ids, skip_special_tokens=False)


def _entropy_to_rgb(val: float, lo: float, hi: float) -> str:
    if not np.isfinite(val):
        return "rgb(120,120,120)"
    if hi <= lo:
        p = 0.5
    else:
        p = float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))
    # blue -> red
    r = int(30 + p * 220)
    g = int(80 + (1.0 - p) * 120)
    b = int(220 - p * 180)
    return f"rgb({r},{g},{b})"


def _influence_to_rgb(val: float, bound: float) -> str:
    """Diverging colormap: deep blue (most negative) → white (zero) → deep red (most positive)."""
    if not np.isfinite(val):
        return "rgb(140,140,140)"
    if bound <= 0:
        bound = 1e-6
    clamped = float(np.clip(val, -bound, bound))
    t = abs(clamped) / bound
    if clamped >= 0:
        r = 255
        g = int(255 * (1.0 - t))
        b = int(255 * (1.0 - t))
    else:
        r = int(255 * (1.0 - t))
        g = int(255 * (1.0 - t))
        b = 255
    return f"rgb({r},{g},{b})"


def _render_token_heatmap(df: pd.DataFrame, lo: float, hi: float, influence_bound: float | None = None) -> str:
    if df.empty:
        return "<div style='padding:12px'>No tokens after filtering.</div>"

    chunks: list[str] = []
    for row in df.itertuples(index=False):
        pos = int(row.position)
        token = html.escape(str(row.token))
        entropy = float(row.entropy)
        valid = bool(row.valid)
        bg = _entropy_to_rgb(entropy, lo, hi) if valid else "rgb(100,100,100)"
        fg = "white" if valid else "rgb(200,200,200)"
        influence_val = float(getattr(row, "influence", float("nan")))
        if influence_bound is not None and np.isfinite(influence_val):
            bg = _influence_to_rgb(influence_val, influence_bound) if valid else "rgb(100,100,100)"
            if valid:
                inf_p = float(np.clip((influence_val + influence_bound) / (2.0 * influence_bound), 0.0, 1.0))
                fg = "#333" if 0.3 < inf_p < 0.7 else "white"
        tooltip = html.escape(
            f"pos={pos} id={int(row.token_id)} entropy={entropy:.5f} influence={influence_val:.5f} valid={valid}"
        )
        chunks.append(
            f"<span title='{tooltip}' style='display:inline-block;margin:2px;padding:4px 6px;border-radius:6px;"
            f"background:{bg};color:{fg};font-family:monospace;font-size:12px'>{token}</span>"
        )
    return "<div style='line-height:1.8'>" + "".join(chunks) + "</div>"


def _compute_step_stats(ent: np.ndarray, mask: np.ndarray) -> str:
    valid = ent[mask]
    if valid.size == 0:
        return "No valid tokens in this step."
    p50, p90, p99 = np.quantile(valid, [0.5, 0.9, 0.99])
    return (
        f"responses={ent.shape[0]}, response_len={ent.shape[1]}, valid_tokens={mask.sum()}  \n"
        f"entropy_mean={valid.mean():.6f}, std={valid.std():.6f}, min={valid.min():.6f}, max={valid.max():.6f}  \n"
        f"p50={p50:.6f}, p90={p90:.6f}, p99={p99:.6f}"
    )


def _compute_influence_stats(inf: np.ndarray, mask: np.ndarray) -> str:
    valid = inf[np.isfinite(inf) & mask]
    if valid.size == 0:
        return "influence: none"
    p50, p90, p99 = np.quantile(valid, [0.5, 0.9, 0.99])
    return (
        f"influence_mean={valid.mean():.6f}, std={valid.std():.6f}, min={valid.min():.6f}, max={valid.max():.6f}  \n"
        f"influence_p50={p50:.6f}, influence_p90={p90:.6f}, influence_p99={p99:.6f}"
    )


def _flatten_topk(
    ent: np.ndarray,
    mask: np.ndarray,
    responses: np.ndarray,
    tokenizer,
    top_k: int,
    highest: bool,
) -> pd.DataFrame:
    valid_idx = np.argwhere(mask)
    if valid_idx.size == 0:
        return pd.DataFrame(columns=["response_idx", "token_pos", "token_id", "token", "entropy"])
    vals = ent[mask]
    order = np.argsort(vals)
    if highest:
        order = order[::-1]
    order = order[:top_k]
    selected = valid_idx[order]

    rows = []
    for i, j in selected:
        token_id = int(responses[i, j])
        rows.append(
            {
                "response_idx": int(i),
                "token_pos": int(j),
                "token_id": token_id,
                "token": _decode_token(tokenizer, token_id),
                "entropy": float(ent[i, j]),
            }
        )
    return pd.DataFrame(rows)


def _build_response_df(
    ent: np.ndarray,
    influence: np.ndarray | None,
    mask: np.ndarray,
    responses: np.ndarray,
    tokenizer,
    response_idx: int,
    min_entropy: float,
    max_entropy: float,
    token_query: str,
    include_invalid: bool,
    sort_by: str,
) -> pd.DataFrame:
    token_ids = responses[response_idx]
    ent_row = ent[response_idx]
    mask_row = mask[response_idx]

    rows = []
    query = token_query.strip().lower()
    for pos, (tid, e, valid) in enumerate(zip(token_ids, ent_row, mask_row)):
        if (not include_invalid) and (not valid):
            continue
        if valid and (e < min_entropy or e > max_entropy):
            continue
        tok = _decode_token(tokenizer, int(tid))
        if query and query not in tok.lower():
            continue
        rows.append(
            {
                "position": int(pos),
                "token_id": int(tid),
                "token": tok,
                "entropy": float(e),
                "influence": float(influence[response_idx, pos]) if influence is not None else float("nan"),
                "valid": bool(valid),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if sort_by == "entropy_desc":
        return df.sort_values(by=["entropy", "position"], ascending=[False, True], ignore_index=True)
    if sort_by == "entropy_asc":
        return df.sort_values(by=["entropy", "position"], ascending=[True, True], ignore_index=True)
    return df.sort_values(by=["position"], ascending=[True], ignore_index=True)


def main() -> None:
    args = parse_args()

    def load_trace(trace_dir_str: str, model_path: str):
        trace_dir = Path(trace_dir_str).expanduser().resolve()
        if not trace_dir.exists():
            return (
                f"Trace dir not found: `{trace_dir}`",
                [],
                gr.Dropdown(choices=[], value=None),
                pd.DataFrame(),
            )
        records = _read_index(trace_dir)
        if not records:
            return (
                f"No trace records found under `{trace_dir}`.",
                [],
                gr.Dropdown(choices=[], value=None),
                pd.DataFrame(),
            )

        try:
            _ = _load_tokenizer(model_path)
            tok_info = f"Tokenizer loaded from `{model_path}`."
        except Exception as e:
            tok_info = f"Tokenizer load failed, fallback to token ids only. Error: {e}"

        df = pd.DataFrame(records)
        steps = [int(r["step"]) for r in records]
        status = f"Loaded {len(records)} rollout steps from `{trace_dir}`. {tok_info}"
        return status, records, gr.Dropdown(choices=steps, value=steps[0]), df

    def on_step_selected(records, trace_dir_str: str, step: int):
        if not records:
            return (
                "No records loaded.",
                gr.Slider(minimum=0, maximum=0, value=0, step=1),
                0.0,
                1.0,
                pd.DataFrame(),
            )
        rec = next((x for x in records if int(x["step"]) == int(step)), None)
        if rec is None:
            return (
                f"Step {step} not found.",
                gr.Slider(minimum=0, maximum=0, value=0, step=1),
                0.0,
                1.0,
                pd.DataFrame(),
            )
        step_path = Path(trace_dir_str).expanduser().resolve() / rec["file"]
        payload = _load_step_npz(str(step_path))
        ent = payload["entropies"].astype(np.float32)
        influence = payload["influence"].astype(np.float32) if "influence" in payload else None
        mask = payload["response_mask"].astype(bool)
        responses = payload["responses"].astype(np.int32)
        valid = ent[mask]
        lo = float(valid.min()) if valid.size else 0.0
        hi = float(valid.max()) if valid.size else 1.0

        overview = pd.DataFrame(
            {
                "response_idx": np.arange(ent.shape[0], dtype=np.int32),
                "valid_tokens": mask.sum(axis=1).astype(np.int32),
                "entropy_mean": np.where(mask.sum(axis=1) > 0, (ent * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1), np.nan),
                "entropy_max": np.where(mask.sum(axis=1) > 0, np.where(mask, ent, -np.inf).max(axis=1), np.nan),
                "entropy_min": np.where(mask.sum(axis=1) > 0, np.where(mask, ent, np.inf).min(axis=1), np.nan),
            }
        )
        if "sample_index" in payload:
            overview["sample_index"] = payload["sample_index"]
        if "uid" in payload:
            overview["uid"] = payload["uid"]
        if "reward" in payload:
            overview["reward"] = payload["reward"]
        if "accepted" in payload:
            overview["accepted"] = payload["accepted"]
        if "group_id" in payload:
            overview["group_id"] = payload["group_id"]
        if influence is not None:
            valid_inf = np.isfinite(influence) & mask
            influence_mean = np.where(valid_inf.sum(axis=1) > 0, np.where(valid_inf, influence, 0.0).sum(axis=1) / np.maximum(valid_inf.sum(axis=1), 1), np.nan)
            overview["influence_mean"] = influence_mean

        step_stats = _compute_step_stats(ent, mask)
        if influence is not None:
            step_stats = step_stats + "  \n" + _compute_influence_stats(influence, mask)
        return (
            step_stats,
            gr.Slider(minimum=0, maximum=max(ent.shape[0] - 1, 0), value=0, step=1),
            lo,
            hi,
            overview.head(300),
        )

    def inspect_response(
        records,
        trace_dir_str: str,
        model_path: str,
        step: int,
        response_idx: int,
        min_entropy: float,
        max_entropy: float,
        token_query: str,
        include_invalid: bool,
        top_k: int,
        sort_by: str,
    ):
        if not records:
            return "<div>Load trace first.</div>", "N/A", "N/A", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if step is None:
            return "<div>Select a step first.</div>", "N/A", "N/A", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        rec = next((x for x in records if int(x["step"]) == int(step)), None)
        if rec is None:
            return "<div>Step not found.</div>", "N/A", "N/A", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        step_path = Path(trace_dir_str).expanduser().resolve() / rec["file"]
        payload = _load_step_npz(str(step_path))
        ent = payload["entropies"].astype(np.float32)
        influence = payload["influence"].astype(np.float32) if "influence" in payload else None
        mask = payload["response_mask"].astype(bool)
        responses = payload["responses"].astype(np.int32)
        response_idx = int(np.clip(response_idx, 0, max(ent.shape[0] - 1, 0)))

        tokenizer = None
        try:
            tokenizer = _load_tokenizer(model_path)
        except Exception:
            tokenizer = None

        df = _build_response_df(
            ent=ent,
            influence=influence,
            mask=mask,
            responses=responses,
            tokenizer=tokenizer,
            response_idx=response_idx,
            min_entropy=_safe_float(min_entropy),
            max_entropy=_safe_float(max_entropy),
            token_query=token_query,
            include_invalid=include_invalid,
            sort_by=sort_by,
        )

        valid = ent[mask]
        lo = float(valid.min()) if valid.size else 0.0
        hi = float(valid.max()) if valid.size else 1.0
        view_df = df if sort_by == "position" else df.sort_values("position")
        entropy_html = _render_token_heatmap(view_df, lo, hi, influence_bound=None)
        influence_html = "<div style='padding:12px'>No influence in this trace step.</div>"
        if influence is not None:
            finite_inf = influence[np.isfinite(influence) & mask]
            if finite_inf.size:
                inf_bound = float(np.quantile(np.abs(finite_inf), 0.99))
                if inf_bound < 1e-8:
                    inf_bound = float(np.max(np.abs(finite_inf)))
            else:
                inf_bound = 1.0
            influence_html = _render_token_heatmap(view_df, lo, hi, influence_bound=inf_bound)
        token_html = (
            "<div style='display:flex;gap:16px'>"
            "<div style='flex:1'><h4>Entropy</h4>" + entropy_html + "</div>"
            "<div style='flex:1'><h4>Influence</h4>" + influence_html + "</div>"
            "</div>"
        )
        prompt_preview = "N/A"
        if "prompt_ids" in payload:
            prompt_preview = _decode_ids(tokenizer, payload["prompt_ids"][response_idx])
        response_meta = []
        if "reward" in payload:
            response_meta.append(f"reward={float(payload['reward'][response_idx]):.4f}")
        if "accepted" in payload:
            response_meta.append("accepted" if bool(payload["accepted"][response_idx]) else "rejected")
        if "group_id" in payload:
            response_meta.append(f"group_id={int(payload['group_id'][response_idx])}")
        response_meta_text = ", ".join(response_meta) if response_meta else "N/A"
        high_df = _flatten_topk(ent, mask, responses, tokenizer, int(top_k), highest=True)
        low_df = _flatten_topk(ent, mask, responses, tokenizer, int(top_k), highest=False)
        return token_html, prompt_preview, response_meta_text, df, high_df, low_df

    def export_selected_csv(
        records,
        trace_dir_str: str,
        model_path: str,
        step: int,
        response_idx: int,
        min_entropy: float,
        max_entropy: float,
        token_query: str,
        include_invalid: bool,
        sort_by: str,
    ):
        token_html, _, _, df, _, _ = inspect_response(
            records,
            trace_dir_str,
            model_path,
            step,
            response_idx,
            min_entropy,
            max_entropy,
            token_query,
            include_invalid,
            10,
            sort_by,
        )
        del token_html
        if df.empty:
            return gr.File(value=None, visible=False)
        tmp = Path(tempfile.mkdtemp(prefix="entropy_trace_")) / f"step_{step:06d}_response_{response_idx:04d}.csv"
        df.to_csv(tmp, index=False)
        return gr.File(value=str(tmp), visible=True)

    with gr.Blocks(title="Archer Rollout Entropy Visualizer") as demo:
        gr.Markdown(
            """
            # Archer Rollout Entropy + Influence Visualizer
            Inspect token-level entropy/influence for every rollout step and response.
            """
        )

        with gr.Row():
            trace_dir_tb = gr.Textbox(label="Trace Directory", value=args.trace_dir, scale=3)
            model_path_tb = gr.Textbox(label="Tokenizer/Model Path", value=args.model_path, scale=3)
            load_btn = gr.Button("Load", variant="primary", scale=1)

        load_status = gr.Markdown()
        records_state = gr.State([])

        manifest_df = gr.Dataframe(label="Rollout Step Manifest", wrap=True, interactive=False)

        with gr.Row():
            step_dd = gr.Dropdown(label="Rollout Step", choices=[], value=None)
            response_slider = gr.Slider(label="Response Index", minimum=0, maximum=0, step=1, value=0)
            top_k = gr.Slider(label="Top-K Tokens", minimum=5, maximum=200, step=5, value=30)

        step_stats_md = gr.Markdown()
        response_overview_df = gr.Dataframe(label="Response Overview (Current Step)", wrap=True, interactive=False)
        prompt_md = gr.Textbox(label="Prompt (Decoded)", lines=6, interactive=False)
        response_meta_md = gr.Textbox(label="Response Label", lines=1, interactive=False)

        with gr.Row():
            min_entropy = gr.Number(label="Min Entropy Filter", value=0.0)
            max_entropy = gr.Number(label="Max Entropy Filter", value=1.0)
            token_query = gr.Textbox(label="Token Contains (case-insensitive)", value="")

        with gr.Row():
            include_invalid = gr.Checkbox(label="Include Invalid/Padded Tokens", value=False)
            sort_by = gr.Dropdown(
                label="Token Table Sort",
                choices=["position", "entropy_desc", "entropy_asc"],
                value="position",
            )
            export_btn = gr.Button("Export Selected Response CSV")

        export_file = gr.File(label="Exported CSV", interactive=False, visible=False)
        token_html = gr.HTML(label="Selected Response Token Heatmaps")
        token_df = gr.Dataframe(label="Selected Response Token Table", wrap=True, interactive=False)

        with gr.Row():
            high_df = gr.Dataframe(label="Top High-Entropy Tokens (Current Step)", wrap=True, interactive=False)
            low_df = gr.Dataframe(label="Top Low-Entropy Tokens (Current Step)", wrap=True, interactive=False)

        load_btn.click(
            fn=load_trace,
            inputs=[trace_dir_tb, model_path_tb],
            outputs=[load_status, records_state, step_dd, manifest_df],
        )

        inspect_inputs = [
            records_state,
            trace_dir_tb,
            model_path_tb,
            step_dd,
            response_slider,
            min_entropy,
            max_entropy,
            token_query,
            include_invalid,
            top_k,
            sort_by,
        ]
        inspect_outputs = [token_html, prompt_md, response_meta_md, token_df, high_df, low_df]

        step_dd.change(
            fn=on_step_selected,
            inputs=[records_state, trace_dir_tb, step_dd],
            outputs=[step_stats_md, response_slider, min_entropy, max_entropy, response_overview_df],
        ).then(fn=inspect_response, inputs=inspect_inputs, outputs=inspect_outputs)
        response_slider.change(fn=inspect_response, inputs=inspect_inputs, outputs=inspect_outputs)
        min_entropy.change(fn=inspect_response, inputs=inspect_inputs, outputs=inspect_outputs)
        max_entropy.change(fn=inspect_response, inputs=inspect_inputs, outputs=inspect_outputs)
        token_query.change(fn=inspect_response, inputs=inspect_inputs, outputs=inspect_outputs)
        include_invalid.change(fn=inspect_response, inputs=inspect_inputs, outputs=inspect_outputs)
        top_k.change(fn=inspect_response, inputs=inspect_inputs, outputs=inspect_outputs)
        sort_by.change(fn=inspect_response, inputs=inspect_inputs, outputs=inspect_outputs)

        export_btn.click(
            fn=export_selected_csv,
            inputs=[
                records_state,
                trace_dir_tb,
                model_path_tb,
                step_dd,
                response_slider,
                min_entropy,
                max_entropy,
                token_query,
                include_invalid,
                sort_by,
            ],
            outputs=[export_file],
        )

    demo.launch(server_name=args.host, server_port=args.port, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
