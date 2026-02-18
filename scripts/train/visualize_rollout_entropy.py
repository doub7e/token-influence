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
        default="output/Archer2.0/Archer2.0-Qwen2.5-1.5B-Math-EntropyTrace-v9/entropy_trace",
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


def _render_token_heatmap(df: pd.DataFrame, lo: float, hi: float) -> str:
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
        tooltip = html.escape(f"pos={pos} id={int(row.token_id)} entropy={entropy:.5f} valid={valid}")
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

        return (
            _compute_step_stats(ent, mask),
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
            return "<div>Load trace first.</div>", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if step is None:
            return "<div>Select a step first.</div>", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        rec = next((x for x in records if int(x["step"]) == int(step)), None)
        if rec is None:
            return "<div>Step not found.</div>", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        step_path = Path(trace_dir_str).expanduser().resolve() / rec["file"]
        payload = _load_step_npz(str(step_path))
        ent = payload["entropies"].astype(np.float32)
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
        token_html = _render_token_heatmap(df if sort_by == "position" else df.sort_values("position"), lo, hi)
        high_df = _flatten_topk(ent, mask, responses, tokenizer, int(top_k), highest=True)
        low_df = _flatten_topk(ent, mask, responses, tokenizer, int(top_k), highest=False)
        return token_html, df, high_df, low_df

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
        token_html, df, _, _ = inspect_response(
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
            # Archer Rollout Entropy Visualizer
            Inspect token-level entropy for every rollout step and response.
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
        token_html = gr.HTML(label="Selected Response Token Entropy Heatmap")
        token_df = gr.Dataframe(label="Selected Response Token Table", wrap=True, interactive=False)

        with gr.Row():
            high_df = gr.Dataframe(label="Top High-Entropy Tokens (Current Step)", wrap=True, interactive=False)
            low_df = gr.Dataframe(label="Top Low-Entropy Tokens (Current Step)", wrap=True, interactive=False)

        load_btn.click(
            fn=load_trace,
            inputs=[trace_dir_tb, model_path_tb],
            outputs=[load_status, records_state, step_dd, manifest_df],
        )

        step_dd.change(
            fn=on_step_selected,
            inputs=[records_state, trace_dir_tb, step_dd],
            outputs=[step_stats_md, response_slider, min_entropy, max_entropy, response_overview_df],
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
        inspect_outputs = [token_html, token_df, high_df, low_df]

        step_dd.change(fn=inspect_response, inputs=inspect_inputs, outputs=inspect_outputs)
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
