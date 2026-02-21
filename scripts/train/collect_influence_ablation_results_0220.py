#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path


ROOT = Path("output")
IN_FILES = [
    ROOT / "influence_ablation_0220" / "experiment_cases.csv",
    ROOT / "influence_ablation_0220_extra_nohessian" / "experiment_cases.csv",
]
OUT_DIR = ROOT / "influence_ablation_0220_all"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_KEYS = [
    "timing_s/step",
    "timing_s/update_actor",
    "timing_s/gen",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "perf/cpu_memory_used_gb",
    "influence_trace/selected_prompts",
    "influence_trace/selected_responses",
]


def parse_last_metric(text: str, key: str):
    vals = []
    pat = re.compile(re.escape(key) + r"[^0-9+\-eE]*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
    for m in pat.finditer(text):
        try:
            vals.append(float(m.group(1)))
        except ValueError:
            pass
    return vals[-1] if vals else None


rows = []
for f in IN_FILES:
    if not f.exists():
        continue
    with f.open() as fp:
        for row in csv.DictReader(fp):
            log_path = Path(row["log_path"])
            text = log_path.read_text(errors="ignore") if log_path.exists() else ""
            item = dict(row)
            if "hessian_mode" not in item:
                item["hessian_mode"] = "inverse" if item.get("mode") != "baseline" else "na"
            item["oom_detected"] = bool(re.search(r"out of memory|cuda out of memory|\boom\b", text, flags=re.IGNORECASE))
            item["trace_written"] = "No trace records found" not in text
            for k in METRIC_KEYS:
                item[k] = parse_last_metric(text, k)
            rows.append(item)

rows.sort(key=lambda x: int(str(x.get("case_id", "999"))))

(OUT_DIR / "results_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

headers = [
    "case_id",
    "mode",
    "scope",
    "hessian_mode",
    "exp_name",
    "status",
    "offload_to_cpu",
    "oom_detected",
    "timing_s/step",
    "timing_s/update_actor",
    "perf/max_memory_reserved_gb",
    "influence_trace/selected_prompts",
    "influence_trace/selected_responses",
    "log_path",
]

lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
for r in rows:
    vals = []
    for h in headers:
        v = r.get(h)
        if isinstance(v, float):
            vals.append(f"{v:.6g}")
        else:
            vals.append(str(v))
    lines.append("| " + " | ".join(vals) + " |")
(OUT_DIR / "results_summary.md").write_text("\n".join(lines) + "\n")

print(OUT_DIR / "results_summary.md")
