#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ ! -f /scratch/cvlab/home/shuli/.codex/secrets/wandb_api_key ]; then
  echo "[FATAL] wandb key file not found: /scratch/cvlab/home/shuli/.codex/secrets/wandb_api_key" >&2
  exit 2
fi

export WANDB_API_KEY="$(cat /scratch/cvlab/home/shuli/.codex/secrets/wandb_api_key)"
export WANDB_MODE=online

summary_root="output/influence_ablation_0220"
log_dir="${summary_root}/logs"
mkdir -p "${log_dir}"

summary_csv="${summary_root}/experiment_cases.csv"
printf "case_id,mode,scope,exp_name,log_path,offload_to_cpu,status\n" > "${summary_csv}"

run_influence_case() {
  local case_id="$1"
  local mode="$2"
  local scope="$3"
  local exp_base="$4"
  local offload="False"

  while true; do
    local exp_name="${exp_base}"
    if [ "${offload}" = "True" ]; then
      exp_name="${exp_base}-offcpu"
    fi
    local log_path="${log_dir}/${exp_name}.log"

    export INFLUENCE_EXP_NAME="${exp_name}"
    export INFLUENCE_TOTAL_STEPS=1
    export INFLUENCE_PPO_EPOCHS=1
    export INFLUENCE_TOTAL_EPOCHS=1
    export INFLUENCE_PROJECTION_DIM_FACTOR=128
    export INFLUENCE_OUTPUT_FUNCTION="${mode}"
    export INFLUENCE_ACCEPTED_REJECTED_SCOPE="${scope}"
    export INFLUENCE_GRAD_OFFLOAD_TO_CPU="${offload}"

    : > "${log_path}"
    echo "[RUN] case=${case_id} mode=${mode} scope=${scope} offload=${offload} exp=${exp_name}" | tee -a "${log_path}"
    echo "[CFG] steps=${INFLUENCE_TOTAL_STEPS} factor=${INFLUENCE_PROJECTION_DIM_FACTOR} output_function=${INFLUENCE_OUTPUT_FUNCTION} scope=${INFLUENCE_ACCEPTED_REJECTED_SCOPE}" | tee -a "${log_path}"

    set +e
    bash scripts/train/run_archer2.0_qwen2.5_1.5b_math_influence_trace.sh 2>&1 | tee -a "${log_path}"
    rc=${PIPESTATUS[0]}
    set -e

    if [ "${rc}" -eq 0 ]; then
      printf "%s,%s,%s,%s,%s,%s,%s\n" "${case_id}" "${mode}" "${scope}" "${exp_name}" "${log_path}" "${offload}" "ok" >> "${summary_csv}"
      return 0
    fi

    if [ "${offload}" = "False" ] && grep -Eqi "out of memory|cuda out of memory|oom" "${log_path}"; then
      echo "[WARN] OOM detected for ${exp_name}, retry with grad offload to cpu." | tee -a "${log_path}"
      offload="True"
      continue
    fi

    printf "%s,%s,%s,%s,%s,%s,%s\n" "${case_id}" "${mode}" "${scope}" "${exp_name}" "${log_path}" "${offload}" "failed" >> "${summary_csv}"
    return "${rc}"
  done
}

run_baseline_case() {
  local case_id="$1"
  local exp_name="$2"
  local log_path="${log_dir}/${exp_name}.log"
  local ckpt_dir="./output/Archer2.0/${exp_name}"
  mkdir -p "${ckpt_dir}"

  : > "${log_path}"
  echo "[RUN] case=${case_id} baseline exp=${exp_name}" | tee -a "${log_path}"

  set +e
  bash scripts/train/run_archer2.0_qwen2.5_1.5b_math.sh \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir="${ckpt_dir}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    +trainer.entropy_trace.enable=False \
    +trainer.influence_trace.enable=False \
    2>&1 | tee -a "${log_path}"
  rc=${PIPESTATUS[0]}
  set -e

  if [ "${rc}" -eq 0 ]; then
    printf "%s,%s,%s,%s,%s,%s,%s\n" "${case_id}" "baseline" "na" "${exp_name}" "${log_path}" "False" "ok" >> "${summary_csv}"
    return 0
  fi
  printf "%s,%s,%s,%s,%s,%s,%s\n" "${case_id}" "baseline" "na" "${exp_name}" "${log_path}" "False" "failed" >> "${summary_csv}"
  return "${rc}"
}

run_influence_case "1" "training_loss" "per_prompt" "archer-infl-abl-1-trainloss-prompt-f128-0220"
run_influence_case "2" "training_loss" "all_selected" "archer-infl-abl-2-trainloss-all-f128-0220"
run_influence_case "3" "log_prob_advantage" "per_prompt" "archer-infl-abl-3-logpadv-prompt-f128-0220"
run_influence_case "4" "log_prob_advantage" "all_selected" "archer-infl-abl-4-logpadv-all-f128-0220"
run_baseline_case "5" "archer-infl-abl-5-baseline-f128-0220"

python - <<'PY'
import csv
import json
import re
from pathlib import Path

root = Path("output/influence_ablation_0220")
cases = list(csv.DictReader((root / "experiment_cases.csv").open()))

metric_keys = [
    "timing_s/step",
    "timing_s/update_actor",
    "timing_s/gen",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "influence_trace/selected_prompts",
    "influence_trace/selected_responses",
]


def parse_last_metric(text: str, key: str):
    vals = []
    key_pat = re.escape(key)
    pat = re.compile(key_pat + r"[^0-9+\-eE]*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
    for m in pat.finditer(text):
        try:
            vals.append(float(m.group(1)))
        except ValueError:
            pass
    return vals[-1] if vals else None


rows = []
for c in cases:
    p = Path(c["log_path"])
    txt = p.read_text(errors="ignore") if p.exists() else ""
    row = dict(c)
    row["oom_detected"] = bool(re.search(r"out of memory|cuda out of memory|\boom\b", txt, flags=re.IGNORECASE))
    row["trace_written"] = "No trace records found" not in txt
    for k in metric_keys:
        row[k] = parse_last_metric(txt, k)
    rows.append(row)

(root / "results_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

headers = [
    "case_id",
    "mode",
    "scope",
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

(root / "results_summary.md").write_text("\n".join(lines) + "\n")
print(root / "results_summary.md")
PY
