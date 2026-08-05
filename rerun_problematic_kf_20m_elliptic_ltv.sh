#!/usr/bin/env bash
#
# Rerun only the suspicious KF-on 20 m elliptic-LTV policy matchups without
# touching the existing MC_eval_20m_elliptic_ltv data used by the thesis.
#
# Optional overrides:
#   DEVICE=cuda|cpu|mps
#   VERIFY_ROOT=Training_Policy_.../MC_eval_20m_elliptic_ltv_RERUN_manual
#   INDEX=15000
#   PREWARM_ELLIPTIC_LTV_CACHE=false
set -euo pipefail

cd "$(dirname "$0")"
source ./run_eval_sequence_lib.sh

SCRIPT_NAME="$(basename "$0")"
RUN_DIR="Training_Policy_0.1u_1vmax_0.05_icVmax_KF_ON"
DYNAMICS="elliptic_ltv"
ARENA_RADIUS="20.0"
STEPS="${STEPS:-6000}"
DT="${DT:-0.1}"
INDEX="${INDEX:-15000}"
DEVICE="${DEVICE:-cuda}"
VERIFY_ROOT="${VERIFY_ROOT:-${RUN_DIR}/MC_eval_20m_elliptic_ltv_RERUN_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[${SCRIPT_NAME}] Missing run directory: ${RUN_DIR}" >&2
  exit 1
fi

if [[ -e "${VERIFY_ROOT}" ]]; then
  echo "[${SCRIPT_NAME}] Refusing to write into existing output path: ${VERIFY_ROOT}" >&2
  echo "[${SCRIPT_NAME}] Set VERIFY_ROOT to a new path and rerun." >&2
  exit 1
fi

if [[ "${PREWARM_ELLIPTIC_LTV_CACHE:-true}" != "false" && "${PREWARM_ELLIPTIC_LTV_CACHE:-1}" != "0" ]]; then
  prewarm_elliptic_ltv_cache "config_rl" "${STEPS}" "${DT}" "${SCRIPT_NAME}"
fi

COMMON_ARGS=(
  --run_dir "${RUN_DIR}"
  --auto_shell_grid
  --grid_mode cartesian
  --index "${INDEX}"
  --shell_fracs 0.2,0.4,0.6,0.8
  --steps "${STEPS}"
  --points_per_shell 40
  --arena_radius "${ARENA_RADIUS}"
  --dynamics "${DYNAMICS}"
  --velocity_controller_enabled true
  --advantage_scale 1.0
  --save_rollout_error_cases
  --device "${DEVICE}"
  --use_kf true
)

run_case() {
  local case_name="$1"
  local def_ckpt="$2"
  local att_ckpt="$3"
  local out_dir="${VERIFY_ROOT}/${case_name}"

  echo "[${SCRIPT_NAME}] Starting ${case_name} -> ${out_dir}"
  python evaluate_policy.py \
    "${COMMON_ARGS[@]}" \
    --def_ckpt_path "${RUN_DIR}/${def_ckpt}" \
    --att_ckpt_path "${RUN_DIR}/${att_ckpt}" \
    --out_dir "${out_dir}"
}

run_case "def0_vs_att1" "def0_teacher.pt" "att1_teacher.pt"
run_case "def1_vs_att1" "def1_teacher.pt" "att1_teacher.pt"
run_case "def1_vs_att2" "def1_teacher.pt" "att2_teacher.pt"
run_case "def2_vs_att2" "def2_teacher.pt" "att2_teacher.pt"

python - "${VERIFY_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print(f"[summary] Results written under: {root}")
for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    results_path = case_dir / "results.json"
    if not results_path.exists():
        print(f"[summary] {case_dir.name}: missing results.json")
        continue
    data = json.loads(results_path.read_text())
    timing = (data.get("metrics_trial", {}) or {}).get("rollout_total_sec_per_step", {}) or {}
    q25 = timing.get("q25")
    q50 = timing.get("q50")
    q75 = timing.get("q75")
    if q50 is None:
        print(f"[summary] {case_dir.name}: no rollout_total_sec_per_step timing found")
        continue
    print(
        f"[summary] {case_dir.name}: "
        f"{1000.0 * q50:.4f} ms"
        + (f" [{1000.0 * q25:.4f}, {1000.0 * q75:.4f}]" if q25 is not None and q75 is not None else "")
    )
PY

echo "[${SCRIPT_NAME}] Done"
