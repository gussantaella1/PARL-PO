#!/usr/bin/env bash
# Run the selected KF training and evaluation queues sequentially.
set -Eeuo pipefail

cd "$(dirname "$0")"

PIPELINE_NAME="weekend_pipeline"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${PIPELINE_LOG_ROOT:-logs/${PIPELINE_NAME}_${RUN_ID}}"
MASTER_LOG="${LOG_ROOT}/pipeline.log"
STATUS_FILE="${PIPELINE_STATUS_FILE:-${PIPELINE_NAME}.status}"
LOCK_FILE="${PIPELINE_LOCK_FILE:-${PIPELINE_NAME}.lock}"
WAIT_FOR_EXISTING_JOBS="${WAIT_FOR_EXISTING_JOBS:-true}"
current_stage="startup"

# Every stage is piped through tee. Keep Python's normal progress messages live
# instead of allowing block buffering when stdout is no longer a terminal.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mkdir -p "${LOG_ROOT}"
touch "${MASTER_LOG}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[${PIPELINE_NAME}] Another pipeline owns ${LOCK_FILE}; refusing to start a duplicate."
  exit 1
fi

write_status() {
  local state="$1"
  local detail="$2"

  printf 'state=%s\nstage=%s\ndetail=%s\nrun_id=%s\nlog=%s\nupdated=%s\n' \
    "${state}" "${current_stage}" "${detail}" "${RUN_ID}" "${MASTER_LOG}" "$(date --iso-8601=seconds)" \
    > "${STATUS_FILE}"
}

finish() {
  local status=$?

  if (( status == 0 )); then
    write_status "complete" "all stages completed"
    echo "[${PIPELINE_NAME}] COMPLETE at $(date --iso-8601=seconds)"
  else
    write_status "failed" "exit code ${status}"
    echo "[${PIPELINE_NAME}] FAILED during ${current_stage} with exit code ${status}"
  fi
}
trap finish EXIT

run_stage() {
  local stage="$1"
  shift

  current_stage="${stage}"
  write_status "running" "stage started"
  echo
  echo "[${PIPELINE_NAME}] Starting ${stage} at $(date --iso-8601=seconds)"
  "$@" 2>&1 | tee "${LOG_ROOT}/${stage}.log"
  echo "[${PIPELINE_NAME}] Completed ${stage} at $(date --iso-8601=seconds)"
}

wait_for_existing_jobs() {
  local matching_jobs

  if [[ "${WAIT_FOR_EXISTING_JOBS}" != "true" && "${WAIT_FOR_EXISTING_JOBS}" != "1" ]]; then
    return 0
  fi

  current_stage="waiting_for_existing_jobs"
  while matching_jobs="$(pgrep -af 'evaluate_policy[.]py|rl_loop[.]py|run_eval_sequence_all|run_eval_kf_sequence_all|run_train_selected_training_policies_zero_sum_kf[.]sh' || true)" && [[ -n "${matching_jobs}" ]]; do
    write_status "waiting" "existing training/evaluation job is still active"
    echo "[${PIPELINE_NAME}] Waiting for an existing GPU job to finish:"
    echo "${matching_jobs}"
    sleep 60
  done
}

validate_kf_training_output() {
  local run_dir="Training_Policy_0.5u_1.5vmax_1.0_icVmax_KF_ON"
  local checkpoint
  local missing=0

  current_stage="validate_kf_training"
  for checkpoint in def0_teacher.pt def1_teacher.pt def2_teacher.pt att1_teacher.pt att2_teacher.pt; do
    if [[ ! -s "${run_dir}/${checkpoint}" ]]; then
      echo "[${PIPELINE_NAME}] Missing trained checkpoint: ${run_dir}/${checkpoint}"
      missing=1
    fi
  done

  if grep -Eq 'failed with exit code|was killed by signal' "${LOG_ROOT}/train_selected_kf.log"; then
    echo "[${PIPELINE_NAME}] The training launcher reported a failed or killed training run."
    missing=1
  fi

  if (( missing != 0 )); then
    return 1
  fi
  echo "[${PIPELINE_NAME}] Validated all expected 0.5u KF checkpoints."
}

echo "[${PIPELINE_NAME}] Run ID: ${RUN_ID}"
echo "[${PIPELINE_NAME}] Logs: ${LOG_ROOT}"
write_status "running" "pipeline started"

wait_for_existing_jobs

run_stage "train_selected_kf" \
  ./run_train_selected_training_policies_zero_sum_kf.sh
validate_kf_training_output

run_stage "eval_kf_all" \
  env FORCE_RERUN=true SKIP_COMPLETED_TODAY=false ./run_eval_kf_sequence_all

run_stage "eval_non_kf_advantage" \
  env FORCE_RERUN=true ./run_eval_sequence_all 100m_advantage \
    Training_Policy_0.1u_1vmax_0.05_icVmax:elliptic_ltv \
    Training_Policy_0.5u_1.5vmax_1.0_icVmax:hcw \
    Training_Policy_0.5u_1.5vmax_1.0_icVmax:elliptic_ltv \
    Training_Policy_2.0u_1vmax_1.0_icVmax:hcw \
    Training_Policy_2.0u_1vmax_1.0_icVmax:elliptic_ltv
