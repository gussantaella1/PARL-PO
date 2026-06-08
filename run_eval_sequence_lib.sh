#!/usr/bin/env bash

# Shared helpers for the Monte Carlo evaluation sequence scripts.
#
# The distance-specific launchers source this file so they agree on dynamics
# naming, run-spec parsing, output suffixes, and warning behavior. Run specs
# accept either of these forms:
#
#   Training_Policy_2.0u_1vmax_1.0_icVmax:elliptic_ltv
#   Training_Policy_2.0u_1vmax_1.0_icVmax|elliptic_ltv

trim_whitespace() {
  # Bash array parsing leaves whitespace exactly as typed, so normalize specs
  # before comparing names or building output folder suffixes.
  local value="${1:-}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s\n' "${value}"
}

canonicalize_dyn_name() {
  # Keep all launchers using the same spellings. This makes folder suffixes and
  # evaluate_policy.py overrides predictable even when I type a variant by hand.
  local dyn
  dyn="$(trim_whitespace "${1:-}")"

  case "${dyn}" in
    "")
      printf '%s\n' ""
      ;;
    HCW|hcw)
      printf '%s\n' "hcw"
      ;;
    ELLIPTIC_LTV|Elliptic_LTV|elliptic_ltv)
      printf '%s\n' "elliptic_ltv"
      ;;
    *)
      printf '%s\n' "${dyn}"
      ;;
  esac
}

format_dynamics_label() {
  # Log HCW explicitly as the default so queue output is easy to read later.
  local dyn
  dyn="$(canonicalize_dyn_name "${1:-}")"
  if [[ -n "${dyn}" ]]; then
    printf '%s\n' "${dyn}"
  else
    printf '%s\n' "hcw(default)"
  fi
}

parse_run_spec() {
  # Split a run spec into "folder" and optional "dynamics" without forcing every
  # caller to remember the delimiter details.
  local spec="$1"
  local -n out_run_dir_ref="$2"
  local -n out_dyn_ref="$3"
  local delim=""

  if [[ "${spec}" == *"|"* ]]; then
    delim="|"
  elif [[ "${spec}" == *":"* ]]; then
    delim=":"
  fi

  if [[ -n "${delim}" ]]; then
    out_run_dir_ref="${spec%%${delim}*}"
    out_dyn_ref="${spec#*${delim}}"
  else
    out_run_dir_ref="${spec}"
    out_dyn_ref=""
  fi

  out_run_dir_ref="$(trim_whitespace "${out_run_dir_ref}")"
  out_dyn_ref="$(canonicalize_dyn_name "${out_dyn_ref}")"
}

detect_common_args_dynamics() {
  # If COMMON_ARGS already contains --dynamics, remember it before stripping the
  # flag out. Per-run dynamics and DYNAMICS_OVERRIDE can still take precedence.
  local -n args_ref="$1"
  local detected=""
  local arg=""
  local i=0

  for ((i=0; i<${#args_ref[@]}; i++)); do
    arg="${args_ref[i]}"
    if [[ "${arg}" == "--dynamics" ]]; then
      if (( i + 1 < ${#args_ref[@]} )); then
        detected="${args_ref[i+1]}"
        ((i++))
      fi
      continue
    fi
    if [[ "${arg}" == --dynamics=* ]]; then
      detected="${arg#--dynamics=}"
    fi
  done

  canonicalize_dyn_name "${detected}"
}

strip_dynamics_args() {
  # Dynamics is injected per run after precedence is resolved, so remove any
  # common dynamics flag before appending the effective value.
  local -n src_ref="$1"
  local -n dst_ref="$2"
  local arg=""
  local i=0

  dst_ref=()
  for ((i=0; i<${#src_ref[@]}; i++)); do
    arg="${src_ref[i]}"
    if [[ "${arg}" == "--dynamics" ]]; then
      if (( i + 1 < ${#src_ref[@]} )); then
        ((i++))
      fi
      continue
    fi
    if [[ "${arg}" == --dynamics=* ]]; then
      continue
    fi
    dst_ref+=("${arg}")
  done
}

resolve_effective_dynamics() {
  # Precedence is: explicit environment override, per-run spec, then COMMON_ARGS.
  local run_dyn override_dyn common_dyn
  run_dyn="$(canonicalize_dyn_name "${1:-}")"
  override_dyn="$(canonicalize_dyn_name "${2:-}")"
  common_dyn="$(canonicalize_dyn_name "${3:-}")"

  if [[ -n "${override_dyn}" ]]; then
    printf '%s\n' "${override_dyn}"
  elif [[ -n "${run_dyn}" ]]; then
    printf '%s\n' "${run_dyn}"
  else
    printf '%s\n' "${common_dyn}"
  fi
}

auto_suffix_for_dynamics() {
  # HCW is the no-suffix baseline; non-HCW evals get a suffix so outputs do not
  # silently overwrite the default folders.
  local dyn
  dyn="$(canonicalize_dyn_name "${1:-}")"
  if [[ -z "${dyn}" || "${dyn}" == "hcw" ]]; then
    printf '%s\n' ""
  else
    printf '%s\n' "_${dyn}"
  fi
}

warn_if_non_hcw_kf_request() {
  # The estimator path is intentionally HCW-only in this rollout stack. Keep this
  # warning centralized so all eval launchers explain the same caveat.
  local dyn
  local use_kf="${2:-false}"
  local script_name="${3:-run_eval_sequence}"

  dyn="$(canonicalize_dyn_name "${1:-}")"
  if [[ "${use_kf}" == "true" && -n "${dyn}" && "${dyn}" != "hcw" ]]; then
    echo "[${script_name}] Warning: use_kf=true requested with dynamics=${dyn}, but the current rollout stack disables the estimator for non-HCW dynamics." >&2
  fi
}
