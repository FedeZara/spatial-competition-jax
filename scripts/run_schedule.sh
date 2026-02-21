#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Experiment scheduler: runs jobs on 4 GPUs, queuing the next
# experiment as soon as a GPU becomes free.
# ──────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

# ── Define all experiments ────────────────────────────────────
# Format: "script|config|experiment_name"

JOBS=(
  # ── 1D MAPPO (5) ────────────────────────────────────────────
  "train|hotelling_1d_discrete.yaml|1d_disc_baseline"
  "train|hotelling_1d_discrete_val_high.yaml|1d_disc_val_high"
  "train|hotelling_1d_discrete_val_med.yaml|1d_disc_val_med"
  "train|hotelling_1d_discrete_val_low.yaml|1d_disc_val_low"
  "train|hotelling_1d_discrete_quadratic.yaml|1d_disc_quadratic"

  # ── 2D MAPPO (5) ────────────────────────────────────────────
  "train|hotelling_2d_conv.yaml|2d_2seller_uniform"
  "train|hotelling_2d_conv_3sellers.yaml|2d_3seller_uniform"
  "train|hotelling_2d_conv_3sellers_mixture.yaml|2d_3seller_mixture"
  "train|hotelling_2d_conv_gaussian_center.yaml|2d_2seller_gaussian_center"
  "train|hotelling_2d_conv_mixture.yaml|2d_2seller_gaussian_mixture"

  # ── 1D Asymmetric PSRO (4) ─────────────────────────────────
  "psro_asym|psro_hotelling_1d_discrete.yaml|psro_asym_1d_baseline"
  "psro_asym|psro_hotelling_1d_discrete_val_high.yaml|psro_asym_1d_val_high"
  "psro_asym|psro_hotelling_1d_discrete_val_med.yaml|psro_asym_1d_val_med"
  "psro_asym|psro_hotelling_1d_discrete_val_low.yaml|psro_asym_1d_val_low"
)

NUM_GPUS=4

# ── GPU tracker ───────────────────────────────────────────────
# gpu_pid[i] = PID of process running on GPU i (0 = free)
declare -a gpu_pid
for ((g=0; g<NUM_GPUS; g++)); do
  gpu_pid[$g]=0
done

wait_for_gpu() {
  # Block until at least one GPU is free; return its index
  while true; do
    for ((g=0; g<NUM_GPUS; g++)); do
      pid=${gpu_pid[$g]}
      if [[ $pid -eq 0 ]] || ! kill -0 "$pid" 2>/dev/null; then
        gpu_pid[$g]=0
        echo "$g"
        return
      fi
    done
    sleep 10
  done
}

launch_job() {
  local gpu=$1
  local script_type=$2
  local config=$3
  local name=$4
  local logfile="logs/${name}.log"

  if [[ "$script_type" == "train" ]]; then
    local cmd="python scripts/train_hotelling.py --config configs/${config} --device gpu:${gpu} --experiment-name ${name}"
  elif [[ "$script_type" == "psro_asym" ]]; then
    local cmd="python scripts/run_psro_asymmetric.py --config configs/${config} --device gpu:${gpu} --experiment-name ${name}"
  else
    echo "Unknown script type: $script_type"
    return 1
  fi

  echo "[$(date +%H:%M:%S)] GPU ${gpu} ← ${name} (${config})"
  $cmd > "$logfile" 2>&1 &
  gpu_pid[$gpu]=$!
}

# ── Main loop ─────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "Experiment Scheduler — ${#JOBS[@]} jobs on ${NUM_GPUS} GPUs"
echo "═══════════════════════════════════════════════════════════"

for job in "${JOBS[@]}"; do
  IFS='|' read -r script_type config name <<< "$job"
  gpu=$(wait_for_gpu)
  launch_job "$gpu" "$script_type" "$config" "$name"
done

# Wait for all remaining jobs
echo ""
echo "All jobs launched. Waiting for remaining to finish …"
for ((g=0; g<NUM_GPUS; g++)); do
  pid=${gpu_pid[$g]}
  if [[ $pid -ne 0 ]] && kill -0 "$pid" 2>/dev/null; then
    wait "$pid" 2>/dev/null || true
  fi
done

echo "═══════════════════════════════════════════════════════════"
echo "All ${#JOBS[@]} experiments complete!"
echo "═══════════════════════════════════════════════════════════"
