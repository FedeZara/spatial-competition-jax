#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10


# PSRO runs
for cfg in conv_val_high conv_val_med conv_val_low conv; do
  python scripts/run_psro.py \
    --config "configs/psro_hotelling_1d_discrete_${cfg}.yaml" \
    --experiment-name "psro_${cfg}" \
    > "logs/psro_${cfg}.log" 2>&1 &
done
