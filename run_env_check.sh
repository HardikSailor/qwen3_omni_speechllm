#!/bin/bash
# Run the Qwen3-Omni inference check in each candidate conda env (run on a GPU node).
#   bash run_env_check.sh                 # default env: nemo_env (flash-attn 2)
#   bash run_env_check.sh py2hf --talker  # one env, extra args passed to the python script
cd "$(dirname "$0")"
CONDA_ROOT=/home/users/astar/i2r/sailorhb/miniconda3
export HF_HOME=/scratch/prj0000000234/sailorhb/hf_models
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1  # envs must be self-contained; ignore ~/.local site-packages
mkdir -p logs

if [ $# -gt 0 ] && [ -d "$CONDA_ROOT/envs/$1" ]; then ENVS=("$1"); shift; else ENVS=(nemo_env); fi

for env in "${ENVS[@]}"; do
    echo "################ $env"
    CONDA_DEFAULT_ENV=$env "$CONDA_ROOT/envs/$env/bin/python" infer_qwen3_omni.py "$@" 2>&1 | tee "logs/infer_${env}.log"
done
grep -h -A6 "##### SUMMARY" logs/infer_*.log
