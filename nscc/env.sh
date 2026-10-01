# NSCC (ASPIRE2A+/A2AP, PBS, enroot, H100 80GB) site settings. Source this, then use $RUN like run_container.sh.
#   source nscc/env.sh
export ROOT=/scratch/users/astar/ares/sailorhb
export P=$ROOT/git_repos/qwen3_omni_speechllm
export C=$ROOT/container          # CONTAINER_HOME: .sqsh, model, pydeps, caches (big files, not in git)
export CONTAINER_HOME=$C
export DATA_BASE=/data/projects/13003558/zoux/datasets/datasets_mosaic_stage_AudioLLM_v2.1/datasets_multimodal
export MODEL=$(ls -d $C/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/* | head -1)
export PYTHONPATH=$P:$C/pydeps
export OMNI_MDS_CACHE=/tmp/$USER/omni_mds_cache
export MIX=$P/mixes/st_v0.yaml
export RUN="bash $P/container/run_container.sh"
# W&B (train_omni.py --wandb true): compute nodes reach api.wandb.ai through the site proxy (forwarded by the launcher).
# The API key comes from the environment, else from ~/.netrc (`wandb login` writes it); it is never stored in the repo.
if [ -z "${WANDB_API_KEY:-}" ] && [ -r ~/.netrc ]; then
  WANDB_API_KEY=$(awk '$2=="api.wandb.ai"{f=1} f&&$1=="password"{print $2; exit}' ~/.netrc)
  [ -n "$WANDB_API_KEY" ] && export WANDB_API_KEY || unset WANDB_API_KEY
fi
