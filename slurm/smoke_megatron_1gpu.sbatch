#!/bin/bash
#SBATCH --job-name=omni_smokeM1
#SBATCH --partition=h200p
#SBATCH --account=prj0000000234
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/prj0000000234/sailorhb/toolkits/qwen3_omni_speechllm/outputs/smoke_megatron/slurm-%j.out
#
# Phase 7 smoke test on 1 GPU (PIPELINE_PLAN.md §6): train_omni.py megatron on mixes/st_v0.yaml.
# LoRA r16 on thinker attention (linear_qkv, linear_proj), encoders/aligner frozen, EP=1 (1 GPU), micro-batch 1,
# global batch 4 (= 4 micro-batches, same as the HF smoke: batch 1 x grad-accum 4), padding-free.
#   MA  20 iterations, eval (6 CoVoST2 sets x 32) + checkpoint (with optimizer, resumable) at 10 and 20
#   MB  resume MA's iteration-10 checkpoint (--mcore_adapter ... --finetune false), run to 20: losses 11..20 ~ MA
#   MC  OMNI_GPU_MEM_GB=80, 10 iterations, no eval / save
# Submit: sbatch slurm/smoke_megatron_1gpu.sbatch     Summary: outputs/smoke_megatron/summary.txt
set -u
P=/scratch/prj0000000234/sailorhb/toolkits/qwen3_omni_speechllm
O=$P/outputs/smoke_megatron
RUN="bash /scratch/prj0000000234/sailorhb/container/run_container.sh"
MODEL=$(ls -d /scratch/prj0000000234/sailorhb/hf_models/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/* | head -1)
export PYTHONPATH=$P:/scratch/prj0000000234/sailorhb/toolkits/pydeps
export OMNI_MDS_CACHE=/tmp/$USER/omni_mds_cache
export NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29561
mkdir -p $O; cd $P
echo "node $(hostname)  start $(date)"; nvidia-smi -L

COMMON=(--mix mixes/st_v0.yaml --val_choose 32
    --model "$MODEL" --model_type qwen3_omni_moe
    --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules linear_qkv linear_proj
    --freeze_llm false --freeze_vit true --freeze_aligner true
    --expert_model_parallel_size 1 --moe_grouped_gemm true --moe_permute_fusion true --moe_aux_loss_coeff 1e-3
    --micro_batch_size 1 --global_batch_size 4 --lr 1e-4 --min_lr 1e-5 --lr_warmup_fraction 0.05
    --recompute_granularity selective --cross_entropy_loss_fusion true
    --max_length 2048 --attention_backend flash
    --merge_lora false --logging_steps 1 --dataloader_num_workers 4 --data_seed 42 --seed 42)

echo "=== MA $(date)"
$RUN python train_omni.py megatron "${COMMON[@]}" --train_iters 20 --save_steps 10 --eval_steps 10 \
    --finetune true --no_save_optim false --no_save_rng false --save_safetensors true \
    --output_dir $O/MA > $O/MA.log 2>&1; echo "MA exit=$?"
echo "--- MA layout"; find $O/MA -maxdepth 4 \( -name 'latest_checkpointed_iteration.txt' -o -name 'omni_mosaic_state.json' -o -name 'checkpoint-*' -o -name 'iter_*' \) | sort
for f in $(find $O/MA -name omni_mosaic_state.json | sort); do echo "$f: $(cat $f)"; done
CK=$(dirname $(find $O/MA -name latest_checkpointed_iteration.txt | head -1) 2>/dev/null)
CK10=$(find $O/MA -maxdepth 3 -type d -name 'checkpoint-10' | head -1)
echo "tracker dir: $CK   checkpoint-10: $CK10"

echo "=== MB (resume) $(date)"
if [ -n "$CK10" ]; then
  $RUN python train_omni.py megatron "${COMMON[@]}" --train_iters 20 --save_steps 10 --eval_steps 10 \
      --mcore_adapter $CK10 --finetune false --no_load_optim false --no_load_rng false \
      --no_save_optim true --save_safetensors true --output_dir $O/MB > $O/MB.log 2>&1; echo "MB exit=$?"
fi

echo "=== MC (80 GB cap) $(date)"
OMNI_GPU_MEM_GB=80 $RUN python train_omni.py megatron "${COMMON[@]}" --train_iters 10 --save_steps 100000 \
    --eval_iters 0 --finetune true --no_save_optim true --output_dir $O/MC > $O/MC.log 2>&1; echo "MC exit=$?"

echo "end $(date)"
$RUN python tests/summarize_smoke.py $O MA MB MC > $O/summary.txt 2>&1; cat $O/summary.txt
