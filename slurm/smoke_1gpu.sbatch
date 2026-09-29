#!/bin/bash
#SBATCH --job-name=omni_smoke1
#SBATCH --partition=h200p
#SBATCH --account=prj0000000234
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/prj0000000234/sailorhb/toolkits/qwen3_omni_speechllm/outputs/smoke_omni/slurm-%j.out
#
# Phase 3 smoke test (PIPELINE_PLAN.md §6): train_omni.py sft on mixes/st_v0.yaml, 1 GPU, LoRA r16 on thinker
# attention + audio proj1/proj2 (explicit 194-module list), micro-batch 1 x grad-accum 4, padding-free.
#   A  40 steps, checkpoint + eval (6 CoVoST2 sets x 32) at step 20 and 40
#   B  resume from A/checkpoint-20, run to 40: losses 21..40 should match A (data position from omni_mosaic_state.json)
#   C  OMNI_GPU_MEM_GB=80 (H100 budget on an H200), 20 steps, no eval
# Submit: sbatch slurm/smoke_1gpu.sbatch        Summary: outputs/smoke_omni/summary.txt
set -u
P=/scratch/prj0000000234/sailorhb/toolkits/qwen3_omni_speechllm
O=$P/outputs/smoke_omni
RUN="bash /scratch/prj0000000234/sailorhb/container/run_container.sh"
MODEL=$(ls -d /scratch/prj0000000234/sailorhb/hf_models/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/* | head -1)
export PYTHONPATH=$P:/scratch/prj0000000234/sailorhb/toolkits/pydeps
export OMNI_MDS_CACHE=/tmp/$USER/omni_mds_cache
export NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0
cd $P
echo "node $(hostname)  start $(date)"; nvidia-smi -L; df -h /tmp | tail -1

COMMON=(--mix mixes/st_v0.yaml --val_choose 32
    --model "$MODEL" --model_type qwen3_omni_moe
    --tuner_type lora --target_modules $(cat lora_targets_thinker_attn_audio_proj.txt) --lora_rank 16 --lora_alpha 32
    --torch_dtype bfloat16 --attn_impl flash_attn --padding_free true
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4
    --learning_rate 1e-4 --warmup_ratio 0.05 --gradient_checkpointing true --max_length 2048
    --logging_steps 1 --dataloader_num_workers 4 --report_to none --data_seed 42 --seed 42
    --save_only_model false --save_total_limit 3)

echo "=== A $(date)"
$RUN python train_omni.py sft "${COMMON[@]}" --max_steps 40 --save_steps 20 --eval_steps 20 \
    --output_dir $O/A > $O/A.log 2>&1; echo "A exit=$?"
CK=$(ls -d $O/A/v*/checkpoint-20 2>/dev/null | head -1)
echo "A checkpoint-20: $CK"; ls $CK 2>/dev/null; cat $CK/omni_mosaic_state.json 2>/dev/null; echo

echo "=== B (resume) $(date)"
if [ -n "$CK" ]; then
  $RUN python train_omni.py sft "${COMMON[@]}" --max_steps 40 --save_steps 20 --eval_steps 20 \
      --resume_from_checkpoint $CK --output_dir $O/B > $O/B.log 2>&1; echo "B exit=$?"
fi

echo "=== C (80 GB cap) $(date)"
OMNI_GPU_MEM_GB=80 $RUN python train_omni.py sft "${COMMON[@]}" --max_steps 20 --save_steps 100000 \
    --eval_strategy no --output_dir $O/C > $O/C.log 2>&1; echo "C exit=$?"

echo "end $(date)"
$RUN python tests/summarize_smoke.py $O > $O/summary.txt 2>&1; cat $O/summary.txt
