#!/bin/bash
#SBATCH --job-name=omni_8gpu
#SBATCH --partition=h200p
#SBATCH --account=prj0000000234
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/prj0000000234/sailorhb/toolkits/qwen3_omni_speechllm/outputs/train_8gpu/slurm-%j.out
#
# Phase 4 + 7 (PIPELINE_PLAN.md §6): both trainers on 1 node x 8 H200, mixes/st_v0.yaml, global batch 32, padding-free.
#   H8A  train_omni.py sft (HF DDP): LoRA r16 on thinker attention + audio proj1/proj2 (194 modules), micro-batch 1 x
#        8 GPUs x grad-accum 4, 60 steps, eval + checkpoint at 30/60
#   H8B  resume H8A step 30 -> 60: losses 31..60 must match H8A (data position = 30 x 32 = 960 samples)
#   M8A  train_omni.py megatron EP=8: LoRA r16 on attention (linear_qkv, linear_proj), micro-batch 1, global 32,
#        60 iterations, eval + resumable checkpoint at 30/60
#   M8B  resume M8A iteration 30 with optimizer (--no_save_optim false, see §10.4) -> 60
# H8A and M8A log the sample ids each rank reads (OMNI_LOG_SAMPLE_IDS) -> tests/check_partitions.py
# Summary: outputs/train_8gpu/summary.txt
set -u
P=/scratch/prj0000000234/sailorhb/toolkits/qwen3_omni_speechllm
O=$P/outputs/train_8gpu
RUN="bash /scratch/prj0000000234/sailorhb/container/run_container.sh"
MODEL=$(ls -d /scratch/prj0000000234/sailorhb/hf_models/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/* | head -1)
export PYTHONPATH=$P:/scratch/prj0000000234/sailorhb/toolkits/pydeps
export OMNI_MDS_CACHE=/tmp/$USER/omni_mds_cache
export NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p $O; cd $P
echo "node $(hostname)  start $(date)"; nvidia-smi -L; df -h /tmp | tail -1

HF=(--mix mixes/st_v0.yaml --val_choose 32
    --model "$MODEL" --model_type qwen3_omni_moe --experts_impl grouped_mm
    --tuner_type lora --target_modules $(cat lora_targets_thinker_attn_audio_proj.txt) --lora_rank 16 --lora_alpha 32
    --torch_dtype bfloat16 --attn_impl flash_attn --padding_free true
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4
    --learning_rate 1e-4 --warmup_ratio 0.05 --gradient_checkpointing true --max_length 2048
    --logging_steps 1 --dataloader_num_workers 4 --report_to none --data_seed 42 --seed 42
    --save_only_model false --save_total_limit 2 --max_steps 60 --save_steps 30 --eval_steps 30)
MEG=(--mix mixes/st_v0.yaml --val_choose 32
    --model "$MODEL" --model_type qwen3_omni_moe
    --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules linear_qkv linear_proj
    --freeze_llm false --freeze_vit true --freeze_aligner true
    --expert_model_parallel_size 8 --moe_grouped_gemm true --moe_permute_fusion true --moe_aux_loss_coeff 1e-3
    --micro_batch_size 1 --global_batch_size 32 --lr 1e-4 --min_lr 1e-5 --lr_warmup_fraction 0.05
    --recompute_granularity selective --cross_entropy_loss_fusion true --max_length 2048 --attention_backend flash
    --merge_lora false --save_total_limit 2 --no_save_optim false --logging_steps 1 --dataloader_num_workers 4
    --data_seed 42 --seed 42 --train_iters 60 --save_steps 30 --eval_steps 30)

echo "=== H8A $(date)"
OMNI_LOG_SAMPLE_IDS=$O/ids_H8A MASTER_PORT=29571 $RUN python train_omni.py sft "${HF[@]}" --output_dir $O/H8A \
    > $O/H8A.log 2>&1; echo "H8A exit=$?"
CK=$(ls -d $O/H8A/v*/checkpoint-30 2>/dev/null | head -1); echo "H8A checkpoint-30: $CK $(cat $CK/omni_mosaic_state.json 2>/dev/null)"
echo "=== H8B (resume) $(date)"
[ -n "$CK" ] && { MASTER_PORT=29572 $RUN python train_omni.py sft "${HF[@]}" --resume_from_checkpoint $CK \
    --output_dir $O/H8B > $O/H8B.log 2>&1; echo "H8B exit=$?"; }

echo "=== M8A $(date)"
OMNI_LOG_SAMPLE_IDS=$O/ids_M8A MASTER_PORT=29573 $RUN python train_omni.py megatron "${MEG[@]}" --finetune true \
    --output_dir $O/M8A > $O/M8A.log 2>&1; echo "M8A exit=$?"
CK=$(ls -d $O/M8A/v*/checkpoint-30 2>/dev/null | head -1); echo "M8A checkpoint-30: $CK $(cat $CK/omni_mosaic_state.json 2>/dev/null)"
grep -h "eval_iters\|train_iters" $O/M8A.log | grep -i setting | head -3
echo "=== M8B (resume with optimizer) $(date)"
[ -n "$CK" ] && { MASTER_PORT=29574 $RUN python train_omni.py megatron "${MEG[@]}" --mcore_adapter $CK --finetune false \
    --no_load_optim false --no_load_rng false --output_dir $O/M8B > $O/M8B.log 2>&1; echo "M8B exit=$?"; }

echo "end $(date)"
{
  echo "## HF DDP (swift sft)";  $RUN python tests/summarize_smoke.py $O H8A H8B H8A
  echo; echo "partitions H8A:";  $RUN python tests/check_partitions.py $O/ids_H8A
  echo; echo "## Megatron EP=8";  $RUN python tests/summarize_smoke.py $O M8A M8B M8A
  echo; echo "partitions M8A:";  $RUN python tests/check_partitions.py $O/ids_M8A
} 2>&1 | grep -v "pynvml\|FutureWarning" > $O/summary.txt
cat $O/summary.txt
du -sh $O
