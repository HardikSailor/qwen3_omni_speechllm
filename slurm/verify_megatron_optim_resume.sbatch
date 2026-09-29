#!/bin/bash
#SBATCH --job-name=omni_optimfix
#SBATCH --partition=h200p
#SBATCH --account=prj0000000234
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/prj0000000234/sailorhb/toolkits/qwen3_omni_speechllm/outputs/debug_optim/slurm-verify-%j.out
#
# Verify the root cause of the Megatron-SWIFT LoRA optimizer-resume failures (PIPELINE_PLAN.md §10.4):
# load_mcore_checkpoint builds its load template with the *resuming* run's args, and _generate_state_dict only adds
# the optimizer entries when `not args.no_save_optim`. All failing resumes passed --no_save_optim true.
#   R1  resume smoke_megatron/MA/checkpoint-10 (default dp_reshardable) with optimizer and --no_save_optim false
#       -> losses 11..20 should track MA closely (like the HF resume)
#   R2  resume smoke_megatron/MA2/checkpoint-5 (fully_reshardable) the same way -> does the NaN go away?
set -u
P=/scratch/prj0000000234/sailorhb/toolkits/qwen3_omni_speechllm
O=$P/outputs/debug_optim
S=$P/outputs/smoke_megatron
RUN="bash /scratch/prj0000000234/sailorhb/container/run_container.sh"
MODEL=$(ls -d /scratch/prj0000000234/sailorhb/hf_models/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/* | head -1)
export PYTHONPATH=$P:/scratch/prj0000000234/sailorhb/toolkits/pydeps
export OMNI_MDS_CACHE=/tmp/$USER/omni_mds_cache
export NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29564
cd $P
echo "node $(hostname)  start $(date)"
COMMON=(--mix mixes/st_v0.yaml --val_choose 32
    --model "$MODEL" --model_type qwen3_omni_moe
    --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules linear_qkv linear_proj
    --freeze_llm false --freeze_vit true --freeze_aligner true
    --expert_model_parallel_size 1 --moe_grouped_gemm true --moe_permute_fusion true --moe_aux_loss_coeff 1e-3
    --micro_batch_size 1 --global_batch_size 4 --lr 1e-4 --min_lr 1e-5 --lr_warmup_fraction 0.05
    --recompute_granularity selective --cross_entropy_loss_fusion true --max_length 2048 --attention_backend flash
    --merge_lora false --save_total_limit 2 --logging_steps 1 --dataloader_num_workers 4 --data_seed 42 --seed 42)

CK=$(ls -d $S/MA/v*/checkpoint-10 | head -1)
echo "=== R1 resume $CK with optimizer, --no_save_optim false $(date)"
$RUN python train_omni.py megatron "${COMMON[@]}" --train_iters 20 --save_steps 10 --eval_steps 10 \
    --mcore_adapter $CK --finetune false --no_load_optim false --no_load_rng false --no_save_optim false \
    --output_dir $O/R1 > $O/R1.log 2>&1; echo "R1 exit=$?"; grep -m3 "KeyError\|NaN\|Error:" $O/R1.log

CK=$(ls -d $S/MA2/v*/checkpoint-5 | head -1)
echo "=== R2 resume $CK (fully_reshardable) with optimizer, --no_save_optim false $(date)"
$RUN python train_omni.py megatron "${COMMON[@]}" --train_iters 10 --save_steps 100000 --eval_iters 0 \
    --mcore_adapter $CK --finetune false --no_load_optim false --no_load_rng false --no_save_optim false \
    --dist_ckpt_optim_fully_reshardable true --output_dir $O/R2 > $O/R2.log 2>&1; echo "R2 exit=$?"
grep -m3 "KeyError\|NaN\|Error:" $O/R2.log
echo "end $(date)"
{ $RUN python tests/summarize_smoke.py $S MA ../debug_optim/R1 MA; echo; $RUN python tests/summarize_smoke.py $S MA2 ../debug_optim/R2 MA2; } 2>&1 | grep -v pynvml
