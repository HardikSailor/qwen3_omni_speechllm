#!/bin/bash
#SBATCH --job-name=omni_smokeM2
#SBATCH --partition=h200p
#SBATCH --account=prj0000000234
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/prj0000000234/sailorhb/toolkits/qwen3_omni_speechllm/outputs/smoke_megatron/slurm-resume-%j.out
#
# Follow-up to smoke_megatron_1gpu.sbatch (job 149697), whose optimizer resume failed inside Megatron-SWIFT:
# `KeyError: 'optimizer'` in distrib_optimizer.load_state_dict (mcore 0.16.1).
#   MB3  resume MA/checkpoint-10 with --no_load_optim true: tests OUR data resume. Loss at iteration 11 is computed
#        before any optimizer update, so it must equal MA's iteration-11 loss if the data position is right.
#   MA2  10 iterations, save with --dist_ckpt_optim_fully_reshardable true (other optimizer checkpoint format)
#   MB2  resume MA2/checkpoint-5 with optimizer (--no_load_optim false) to 10: does full resume work this way?
set -u
P=/scratch/prj0000000234/sailorhb/toolkits/qwen3_omni_speechllm
O=$P/outputs/smoke_megatron
RUN="bash /scratch/prj0000000234/sailorhb/container/run_container.sh"
MODEL=$(ls -d /scratch/prj0000000234/sailorhb/hf_models/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/* | head -1)
export PYTHONPATH=$P:/scratch/prj0000000234/sailorhb/toolkits/pydeps
export OMNI_MDS_CACHE=/tmp/$USER/omni_mds_cache
export NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29562
cd $P
echo "node $(hostname)  start $(date)"

COMMON=(--mix mixes/st_v0.yaml --val_choose 32
    --model "$MODEL" --model_type qwen3_omni_moe
    --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules linear_qkv linear_proj
    --freeze_llm false --freeze_vit true --freeze_aligner true
    --expert_model_parallel_size 1 --moe_grouped_gemm true --moe_permute_fusion true --moe_aux_loss_coeff 1e-3
    --micro_batch_size 1 --global_batch_size 4 --lr 1e-4 --min_lr 1e-5 --lr_warmup_fraction 0.05
    --recompute_granularity selective --cross_entropy_loss_fusion true
    --max_length 2048 --attention_backend flash
    --merge_lora false --logging_steps 1 --dataloader_num_workers 4 --data_seed 42 --seed 42)

CK10=$(ls -d $O/MA/v*/checkpoint-10 | head -1)
echo "=== MB3 (resume data+weights, no optimizer) from $CK10 $(date)"
$RUN python train_omni.py megatron "${COMMON[@]}" --train_iters 20 --save_steps 100000 --eval_steps 10 \
    --mcore_adapter $CK10 --finetune false --no_load_optim true --no_load_rng false \
    --no_save_optim true --output_dir $O/MB3 > $O/MB3.log 2>&1; echo "MB3 exit=$?"

echo "=== MA2 (fully reshardable optimizer checkpoint) $(date)"
$RUN python train_omni.py megatron "${COMMON[@]}" --train_iters 10 --save_steps 5 --eval_iters 0 \
    --finetune true --no_save_optim false --no_save_rng false --dist_ckpt_optim_fully_reshardable true \
    --output_dir $O/MA2 > $O/MA2.log 2>&1; echo "MA2 exit=$?"
CK5=$(ls -d $O/MA2/v*/checkpoint-5 2>/dev/null | head -1)
echo "=== MB2 (resume with optimizer) from $CK5 $(date)"
if [ -n "$CK5" ]; then
  $RUN python train_omni.py megatron "${COMMON[@]}" --train_iters 10 --save_steps 100000 --eval_iters 0 \
      --mcore_adapter $CK5 --finetune false --no_load_optim false --no_load_rng false \
      --dist_ckpt_optim_fully_reshardable true --no_save_optim true --output_dir $O/MB2 > $O/MB2.log 2>&1
  echo "MB2 exit=$?"; grep -m3 "KeyError\|Error" $O/MB2.log
fi
echo "end $(date)"
{ $RUN python tests/summarize_smoke.py $O MA MB3 MC; echo; $RUN python tests/summarize_smoke.py $O MA2 MB2 MC; } > $O/summary_resume.txt 2>&1
grep -v pynvml $O/summary_resume.txt
