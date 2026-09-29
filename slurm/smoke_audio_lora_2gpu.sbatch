#!/bin/bash
#SBATCH --job-name=omni_alora
#SBATCH --partition=h200p
#SBATCH --account=prj0000000234
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/prj0000000234/sailorhb/toolkits/qwen3_omni_speechllm/outputs/smoke_audio_lora/slurm-%j.out
#
# Smoke test for LoRA inside the audio encoder (train_omni.py --audio_lora_layers / --audio_lora_modules), 2 GPUs.
#   HA  sft (HF DDP): thinker attention + proj1/proj2 (as train_8gpu_both) + audio layers top4 x attn_mlp, 20 steps
#   MA  megatron EP=2: linear_qkv linear_proj + audio layers top4 x attn_mlp, 20 iterations
# Pass: both train, adapters hold LoRA only in audio layers 28..31 with all lora_B non-zero, and MA's
#       adapter_config.json lists exact module names (fix_peft_target_modules).
set -u
P=/scratch/prj0000000234/sailorhb/toolkits/qwen3_omni_speechllm
O=$P/outputs/smoke_audio_lora
RUN="bash /scratch/prj0000000234/sailorhb/container/run_container.sh"
MODEL=$(ls -d /scratch/prj0000000234/sailorhb/hf_models/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/* | head -1)
export PYTHONPATH=$P:/scratch/prj0000000234/sailorhb/toolkits/pydeps
export OMNI_MDS_CACHE=/tmp/$USER/omni_mds_cache
export NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1
mkdir -p $O; cd $P
echo "node $(hostname)  start $(date)"; nvidia-smi -L

AUDIO=(--audio_lora_layers top4 --audio_lora_modules attn_mlp)
HF=(--mix mixes/st_v0.yaml --val_choose 8 "${AUDIO[@]}"
    --model "$MODEL" --model_type qwen3_omni_moe --experts_impl grouped_mm
    --tuner_type lora --target_modules $(cat lora_targets_thinker_attn_audio_proj.txt) --lora_rank 16 --lora_alpha 32
    --torch_dtype bfloat16 --attn_impl flash_attn --padding_free true
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4
    --learning_rate 1e-4 --warmup_ratio 0.05 --gradient_checkpointing true --max_length 2048
    --logging_steps 1 --dataloader_num_workers 4 --report_to none --data_seed 42 --seed 42
    --save_total_limit 1 --max_steps 20 --save_steps 20 --eval_steps 20)
MEG=(--mix mixes/st_v0.yaml --val_choose 8 "${AUDIO[@]}"
    --model "$MODEL" --model_type qwen3_omni_moe
    --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules linear_qkv linear_proj
    --freeze_llm false --freeze_vit true --freeze_aligner true
    --expert_model_parallel_size 2 --moe_grouped_gemm true --moe_permute_fusion true --moe_aux_loss_coeff 1e-3
    --micro_batch_size 1 --global_batch_size 8 --lr 1e-4 --min_lr 1e-5 --lr_warmup_fraction 0.05
    --recompute_granularity selective --cross_entropy_loss_fusion true --max_length 2048 --attention_backend flash
    --merge_lora false --save_total_limit 2 --no_save_optim true --logging_steps 1 --dataloader_num_workers 4
    --data_seed 42 --seed 42 --train_iters 20 --save_steps 20 --eval_steps 20 --finetune true)

echo "=== HA $(date)"
MASTER_PORT=29591 $RUN python train_omni.py sft "${HF[@]}" --output_dir $O/HA > $O/HA.log 2>&1; echo "HA exit=$?"
echo "=== MA $(date)"
MASTER_PORT=29592 $RUN python train_omni.py megatron "${MEG[@]}" --output_dir $O/MA > $O/MA.log 2>&1; echo "MA exit=$?"
echo "end $(date)"
{
  echo "## HF DDP";  $RUN python tests/summarize_smoke.py $O HA HA HA
  echo; echo "## Megatron EP=2";  $RUN python tests/summarize_smoke.py $O MA MA MA
  echo; grep -h "audio encoder LoRA\|trainable params\|target_modules set" $O/HA.log $O/MA.log | sort -u | head
} 2>&1 | grep -v "pynvml\|FutureWarning" > $O/summary.txt
cat $O/summary.txt
