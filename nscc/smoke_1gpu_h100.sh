#!/bin/bash
# 1x H100 80GB smoke test on NSCC, run inside an interactive PBS GPU job (no scheduler header needed).
#   A: 10 steps, save + eval (6 CoVoST2 sets x 8) at 10   B: resume-from-A ckpt-5 to step 10, compare losses
#   bash nscc/smoke_1gpu_h100.sh        logs: $OUT/*.log   summary: $OUT/summary.txt
set -u
source "$(dirname "$0")/env.sh"
OUT=${OUT:-$C/smoke_outputs/nscc_h100}; mkdir -p $OUT
export NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0
cd $P
nvidia-smi -L; df -h /tmp | tail -1
COMMON=(--mix $MIX --val_choose 8 --model "$MODEL" --model_type qwen3_omni_moe
    --tuner_type lora --target_modules $(cat lora_targets_thinker_attn_audio_proj.txt) --lora_rank 16 --lora_alpha 32
    --torch_dtype bfloat16 --attn_impl flash_attn --padding_free true
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4
    --learning_rate 1e-4 --warmup_ratio 0.05 --gradient_checkpointing true --max_length 2048
    --logging_steps 1 --dataloader_num_workers 4 --report_to none --data_seed 42 --seed 42
    --save_only_model false --save_total_limit 3)
echo "=== A $(date)"
$RUN python train_omni.py sft "${COMMON[@]}" --max_steps 10 --save_steps 5 --eval_steps 10 --output_dir $OUT/A > $OUT/A.log 2>&1; echo "A exit=$?"
CK=$(ls -d $OUT/A/v*/checkpoint-5 2>/dev/null | head -1); echo "ckpt: $CK"
echo "=== B resume $(date)"
[ -n "$CK" ] && { $RUN python train_omni.py sft "${COMMON[@]}" --max_steps 10 --save_steps 5 --eval_steps 10 --resume_from_checkpoint $CK --output_dir $OUT/B > $OUT/B.log 2>&1; echo "B exit=$?"; }
$RUN python tests/summarize_smoke.py $OUT > $OUT/summary.txt 2>&1; cat $OUT/summary.txt
echo "end $(date)"
