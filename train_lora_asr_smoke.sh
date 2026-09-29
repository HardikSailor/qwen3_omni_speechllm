#!/bin/bash
# ms-swift LoRA smoke test: Qwen3-Omni-30B-A3B-Instruct ASR on LibriSpeech (1x H200, venv toolkits/venvs/swift_omni).
#   bash train_lora_asr_smoke.sh [extra swift sft args, e.g. --max_steps 200]
# LoRA on thinker attention (q/k/v/o) + audio adapter (proj1/proj2); router, experts, encoder, vision frozen.
# Targets are an explicit list of full module paths: with PEFT 0.20 a --target_regex gets rewritten to bare
# names (q_proj, ...) and silently also wraps the audio encoder's self_attn.q/k/v_proj.
cd "$(dirname "$0")"
PY=/scratch/prj0000000234/sailorhb/toolkits/venvs/swift_omni/bin  # venv over nemo_env (--system-site-packages) + ms-swift
MODEL=$(ls -d /scratch/prj0000000234/sailorhb/hf_models/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/* | head -1)
OUT=outputs/swift_lora_asr_smoke

export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/prj0000000234/sailorhb/hf_models
export HF_HUB_OFFLINE=1
export ENABLE_AUDIO_OUTPUT=0  # drop talker + code2wav (text-only training)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

$PY/swift sft \
    --model "$MODEL" \
    --model_type qwen3_omni_moe \
    --dataset data/asr_train.jsonl \
    --val_dataset data/asr_val.jsonl \
    --tuner_type lora \
    --target_modules $(cat lora_targets_thinker_attn_audio_proj.txt) \
    --lora_rank 16 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --max_steps 60 \
    --gradient_checkpointing true \
    --max_length 2048 \
    --eval_steps 30 \
    --save_steps 30 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 2 \
    --report_to tensorboard \
    --output_dir "$OUT" \
    "$@"
