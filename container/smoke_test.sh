#!/bin/bash
# Smoke tests for swift_megatron_cu128.sif on 1 GPU (run on a GPU node):
#   bash smoke_test.sh env        # versions, GPU, Transformer Engine, imports
#   bash smoke_test.sh infer      # swift infer (transformers backend) on 3 LibriSpeech dev-other clips
#   bash smoke_test.sh vllm       # same 3 clips through the vLLM backend
#   bash smoke_test.sh sft        # 5-step swift sft LoRA (same setup as train_lora_asr_smoke.sh)
#   bash smoke_test.sh megatron   # 5-step Megatron-SWIFT LoRA SFT on the ASR smoke data (EP=1)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_HOME=${CONTAINER_HOME:-/scratch/users/astar/ares/sailorhb/container}
RUN="bash $HERE/run_container.sh"
PRJ=$(dirname "$HERE")   # the repo; the ASR smoke data ($PRJ/data/asr_*.jsonl, LibriSpeech) is not on NSCC yet
MODEL=$(ls -d $CONTAINER_HOME/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/* | head -1)
OUT=$CONTAINER_HOME/smoke_outputs
mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

case "$1" in
env)
    $RUN bash -c '
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
        python - <<EOF
import torch, transformers, transformer_engine, megatron.core, mcore_bridge, swift, peft, datasets, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu ok", torch.cuda.is_available())
print("transformers", transformers.__version__, "| TE", transformer_engine.__version__,
      "| megatron-core", megatron.core.__version__, "| ms-swift", swift.__version__)
print("peft", peft.__version__, "| datasets", datasets.__version__, "| vllm", vllm.__version__)
import transformer_engine.pytorch as te
print("TE linear", te.Linear(256, 256).cuda()(torch.randn(4, 256, device="cuda")).shape)
import site, sys; print("user site enabled:", site.ENABLE_USER_SITE)
EOF'
    ;;
infer)
    head -3 $PRJ/data/asr_val.jsonl > $OUT/infer_in.jsonl
    $RUN swift infer --model "$MODEL" --model_type qwen3_omni_moe --infer_backend transformers \
        --val_dataset $OUT/infer_in.jsonl --max_new_tokens 128 --temperature 0 \
        --attn_impl flash_attn --result_path $OUT/infer_out.jsonl
    ;;
vllm)
    head -3 $PRJ/data/asr_val.jsonl > $OUT/infer_in.jsonl
    $RUN swift infer --model "$MODEL" --model_type qwen3_omni_moe --infer_backend vllm \
        --val_dataset $OUT/infer_in.jsonl --max_new_tokens 128 --temperature 0 \
        --vllm_max_model_len 8192 --vllm_gpu_memory_utilization 0.85 --vllm_limit_mm_per_prompt '{"audio": 1}' \
        --result_path $OUT/infer_out_vllm.jsonl
    ;;
sft)
    $RUN swift sft --model "$MODEL" --model_type qwen3_omni_moe \
        --dataset $PRJ/data/asr_train.jsonl'#40' --split_dataset_ratio 0 \
        --tuner_type lora --target_modules $(cat $PRJ/lora_targets_thinker_attn_audio_proj.txt) \
        --lora_rank 16 --lora_alpha 32 --torch_dtype bfloat16 --attn_impl flash_attn \
        --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --learning_rate 1e-4 \
        --max_steps 5 --gradient_checkpointing true --max_length 2048 \
        --save_steps 5 --logging_steps 1 --dataset_num_proc 2 --dataloader_num_workers 2 \
        --report_to none --output_dir $OUT/swift_sft_lora
    ;;
megatron)
    NPROC_PER_NODE=${NPROC_PER_NODE:-1} $RUN megatron sft --model "$MODEL" --model_type qwen3_omni_moe \
        --dataset $PRJ/data/asr_train.jsonl'#40' \
        --tuner_type lora --lora_rank 8 --lora_alpha 32 --target_modules linear_qkv linear_proj \
        --freeze_llm false --freeze_vit true --freeze_aligner true \
        --expert_model_parallel_size 1 --moe_grouped_gemm true --moe_permute_fusion true \
        --moe_aux_loss_coeff 1e-3 --packing true \
        --micro_batch_size 1 --global_batch_size 4 --train_iters 5 \
        --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
        --finetune true --cross_entropy_loss_fusion true --lr 1e-4 --min_lr 1e-5 \
        --max_length 2048 --attention_backend flash --split_dataset_ratio 0 \
        --save_safetensors true --merge_lora false --no_save_optim true --no_save_rng true \
        --dataloader_num_workers 2 --dataset_num_proc 2 --logging_steps 1 \
        --output_dir $OUT/megatron_lora
    ;;
*) echo "usage: $0 env|infer|vllm|sft|megatron"; exit 1 ;;
esac
