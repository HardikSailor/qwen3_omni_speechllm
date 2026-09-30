#!/bin/bash
# Audit: does train_omni.py use what a --config YAML says? 1 GPU, both trainers, ~20 min. Run in an interactive job:
#   bash nscc/audit_config_1gpu.sh            -> outputs/config_audit/{summary.txt, *.log}
# The configs are copies of configs/lora_{ddp,megatron}.yaml with NON-default values (so a pass cannot come from ms-swift
# defaults happening to match). A few keys are overridden on the command line (1 GPU, 4 steps); the checker is told so.
set -u
source "$(dirname "$0")/env.sh"
O=$P/outputs/config_audit; rm -rf $O; mkdir -p $O; cd $P
export CUDA_VISIBLE_DEVICES=0
set_keys () {   # set_keys <in.yaml> <out.yaml> key=value ...
  local in=$1 out=$2; shift 2; cp $in $out
  for kv in "$@"; do k=${kv%%=*}; v=${kv#*=}
    grep -q "^$k:" $out || { echo "no key $k in $in" >&2; exit 1; }
    sed -i -E "s#^($k):[^#]*#\1: $v   #" $out
  done
}
set_keys configs/lora_ddp.yaml $O/ddp.yaml learning_rate=2.0e-4 weight_decay=0.05 lora_rank=8 lora_alpha=16 \
  lora_dropout=0.1 adam_beta2=0.98 warmup_ratio=0.25 max_grad_norm=0.5 lr_scheduler_type=linear \
  audio_lora_layers=top2 audio_lora_modules=attn gradient_accumulation_steps=2
set_keys configs/lora_megatron.yaml $O/meg.yaml lr=2.0e-4 min_lr=2.0e-5 weight_decay=0.05 lora_rank=8 lora_alpha=16 \
  lora_dropout=0.1 adam_beta2=0.98 clip_grad=0.5 lr_warmup_fraction=0.25 lr_decay_style=linear \
  audio_lora_layers=top2 audio_lora_modules=attn
diff configs/lora_ddp.yaml $O/ddp.yaml > $O/ddp.yaml.diff; diff configs/lora_megatron.yaml $O/meg.yaml > $O/meg.yaml.diff

echo "=== HF $(date)"
NPROC_PER_NODE=1 MASTER_PORT=29581 $RUN python train_omni.py sft --config $O/ddp.yaml \
  --max_steps 4 --save_steps 4 --eval_steps 4 --val_choose 4 --save_total_limit 1 --output_dir $O/HF > $O/HF.log 2>&1
echo "HF exit=$?"
echo "=== MEG $(date)"
NPROC_PER_NODE=1 MASTER_PORT=29582 $RUN python train_omni.py megatron --config $O/meg.yaml --finetune true \
  --expert_model_parallel_size 1 --global_batch_size 4 --train_iters 4 --save_steps 4 --eval_steps 4 --val_choose 4 \
  --output_dir $O/MEG > $O/MEG.log 2>&1
echo "MEG exit=$?"
{
  echo "## HF (swift sft)"
  $RUN python tests/check_effective_args.py $O/ddp.yaml $(ls -d $O/HF/v*) --expect max_steps save_steps eval_steps val_choose save_total_limit
  echo; echo "## Megatron"
  $RUN python tests/check_effective_args.py $O/meg.yaml $(ls -d $O/MEG/v*) --expect expert_model_parallel_size global_batch_size train_iters save_steps eval_steps val_choose
  echo; echo "## LoRA adapters saved (HF, Megatron)"
  for a in $O/HF/v*/checkpoint-4 $O/MEG/v*/checkpoint-4; do
    python3 -c "import json,sys; c=json.load(open(sys.argv[1]+'/adapter_config.json')); print(sys.argv[1].split('/')[-3], 'r', c['r'], 'alpha', c['lora_alpha'], 'dropout', c['lora_dropout'], 'targets', len(c['target_modules']))" $a
  done
} 2>&1 | grep -v "pynvml\|FutureWarning" > $O/summary.txt
cat $O/summary.txt
echo AUDITDONE
