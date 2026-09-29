#!/bin/bash
#SBATCH --job-name=omni_enroot
#SBATCH --partition=h200p
#SBATCH --account=prj0000000234
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/prj0000000234/sailorhb/toolkits/qwen3_omni_speechllm/outputs/enroot_test/slurm-%j.out
#
# enroot path check (for clusters without Apptainer, e.g. ASPIRE2A+):
#   1. container/sif_to_enroot.sh: .sif -> .sqsh (skipped if the .sqsh exists)
#   2. environment inside enroot: GPUs, torch/TE/swift/megatron versions, /tmp on local disk (not tmpfs), user site off
#   3. slurm/smoke_audio_lora_2gpu.sbatch with CONTAINER_RUNTIME=enroot (both trainers) -> outputs/enroot_test/
set -u
P=/scratch/prj0000000234/sailorhb/toolkits/qwen3_omni_speechllm
C=/scratch/prj0000000234/sailorhb/container
O=$P/outputs/enroot_test
mkdir -p $O; cd $P
echo "node $(hostname)  start $(date)"; enroot version

echo "=== 1. convert $(date)"
[ -f $C/swift_megatron_cu128.sqsh ] || bash $C/sif_to_enroot.sh || exit 1
ls -la $C/swift_megatron_cu128.sqsh

echo "=== 2. environment $(date)"
export CONTAINER_RUNTIME=enroot CUDA_VISIBLE_DEVICES=0,1
bash $C/run_container.sh bash -c 'nvidia-smi -L; df -h /tmp | tail -1; findmnt -no FSTYPE,SOURCE /tmp; echo "HOME=$HOME PYTHONNOUSERSITE=$PYTHONNOUSERSITE HF_HOME=$HF_HOME NLTK_DATA=$NLTK_DATA"; pwd'
bash $C/run_container.sh python -c "
import torch, transformer_engine.pytorch as te, swift, megatron.core, vllm, site
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())
x = te.Linear(64, 64).cuda(); print('TE linear ok', tuple(x(torch.randn(8, 64, device='cuda')).shape))
print('swift', swift.__version__, 'mcore', megatron.core.__version__, 'vllm', vllm.__version__, 'user site', site.ENABLE_USER_SITE)
"
echo "=== 3. audio-LoRA smoke through enroot $(date)"
OUT_DIR=$O bash slurm/smoke_audio_lora_2gpu.sbatch
echo "end $(date)"
