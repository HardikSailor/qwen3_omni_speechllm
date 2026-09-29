# Qwen3-Omni adaptation: work log

## 2026-09-24

### Done today

**Planning documents**
- `FINETUNING_GUIDE.md`: the fine-tuning plan for our SEA / Singapore multi-task data (ASR, ST, SER, SQA, AEC, with code-switching and cultural grounding). It covers:
  - Stage 0 (language/acoustic adaptation for languages the encoder doesn't support: Tamil, Thai, Filipino, Burmese, Khmer, Lao, Hokkien), then joint multi-task training, then the speech / acoustic-event tracks;
  - architecture ideas, RL rewards, evaluation and a roadmap;
  - §12: the choice of training stack.
- Shared doc **Megatron-SWIFT vs swift sft for Qwen3-Omni**: https://claude.ai/code/artifact/c06ff669-814a-4abf-8feb-de36b7f4ea8d

**Facts verified from the local model files and code**
- Thinker: 48 layers, 128 experts with 8 active, **no shared expert**; 30.5B parameters in total, ~3.3B active per token.
- Audio encoder (AuT): 32 layers, 0.65B parameters, **13 audio tokens/s**.
- No Whisper-style 30 s padding: the encoder takes variable-length input in 1 s chunks, and attention covers 8 s windows.
- Tokenizer cost relative to English for the same sentence: Thai ≈2.5×, Tamil ≈5×, Burmese ≈6×.

**Megatron assessment**
- The plain `toolkits/Megatron-LM` repo (core 0.20) cannot load Qwen3-Omni.
- Megatron-SWIFT (ms-swift + mcore-bridge) supports it.
- Cluster: 13 nodes × 8 H200, 8 × 400 Gb/s NDR InfiniBand per node.

**Container** (`/scratch/prj0000000234/sailorhb/container/`, see its `README.md`)
- `swift_megatron_cu128.sif`, 12.6 GB. Contents:
  - ms-swift 4.6.0.dev0 (our checkout at commit 8ec0455);
  - megatron-core 0.16.1 (0.17+ needs Python 3.12; the image has 3.11), mcore-bridge 1.6.4, Transformer Engine 2.13;
  - torch 2.10 with CUDA 12.8, which runs on our driver 565; vLLM 0.17.1.
- Scripts:
  - `run_container.sh` is the launcher. It isolates the container from home-directory packages and handles Slurm multi-node settings.
  - `build_container.sh` + `swift_megatron_cu128.def` rebuild the image.
  - `smoke_test.sh` re-runs the tests below.
- Smoke tests on 1 GPU, all passing:

  | Test | Result |
  |---|---|
  | `env` | GPU and Transformer Engine OK; user site-packages disabled |
  | `infer` (transformers backend) | 3/3 LibriSpeech clips exact, 14.2 s |
  | `vllm` | 3/3 clips exact, 2.4 s |
  | `sft` (swift sft LoRA) | loss 0.65 → 0.43, 60 GiB, ~1.7 s/step |
  | `megatron` (Megatron-SWIFT LoRA) | loss 0.56 → 0.38, 63.5 GiB, ~4.2 s/step; PEFT adapter exported, no audio-encoder LoRA |

**Gotchas found**
- `megatron sft` needs `NPROC_PER_NODE` set, even to 1.
- The Megatron logging argument is `--logging_steps`, not `--log_interval`.
- Megatron-exported adapters list bare `target_modules` names. When PEFT loads them, it adds unused LoRA layers to the audio encoder. Harmless, since they stay at zero.
- `/scratch` is **99% full** (3.3 TB free, shared by everyone). Build on node-local `/tmp` (28 TB); don't save optimizer state unless you need to resume.

### Running overnight

- **Slurm job 147727** (`omni_bench8`, partition h200p, 1 node × 8 H200, 4 h limit). It was pending on resources at submission.
- It runs `container/bench_8gpu/bench_8gpu.sbatch`: 30 steps × 32 samples, LoRA r16 on attention only, same 1000 ASR samples:
  - A: `swift sft` DDP with `--experts_impl grouped_mm`;
  - B: `swift sft` DDP with default experts;
  - C: `megatron sft` with EP=8.
- Results land in `container/bench_8gpu/summary.md` (steady-state s/step, samples/s, memory, loss). Per-run logs are in `bench_8gpu/logs/`.
 
## Tomorrow (2026-09-25)

1. **Check the benchmark.**
   - `squeue -u $USER`, then `cat /scratch/prj0000000234/sailorhb/container/bench_8gpu/summary.md`.
   - If a run failed, see `bench_8gpu/logs/{A_swift_grouped,B_swift_default,C_megatron_ep8}.log` and `logs/slurm-147727.out`.
   - To re-run the summary alone: `python3 bench_8gpu/summarize.py`.
2. **Decide the trainer.**
   - If Megatron-SWIFT is not clearly faster at 8 GPUs for attention-only LoRA, use `swift sft` (grouped_mm) for LoRA work. Keep Megatron for expert LoRA, full fine-tuning and multi-node.
   - Optional follow-up benchmarks:
     - Megatron with `--packing true`: compare samples/s, not s/step;
     - LoRA including experts (`all-linear`), where Megatron should gain most;
     - a 2-node run to test InfiniBand and NCCL.
3. **Record the decision** in `FINETUNING_GUIDE.md` §12 and in the shared doc's "Next steps".
4. **Start on data** (roadmap steps 1–2 in the guide):
   - inventory the SEA/SG datasets we actually have (languages, hours, tasks, licences);
   - agree normalisation and code-switch conventions (guide §4.3);
   - run a zero-shot baseline eval per task × language.
5. **Verify the dataset list** in guide §4.1 (NSC, SEAME, MERLIon, FLEURS, GigaSpeech 2, CoVoST 2) against what we can access. It was written from memory.
6. Housekeeping:
   - interactive job 147327 ends at its 12 h limit; ~50 GB of build files in `/tmp/sailorhb_apptainer` on DDS9B74 are node-local;
   - `container/smoke_outputs/` (~30 MB) can be deleted.

## 2026-09-29

### Done today

**4-GPU check** (8-GPU job 149729 was still pending: no node had 8 free GPUs)
- `slurm/train_4gpu_both.sbatch`, **job 150431**, 24 min, both paths pass. The global batch stays 32 (HF grad-accum 8; Megatron EP=4).
- Results:
  - HF: loss 1.175 → 0.831, 61.8 GiB, 4.8 s/step, resume within 0.004;
  - Megatron EP=4: loss 1.172 → 0.837, **23.5 GiB**, 5.6 s/step, resume within 0.003 (with optimizer);
  - partitions disjoint (4 × 488);
  - Megatron `eval_iters` = 6 = 192/32, which confirms D16 on more than one GPU.
- The eval-loss gap (HF 1.31 vs. Megatron 0.80) matches the 1-GPU runs; it comes from how each trainer averages eval loss.
- Details: `PIPELINE_PLAN.md` §10.6.

**Transfer kit for the H100 server** (`transfer/`, see `TRANSFER_TO_H100.md`)
- **What to copy (~275 GB):**
  - code (~140 MB, with the ms-swift source);
  - container (12.6 GB `.sif`, `run_container.sh`, `nltk_data`);
  - model (66 GB);
  - `st_v0` MDS data (~195 GB).
- **Not copied:** `venvs/swift_omni` (tied to `nemo_env` in `$HOME`) and `outputs/`.
- **Scripts:**
  - `copy_to_h100.sh` (rsync, dry run by default);
  - `relocate_paths.sh` (rewrites `/scratch/...` paths and bind mounts; tested on a local copy).
- **Requirement files:** pinned top-level packages, the pydeps extras, and the image's full `pip freeze` (406 packages).

**LoRA inside the audio encoder**
- **Options:** `train_omni.py --audio_lora_layers none|all|top<K>|bottom<K>|a-b|i,j,k` and `--audio_lora_modules attn|mlp|attn_mlp`, for both trainers.
- **Test, job 150442** (2 GPUs, top4 × attn_mlp): both trainers put LoRA only in layers 28–31, and 24/24 encoder `lora_B` weights trained.
  - Memory: HF 61.8 GiB, Megatron EP=2 34.4 GiB.
  - The first attempt, job 150441, stopped on Megatron's `--save_total_limit >= 2` rule (a test-script mistake).
- **Adapter configs:** saved `adapter_config.json` files now list exact module names. Both trainers otherwise write names that also match untrained modules (B17).
- Details: `PIPELINE_PLAN.md` §10.7; decisions D19–D21 in `DECISION_LOG.md`.

### Next
1. Copy to the H100 server. Run `relocate_paths.sh`, then `container/smoke_test.sh env` and `slurm/smoke_audio_lora_2gpu.sbatch` there.
2. Job 149729 (8 GPUs) for EP=8 throughput, if it's still wanted.
3. Test loading an audio-encoder LoRA adapter for inference (PEFT; check vLLM).

**Deployment / partner on-prem (later on 2026-09-29)**
- **Finding:** vLLM 0.17.1 can't serve LoRA adapters for Qwen3-Omni (`supports_lora` = False for the thinker class; True for Qwen2.5-Omni). This affects every adapter we train.
- **Wrote `DEPLOYMENT.md`:**
  - ship a merged model on the pinned official vLLM image; the partner needs a GPU host, driver, Docker, the image and the model folder;
  - hardware sizing: bf16 ~63 GB (1× H200 or 2× 80 GB); FP8 and 4-bit untested;
  - the hand-off package: prompts, per-task clients, a check set, a model card;
  - training rules: keep the architecture stock;
  - licences: review the data mix before a commercial hand-off.
- **Corrected docs** that assumed "vLLM loads adapters": `PIPELINE_PLAN.md` (§10.9 added), `FINETUNING_GUIDE.md`, `DECISION_LOG.md` (D22, B20).
- **Next:** test `swift export --merge_lora` on HF and Megatron adapters (with audio-encoder LoRA), then serve the result in vLLM and compare with PEFT outputs.

**Git**
- Project under git: `README.md`, `.gitignore`, first push to `git@github.com:HardikSailor/qwen3_omni_speechllm.git` (done by the user).
- ms-swift added as a submodule, `third_party/ms-swift` @ `8ec0455`, the same commit as in the container.
  - Cloned from the local checkout, so nothing was downloaded.
  - Upgrade rule: branch → rebuild the container → smoke tests → merge (D23).
- **To do:** point `container/build_container.sh` at the submodule.
