# Qwen3-Omni adaptation: work log

> **Cluster note (2026-09-30):** this document was written on the previous cluster (Orion: Slurm, H200, Apptainer). The project now runs on NSCC (PBS, H100, enroot). Map the old paths with the *Paths* section of `README.md`; the container is described in `container/README.md`. Job numbers (e.g. 150431) are Orion Slurm jobs.

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

**8-GPU job 149729 finished** (09-29 11:47–12:07, node 374Y574). Both paths pass (`PIPELINE_PLAN.md` §10.11):
- resume matches: max |Δloss| 0.004 on HF and on Megatron with optimizer; data position 960 at step 30;
- 8 ranks × 248 ids, disjoint; Megatron `eval_iters` 6 (global-length fix holds at DP=8);
- steady throughput 14.6 samples/s HF DDP vs. 13.9 Megatron EP=8 (the gap was 17.0 vs. 13.5 on the LibriSpeech benchmark); memory 61.8 vs. 16.7 GiB;
- 4 → 8 GPUs scales 1.92× / 2.0×, so data loading keeps up.


**enroot (the new server has no Apptainer)**
- `container/sif_to_enroot.sh`: `.sif` → `swift_megatron_cu128.sqsh` (12.6 GB, ~1 min). Same filesystem, plus `/etc/environment` and `/etc/rc`.
- `container/run_container_enroot.sh`: same interface and environment as `run_container.sh`.
  - `run_container.sh` switches to it automatically when `apptainer` is missing.
  - It binds the host `/tmp`, because enroot's default `/tmp` is in RAM.
- **Test, job 150477:** passed. GPUs, TE and versions OK; `/tmp` on xfs; the audio-LoRA smoke test on both trainers matches the Apptainer results. Job 150476 failed on a converter bug (B21).
- `copy_to_h100.sh`: `IMAGE=sqsh` copies the enroot image. `relocate_paths.sh` handles both launchers.


## 2026-09-29/30: moved to NSCC (PBS, H100, enroot)

**Done**
- Checked the copied container on NSCC in an interactive GPU job: enroot image, model and MDS data are all present; `run_container.sh` picks enroot.
- Installed `mosaicml-streaming` (`pydeps`) offline: wheels from the login node, `pip install --no-index --target` inside the container.
- Ran on 1 H100: `train_omni.py sft` and `megatron` with resume (60.3 / 61.5 GiB, fits 80 GB), then `pbs/smoke_audio_lora_2gpu.pbs` on 2 GPUs through `qsub` (job 214221): pass.
- Converted the Slurm scripts to PBS (`pbs/`), added `nscc/` (env, per-node launcher, NCCL check) and brought the container files into the repo (`container/`).
- Wrote `pbs/train_32gpu_4node.pbs` (4 nodes x 8 GPUs) and used the converted `train_8gpu_both.pbs` (1 node x 8 GPUs) for the dedicated-queue tests.
- Docs: NSCC paths in the README and tests; `container/README.md` (where the image is, what is in it, what was added); banners on the older docs; D25.

**Findings**
- Resume diff on H100: HF 0.007, Megatron 0.020 (H200: 0.003 to 0.004): not investigated.
- The stale `--no_save_optim true` in the Megatron resume smoke script (B22).

**Next**
1. `qsub -q dedicated pbs/train_8gpu_both.pbs`, then `pbs/train_32gpu_4node.pbs`: read the NCCL line first (`nscc/README.md`).
2. Check the Megatron resume diff on 8 GPUs.
3. The eval script; vLLM inference on NSCC; `swift export --merge_lora` (see `DEPLOYMENT.md` §7).

**2026-09-30 (later): FSDP2 test added**
- Question was why FSDP1/2 "don't work" with ms-swift: never tested here. ms-swift supports FSDP2 in the HF trainer (`--fsdp fsdp2`, preset `swift/config/fsdp2.json`); FSDP1 has no preset. FSDP shards parameters but not experts (91% of the thinker), hence the earlier "fallback only" call (`PIPELINE_PLAN.md` §8).
- Added phase `F8A` (HF trainer, `--fsdp fsdp2`, 30 steps) as the last phase of `pbs/train_8gpu_both.pbs`; walltime 3.5 h. Untested: `train_omni.py`'s custom dataloaders and checkpoint saving under FSDP2. Commit `eb1768a`.
- LoRA-only guidance: DDP (~62 GiB/GPU) is the simplest for attention / audio-projector LoRA; Megatron EP (~17 GiB/GPU) when memory or expert LoRA forces it. Both trainers are ready and tested for attention and audio-encoder LoRA; expert LoRA has no script or target list yet.

**2026-09-30 (later): YAML recipe files (`--config`) and an audit of what the trainers really use**
- `train_omni.py --config <yaml>`: flag: value pairs for both trainers; command-line flags replace the file's values.
  Recipes: `configs/lora_ddp.yaml`, `configs/lora_megatron.yaml` (= the H8A / M8A settings of `pbs/train_8gpu_both.pbs`,
  with ms-swift defaults written out: weight decay 0.1, beta2 0.95, LoRA dropout 0.05, cosine, grad clip 1.0).
- Each run now writes `omni_effective.json`: optimizer groups, scheduler, LoRA rank/alpha/scaling/dropout per module, batch,
  parallel sizes, read from the live objects. `tests/check_effective_args.py` compares config / `args.json` / runtime per key.
- **Audit** (`nscc/audit_config_1gpu.sh`, job 214383, 1 H100): configs with non-default values (lr 2e-4, weight decay 0.05,
  rank 8, alpha 16, dropout 0.1, beta2 0.98, warmup 0.25, clip 0.5, linear schedule, audio LoRA top2) -> **both trainers PASS**,
  every key equal in all three layers; saved adapters r 8 / alpha 16 / dropout 0.1.
  - What `train_omni.py` changes on purpose: adds the audio-encoder names to `target_modules` (+8), sets the dataset
    placeholders, adds `save_total_limit 3` / `merge_lora false` only when not given.
  - Weight decay: both trainers put all LoRA tensors in one group with the configured decay (no bias/norm group, since only LoRA trains).
- Fixes found on the way: the launcher did not forward `MODEL` / `MIX` into the container (added); first audit try crashed in my new
  LoRA summary (`lora_dropout` is a `ModuleDict`), fixed and re-run.
- `tests/test_config.py` (CPU): expansion rules, and that both configs give exactly the flags of `pbs/train_8gpu_both.pbs`.
- **`pbs/` scripts switched to `--config`** (user asked): each array is now `--config configs/lora_*.yaml` plus only that
  script's differences (GPU count / EP size, batch, eval size, audio LoRA, steps). Checked against the old explicit arrays:
  identical flags, except ms-swift defaults now written out, and `experts_impl grouped_mm` added to `smoke_1gpu.pbs`
  (the only script that lacked it; speed only). `debug_megatron_optim_resume.pbs` is unchanged: it also runs plain
  `megatron sft`, which cannot read our train_omni-only keys. `tests/test_config.py` pins every script's differences.


**2026-10-01: reserved queue R212478, first 2-node run**
- Reservation `R212478` (RITM0187695): 12 DGX H100 nodes (96 GPUs), Oct 1 – Oct 31, project `13003558_R4`, `place=free`.
- New `pbs/train_16gpu_2node.pbs` (2 x 8 GPUs, same phases as the 4-node script; HF grad-accum 2 -> global 32; Megatron EP=8, DP=16).
  It has no `place=scatter:excl` line: qsub rejected it ("job and reservation have conflicting specification Resource_List.place").
- **Launcher bug fixed** (also in `train_32gpu_4node.pbs`, which had never run): `pbsdsh` re-parses its arguments in a shell on each
  node, so `$RUN` expanded empty -> `python: command not found` (job 214958, every phase exit 127). `run_phase` now writes the
  phase command to `$O/<phase>.cmd.sh` and runs that file. Also: `run_phase` prints exit=0 even when the node tasks fail; read the
  `[node_run] ... exit=` lines.
- Job 214959 (dgx001 + dgx003): **NCCL across 2 nodes OK: world 16, all_reduce 512 MB 2.4 ms, busbw 415 GB/s** (InfiniBand used).
  H16A / M16A failed at once: `mixes/st_v0.yaml: 14 dataset paths not found`. Cause: `/data/projects/13003558` is now
  `drwxrws--- root:root` (group should be 13003558), so even the login node gets "Permission denied" under it. NSCC has to restore
  the group; this also blocks the queued aiq2 jobs that use `/data/projects/13003558/hardik/...`.
- Next: once data access is back, `qsub pbs/train_16gpu_2node.pbs` (failed logs kept in `outputs/train_16gpu_failed_214958`).

**2026-10-01 (later): 2-node run PASSED on a stand-in dataset (job 215025, R212478, dgx001 + dgx002, 31 min)**
- While the project folder is unreadable: `mixes/asr_ta_codeswitch_test.yaml` = the user's
  `/scratch/users/astar/ares/sailorhb/datasets_v3/mixing_tamil_eng_codeswitch_30_ASR` (synthetic multi-speaker Tamil/English
  code-switch ASR, 40,622 samples, 4 GB; MDS language "unk"). No held-out split: validation uses the same data.
- `pbs/train_16gpu_2node.pbs` takes `qsub -v MIX_FILE=mixes/<mix>.yaml` (adds `--mix`, output dir `outputs/train_16gpu_<mix>`).
- Results (`outputs/train_16gpu_asr_ta_codeswitch_test/summary.txt`), global batch 32:
  | phase | loss 1 -> 60 | eval @30/60 | peak mem | s/it |
  |---|---|---|---|---|
  | H16A HF DDP | 0.478 -> 0.271 | 0.305 / 0.295 | 62.1 GiB | 1.4 |
  | M16A Megatron EP=8, DP=16 | 0.479 -> 0.281 | 0.328 / 0.315 | 17.3 GiB | 2.3 |
  - Resume 30 -> 60: max |loss diff| 0.0015 (HF) and 0.0014 (Megatron); data position 960 samples restored.
  - Sample partitions over 16 ranks: 128 each, disjoint (both trainers). NCCL busbw 416.6 GB/s.
- Multi-node launch (pbsdsh + `nscc/node_run.sh`) works. Next: the ST mix on 2 / 4 nodes once `/data/projects/13003558` is readable.

**2026-10-01 (later): Weights & Biases logging (`--wandb`)**
- Modelled on the user's sortformer script (`WANDB=1`, entity `i2r-llm`, project / run-name variables) and multimodal_trainer
  (run id kept in the training state so a resume continues the run). The multimodal_trainer enroot scripts hard-code an API key;
  not copied: the key comes from `WANDB_API_KEY` or `~/.netrc` (`nscc/env.sh`).
- `train_omni.py --wandb true` (or `WANDB=1`): adds `wandb` to `--report_to` (dropping the recipes' `none`), sets
  `--run_name` (sft) / `--wandb_project --wandb_exp_name` (megatron) and `WANDB_ENTITY/RUN_GROUP/TAGS/MODE/DIR/NOTES/RUN_ID`.
  Defaults: project `meralion_v4`, entity `i2r-llm` (user's choice), run name = basename of `--output_dir`, files in
  `<output_dir>/wandb`, no checkpoint upload. Run config also gets our options, the mix and `omni_effective`.
  Each checkpoint gets `omni_wandb.json` (run id, written by the logging rank: rank 0 for sft, last rank for Megatron).
  `--wandb_resume auto`: continue that run when the checkpoint is inside `--output_dir` (a restart), else a new run in its group.
- Network: compute nodes reach api.wandb.ai through the site proxy (probe job 215044); the container launchers now forward
  `http(s)_proxy` / `no_proxy` and the `WANDB_*` settings. (So the earlier "no PyPI from compute nodes" note was wrong: it goes via the proxy.)
- PBS training scripts: `qsub -v WANDB=1 ...` -> all phases of a job in group `<job name>_<job id>`.
- Tests: `tests/test_wandb.py` (CPU, 6 tests) pass; `tests/test_config.py` passes (2-node script added).
- **Live test, job 215052** (2 nodes, stand-in mix, `WANDB=1`): all phases exit 0; resume diff 0.0014 (HF) / 0.0017 (Megatron);
  partitions disjoint. W&B: 4 runs finished in group `omni_2x8_215052` with train/eval loss, config, "resumed from" notes on the
  B runs. H16A/H16B went to the key's default team (entity default was changed to i2r-llm mid-job); M16A/M16B are in
  `i2r-llm/meralion_v4`. First try (215045) failed at once: `report_to=['none','wandb']` -> fixed.
- Not exercised live: a restart continuing the same run id (only in `tests/test_wandb.py`).

**2026-10-01 (later): project data readable again; reserved queue only**
- `/data/projects/13003558` back to group 13003558; all 14 `mixes/st_v0.yaml` paths found.
- The aiq2 jobs 214952-214954 left the queue at 10:20 without starting ("would conflict with reservation ... terminated").
- **User rule: GPU jobs use only `-q R212478 -P 13003558_R4`** (12 nodes). All `pbs/*.pbs` headers switched from
  `-q normal -P 13003558`; `train_32gpu_4node.pbs` lost `place=scatter:excl` (the reservation is place=free); docs updated.
- Job 215070: `qsub -v WANDB=1 pbs/train_16gpu_2node.pbs` on the ST mix (609,632 samples/epoch), W&B `i2r-llm/meralion_v4`.
- **Job 215070 PASSED** (dgx001 + dgx003, 27 min, all phases exit 0), ST mix, global batch 32, `outputs/train_16gpu/summary.txt`:
  | phase | loss 1 -> 60 | eval @30/60 | peak mem | s/it |
  |---|---|---|---|---|
  | H16A HF DDP, 16 GPUs | 1.123 -> 0.751 | 0.831 / 0.831 | 61.7 GiB | 1.45 |
  | M16A Megatron EP=8, DP=16 | 1.121 -> 0.745 | 0.803 / 0.800 | 14.1 GiB | 2.2 |
  - Resume 30 -> 60: max |loss diff| 0.0041 (HF), 0.0034 (Megatron with optimizer): same as the earlier ST runs (0.003-0.004).
  - Partitions disjoint (both). NCCL busbw 415.8 GB/s. W&B: 4 runs in `i2r-llm/meralion_v4`, group `omni_2x8_215070`.
- Next: `pbs/train_32gpu_4node.pbs` (4 nodes) on the reserved queue; then real training runs.
- **Job 215088 PASSED: 4 nodes x 8 GPUs** (dgx001/003/005/006, R212478, 27 min, all phases exit 0), ST mix, global batch 32,
  `outputs/train_32gpu/summary.txt`. NCCL world 32: busbw 307.5 GB/s.
  | phase | loss 1 -> 60 | eval @30/60 | peak mem | s/it |
  |---|---|---|---|---|
  | H32A HF DDP, 32 GPUs | 1.208 -> 0.659 | 0.824 / 0.823 | 61.7 GiB | 0.91 |
  | M32A Megatron EP=8, DP=32 | 1.209 -> 0.658 | 0.804 / 0.803 | 14.0 GiB | 1.64 |
  - Resume 30 -> 60: max |loss diff| 0.0043 (HF), 0.0048 (Megatron with optimizer). Partitions over 32 ranks disjoint.
  - Eval loss (same fixed validation set) as on 16 GPUs (0.82-0.83 HF, 0.80 Megatron). The last-step train loss differs
    (0.66 vs 0.75): one 32-sample batch, and each world size reads a different sample order (canonical nodes 4 vs 2);
    likely batch noise, not checked further.
  - Speed per step at global batch 32: HF 1.45 s (16 GPUs) -> 0.91 s (32); Megatron 2.2 s -> 1.64 s.
  - W&B group `omni_4x8_215088` in `i2r-llm/meralion_v4`.
- The multi-node pipeline is tested on 2 and 4 nodes. Next: real training runs (recipe, steps, eval, data mix to decide).
