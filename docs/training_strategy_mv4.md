# MERaLiON-4 after stage 1: more steps, a new seed, or continue? (2026-10-02)

Question (user): mv4_lora_v0 trained one epoch of `mixes/mv4_v0.yaml` (18,500 steps x 256 = 4.74 M samples), and most
datasets are subsampled (e.g. ASR public: 319,664 samples, 10,000 per epoch). To beat the baseline on average, should the
next run train longer, start from the v0 checkpoint, or use another seed so that unseen samples get used?
Files: `configs/mv4_lora_v1_cont.yaml` (draft), `tools/average_lora.py`, `pbs/eval_omni.pbs` (`AVERAGE=`, parallel jobs),
`train_omni.py --adapters`. Run: `outputs/runs/mv4_lora_v0/v0-20261002-015359` (8 nodes, 14 h 44 min).

## How `choose` subsets are drawn (checked in the code)

- `omni_mds/mosaic_stream.py` uses mosaic `sampling_method='balanced'`. `streaming/base/dataset.py` (resample_streams)
  seeds the subset RNG with `shuffle_seed + epoch`, so **every epoch draws a new `choose` subset** of each subsampled
  dataset. `train_omni.py` passes `--data_seed` as `shuffle_seed`.
- So a run past step 18,511 (epoch 2) or a run with another `data_seed` sees mostly new samples of the subsampled sets;
  datasets used whole (no `choose`) are repeated. Overlap between two random subsets is about `choose / size` (3 % for
  ASR public).
- Because the seed is `shuffle_seed + epoch`, data seed 43 at epoch 0 = data seed 42 at epoch 1. Use well-separated seeds
  (1042, 2042, ...) for runs that should not repeat each other's subsets.
- Samples never drawn contribute nothing. What matters is whether the drawn ones cover the dataset's distribution;
  for large uniform read-speech sets 10 k random samples often nearly do, and more samples give diminishing returns.

## What large-scale training usually does

- Budget each task by samples / tokens (its **mix weight**), not by dataset epochs. Data is mostly seen once; only small
  high-quality sets are repeated (2-4 times at most). The mix weights, tuned in small proxy runs, are the main lever for
  "better on average".
- Train in stages: a long main stage, then a short low-LR **annealing** stage on the best data. Continuing from a
  checkpoint with a short LR re-warmup is standard; restarting from scratch is rare.
- **Average weights** of the last checkpoints (or keep an EMA): usually a small free gain.
- Decide from **benchmark curves**, not from eval loss.

## What v0 shows

Eval loss (67 validation sets x 32): 0.746 (1k) -> 0.695 (4k) -> 0.681 (10k) -> 0.677 (12k) -> 0.675 (18.5k); token
accuracy 0.801 -> 0.818. Flat from about 11 k on, with the cosine LR down to 1e-5 at the end. As a proxy this says the
recipe is close to saturating on this mix; the benchmark curve below decides.

## Decision

1. **Score the checkpoints** (base, 2k ... 18.5k) on the quick tier. If the benchmark average still rises clearly from
   12 k to 18.5 k: more data helps, continue (3). If flat: more samples of the same mix will not beat the baseline; change
   the mix weights first (up-weight the tasks where base / the baseline wins, keep the rest so they are not forgotten).
2. **Checkpoint average** of 16000 / 17000 / 18000 / 18500 (below). Free; if it wins, use it as the v0 result and as the
   init for (3).
3. **Next run: continue from v0's LoRA**, not from scratch: `configs/mv4_lora_v1_cont.yaml` (draft)
   - `adapters: .../checkpoint-18500` (or the average): weights only; new AdamW state, LR schedule, data order, W&B run.
   - `data_seed: 1042`: new `choose` subsets.
   - LR 5e-5 (v0 peak 1e-4), 300 warmup steps, cosine to 1e-5; 9,250 steps (half an epoch, ~7.5 h on 8 nodes).
   - `mix:` still `mv4_v0`: replace with the re-weighted mix from step 1 before launching.
   - Launch: `qsub -v CONFIG=configs/mv4_lora_v1_cont.yaml,RUN_NAME=mv4_lora_v1_cont pbs/train_mv4_lora_v0.pbs`
     (`INIT=<adapter dir>` overrides the config's `adapters:`).
4. **Not worth it:** the same recipe from scratch with only another seed (measures noise; LoRAs of different seeds cannot be
   averaged: their rank directions differ), or a 37 k-step run from scratch (2x the cost; the gain over continuing is
   only a cleaner LR schedule).

## `--adapters` init (train_omni.py)

ms-swift sft already loads `--adapters <dir>` as a trainable PEFT model (`pipelines/train/tuner.py`:
`from_pretrained(..., is_trainable=True)`); `train_omni.py check_init_adapters` adds: swift sft only (Megatron:
`--mcore_adapter` + `--finetune true`), refuses `--adapters` with `--resume_from_checkpoint`, checks the dir, warns when
`--lora_rank / --lora_alpha / --lora_dropout` differ from the adapter's config (ignored by ms-swift: the adapter decides),
and records `init_adapter` in `omni_effective.json`. `--resume_from_checkpoint` stays the exact restart (optimizer,
scheduler, mosaic position, same W&B run). Not yet run on GPUs: the first launch should be a smoke
(`qsub -v SMOKE=1,CONFIG=...,RUN_NAME=... -l walltime=01:00:00`) and step-1 loss should be close to v0's final loss
(~0.5), not the 1.44 of a fresh adapter.

## Checkpoint averaging (`tools/average_lora.py`)

A LoRA layer adds `scale * B @ A` (scale = alpha / r). For k checkpoints:
- **concat** (exact): `A' = [A_1; ...; A_k]`, `B' = [B_1 ... B_k] / k`, `r' = k r`, `alpha' = k alpha` (same scale), so
  `B' A' = mean_i(B_i A_i)`. Rank 256 for 4 checkpoints (4x the adapter size, 1.5 GB). A self-check compares a few
  modules against `mean_i(B_i A_i)` and fails the job above 2 % relative error (bf16 rounding is far below).
- **mean** (approximate): average A and B separately at rank 64. Only reasonable for close checkpoints of one run.
  For v0 16k-18.5k it differs from the exact average by <= 0.065 % (relative, per module): the four checkpoints are
  very close, so expect a small effect from averaging them.
Both are evaluated: `qsub -v AVERAGE="checkpoint-16000 checkpoint-17000 checkpoint-18000 checkpoint-18500" pbs/eval_omni.pbs`
writes `$RUN_DIR/avg4-16000-18500-{concat,mean}` (with `omni_average.json`) and evaluates them.

## Evaluation on several nodes

`pbs/eval_omni.pbs` jobs with disjoint `CKPTS` can run at once on the same `NAME/TIER`: predictions and scores go to
per-model dirs, logs to `$O/_logs/<job id>/`, each job needs its own `-o`. Finished sets are skipped, so a job whose
models are all generated only scores and judges. Free nodes: `python3 nscc/check_nodes.py pbs/eval_omni.pbs`.
2026-10-02 (jobs 215601-215607, 7 nodes, one each): base + 2k/4k/6k/8k; 10k+12k; 11k+13k; 14k+16k; 15k+18.5k; 17k+18k;
the two averages. Report: `outputs/evals/mv4_lora_v0/quick/report.md`.

## Results

(to fill in from the report)
