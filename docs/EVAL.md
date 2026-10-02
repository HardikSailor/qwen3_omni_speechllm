# Checkpoint evaluation (`eval_omni.py`)

Written 2026-10-02. Evaluates the base model and LoRA checkpoints on AudioBench-style test sets, the way MERaLiON-3 was
benchmarked (`/data/projects/13003558/AudioBench`, `a2ap_scripts`: AudioBench processors + Llama-3-70B-Instruct AWQ
judge on vLLM).

## Run it

```bash
# all checkpoints of the current run + the base model, 200 samples per set, with judge (one node, ~8 GPUs)
qsub -v TIER=quick pbs/eval_omni.pbs
# chosen checkpoints only / a few sets / final numbers
qsub -v CKPTS="base checkpoint-18500",TIER=full pbs/eval_omni.pbs
qsub -v CKPTS="checkpoint-5000",ONLY="librispeech_test_clean covost2_en_zh_test",TIER=quick pbs/eval_omni.pbs
# report over whatever has been scored (login node is fine: pure python + json)
python eval_omni.py report outputs/evals/mv4_lora_v0/quick/*/ --out outputs/evals/mv4_lora_v0/quick/report
```
Name a free healthy node if the scheduler might pick a2ap-dgx011: `-l select=1:host=a2ap-dgxNNN:ngpus=8:ncpus=112:mem=1880GB`.

Outputs: `outputs/evals/<run>/<tier>/<model>/` with `<set>.json` (predictions, AudioBench format), `<set>_<metric>_score.json`
(per-sample details, judge explanations), `scores.json`, `meta.json`; `report.md` / `report.csv` per tier;
`check_suite.log`, `judge_server.log`. Every step skips finished work, so a rerun only does what is missing.

## How it works

| Step | What |
|---|---|
| suite | `evals/suite_v0.yaml`: 69 sets (ASR 29, ST 15, SQA 10, SDS 4, PQA 11). Tiers: `smoke` 8, `quick` 200, `full` 2000 samples per set (AudioBench's `shuffle(seed=42).select(n)`) |
| prompts | AudioBench sets: the vendored AudioBench processor (`third_party/audiobench`, same instruction lists, `random.seed(42)`). Other sets (SEA-AudioBench, GigaSpeech2, YouTube SG, SEAME): the set's own instruction, or `prompt:` in the YAML |
| generate | ms-swift `TransformersEngine`, the training template (no system prompt, `<audio>` + instruction), greedy; 8 processes = 8 model copies. The base model is loaded once per job (~7 min); each checkpoint is then switched in within seconds: PEFT `unload()` (the original Linear modules come back, base weights untouched) + a fresh `PeftModel.from_pretrained`, i.e. exactly a fresh load (`load_adapter` + `delete_adapter` on the wrapped model gave different outputs, job 215360). Every job re-checks this: the last model is loaded fresh in a new process and 6 sets are compared (`VERIFY_SWITCH`, `verify_switch.log`). Clips are dealt to the 8 ranks longest first; batches hold at most 8 clips and 480 s of audio; output budget max(per-task minimum, 15 tokens per audio second) for ASR / ST |
| score | ASR (all sets): the team's AudioBench-SEA language normalisers + WER / CER (`third_party/audiobench_sea`), chosen by the set's `lang` (en, sgen, zh, ms, ta, id, th, vi, tl); CER is the main metric for zh / th, Tamil reports both; AudioBench sets also keep AudioBench's old WER as `wer_ab` (MERaLiON-3 comparability). BLEU / judge: AudioBench sets use the processor's own `compute_score`; generic sets use sacrebleu (`zh` tokenizer for Chinese targets) and the same judge functions |
| judge | `casperhansen/llama-3-70b-instruct-awq` on vLLM 0.17.1 (our container): 8 servers, TP=1, one per GPU (TP=4 crashed in custom all-reduce inside enroot), 8 `score --judge --shard i/8` workers in parallel; prompts unchanged from AudioBench: `llama3_70b_judge` = 0-5 x 20, binary = 0/1 x 100 (the AudioBench processor picks binary for MCQ, emotion, gender, accent) |
| report | sets x models table and per-task averages over sets every model has (WER / CER shown in %) |

Optional normaliser libraries for eval jobs: `$CONTAINER_HOME/pydeps_eval`, appended to `PYTHONPATH` by the PBS scripts.

Why not vLLM for generation: vLLM 0.17.1 cannot load Qwen3-Omni LoRA adapters (`DEPLOYMENT.md` §1). For final numbers
on a merged model the engine can be swapped (planned, not done).

## Caveats

- Long audio: Qwen3-Omni accepts up to ~40 min of audio per request (64 K context; ms-swift turns off the Whisper
  feature extractor's 30 s truncation), and it fits in memory (job 215360: 206 clips up to 33 min, 2.5 min on 8 GPUs).
  But whole-recording ASR stops early (the 33-min clip: one sentence; 10-min clips: about half), so ASR and ST clips
  longer than 30 s are transcribed / translated in 30 s chunks joined in order (default `chunk_s` per task,
  AudioBench-SEA's `simple` chunking), all tiers, full length; speech QA, summary and paralinguistic sets (which need
  the whole context) always get the whole recording. `ytb_asr_batch3_chinese_whole` (full tier) keeps one request per recording to track native long-audio ASR.
- The AudioBench-SEA normalisers use optional libraries when present (NeMo TN, malaya, num2words, cn2an, pythainlp,
  attacut, indic-nlp); ours are in `$CONTAINER_HOME/pydeps_eval` (`container/install_pydeps_eval.sh`), and each
  score file lists which were importable (`normalizer_libs`). Numbers match the team's only with the same set.
- The mix's validation part (eval loss) uses test splits of IMDA, SEAME, CoVoST2, GigaSpeech2, LibriSpeech, AIShell,
  public_sg_speech_qa, cn_college: fine for scoring, but selecting checkpoints by eval loss is not blind on those sets.
- Judge scores depend on the judge; compare only numbers from this judge and prompt set.
