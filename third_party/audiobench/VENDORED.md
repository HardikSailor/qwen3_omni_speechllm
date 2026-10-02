# AudioBench (vendored)

`dataset_src/` is copied from `/data/projects/13003558/AudioBench/src/dataset_src` (the a2ap copy the MERaLiON team used
for MERaLiON-2/3 benchmarking; not a git checkout) on 2026-10-02, without `__pycache__` and the two files whose names
contain spaces. Upstream: https://github.com/AudioLLMs/AudioBench. Licence: CC BY-NC (see the upstream LICENSE); internal
research use only.

`eval_omni.py` uses the per-dataset processors as they are: `prepare_model_input()` builds the prompts (same
instructions and the same `shuffle(seed=42).select(n)` subsets as AudioBench) and `compute_score()` computes WER / BLEU /
METEOR / the Llama-3-70B judge scores, so our numbers are comparable with the MERaLiON-3 AudioBench numbers.

## Patches (keep this list complete)

| File | Change | Why |
|---|---|---|
| `dataset_src/eval_methods/eval_llama3_70b*.py` | `AutoTokenizer.from_pretrained(model_path)` -> `os.environ.get("OMNI_JUDGE_TOKENIZER", model_path)` | the processors pass the hub id `meta-llama/Meta-Llama-3-70B-Instruct`; compute nodes run offline, so we point at the local AWQ snapshot (same chat template) |
| same | `num_processes = 200` -> `int(os.environ.get("OMNI_JUDGE_PROCS", 200))` | concurrency of judge requests; default unchanged |

Run-time shims (in `omni_eval/compat.py`, not edits here): `jiwer.compute_measures` for jiwer 4, and `evaluate.load`
of `sacrebleu` / `meteor` / `bleu` redirected to the local copies in `omni_mds/metrics/hf_metrics` (offline nodes).
