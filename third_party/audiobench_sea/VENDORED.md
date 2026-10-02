# AudioBench-SEA (vendored subset)

`audiobench/normalizer/` (whole) and `audiobench/metrics/{asr_metrics,base}.py` are copied from the team's new AudioBench
(`/home/users/astar/ares/sailorhb/scratch/git_repos/AudioBench-Temp`, git `f25d990`, 2026-10-02), without `__pycache__`.
`eval_omni.py` uses them for every ASR WER / CER: `LanguageNormalizerFactory.get_normalizer(lang)` (one instance per
language and process, same output as `normalize_text_for_language`) + `WERMetric` / `CERMetric`, disfluency removal off
(as their `asr.yaml`: `normalize_disfluency: False`); CER is the main metric for zh / th / lo / km / my as there.

## Patches (keep this list complete)

| File | Change | Why |
|---|---|---|
| `audiobench/__init__.py`, `audiobench/metrics/__init__.py` | replaced by a one-line docstring | the originals import the whole framework (config, every metric, judges) |
| `audiobench/normalizer/nemo_english.py` | `import spacy` wrapped in try/except (`SPACY_AVAILABLE = False`) | spaCy is only used with `AB_USE_SPACY=1` (default off), but the bare import made the English and Singlish normalisers fail to load without spaCy |

## Run-time requirements

- `jiwer.compute_measures` (removed in jiwer 4): `omni_eval/compat.py` adds it before import. Without it their metrics
  silently use an approximate positional WER (`method: fallback`); `omni_eval/bench.py` refuses such results.
- Optional libraries the normalisers use when present (number words, Thai segmentation, Tamil, Malay, NeMo English TN):
  `$CONTAINER_HOME/pydeps_eval`, installed by `container/install_pydeps_eval.sh`. Every score file records which were
  importable (`normalizer_libs`).
