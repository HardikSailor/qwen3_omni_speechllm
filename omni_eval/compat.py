"""Run-time shims so the vendored AudioBench code (third_party/audiobench) runs in our container unchanged.

- jiwer 4 removed `compute_measures`, which 36 AudioBench processors import.
- Compute nodes are offline (HF_HUB_OFFLINE=1): `evaluate.load("sacrebleu" | "meteor" | "bleu")` is redirected to the
  local metric scripts in omni_mds/metrics/hf_metrics.
- `dataset_src` (the AudioBench package name) and `audiobench` (the vendored AudioBench-SEA normalisers / WER / CER,
  third_party/audiobench_sea) are put on sys.path. The jiwer shim must come first: AudioBench-SEA's metrics fall back
  to an approximate WER when `from jiwer import compute_measures` fails.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
AUDIOBENCH = os.path.join(REPO, 'third_party', 'audiobench')
AUDIOBENCH_SEA = os.path.join(REPO, 'third_party', 'audiobench_sea')
# optional libraries the AudioBench-SEA normalisers use when present (number words, segmentation, NeMo TN);
# installed for eval jobs in $CONTAINER_HOME/pydeps_eval (container/install_pydeps_eval.sh)
SEA_OPTIONAL_LIBS = ('nemo_text_processing', 'malaya', 'num2words', 'cn2an', 'pythainlp', 'attacut', 'indicnlp',
                     'indic_numtowords', 'spacy')
HF_METRICS = os.path.join(REPO, 'omni_mds', 'metrics', 'hf_metrics')

_installed = False


def install():
    global _installed
    if _installed:
        return
    import jiwer
    if not hasattr(jiwer, 'compute_measures'):

        def compute_measures(truth, hypothesis):
            out = jiwer.process_words(truth, hypothesis)
            return {'wer': out.wer, 'mer': out.mer, 'wil': out.wil, 'wip': out.wip, 'hits': out.hits,
                    'substitutions': out.substitutions, 'deletions': out.deletions, 'insertions': out.insertions}

        jiwer.compute_measures = compute_measures

    import evaluate
    orig_load = evaluate.load
    local = {name: os.path.join(HF_METRICS, name, f'{name}.py') for name in ('sacrebleu', 'meteor', 'bleu')}

    def load(path, *args, **kwargs):
        return orig_load(local.get(path, path), *args, **kwargs)

    evaluate.load = load
    for p in (AUDIOBENCH, AUDIOBENCH_SEA):
        if p not in sys.path:
            sys.path.insert(0, p)
    _installed = True


def sea_optional_libs():
    import importlib.util
    return {m: importlib.util.find_spec(m) is not None for m in SEA_OPTIONAL_LIBS}
