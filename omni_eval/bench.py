"""Eval suites: which test sets, how prompts are built, how predictions are scored.

A suite YAML (evals/suite_*.yaml) lists test sets saved with `datasets.save_to_disk`. Two processor kinds:

- `audiobench`: the AudioBench processor of the same name (third_party/audiobench/dataset_src/<name>.py) builds the
  prompts, exactly as in the MERaLiON-3 AudioBench runs (same instruction lists, random.seed, subset), and scores
  BLEU / judge metrics and the old-style WER (`wer_ab`, kept for comparison with MERaLiON-3 numbers).
- `generic`: our processor for sets AudioBench has no processor for (SEA-AudioBench, GigaSpeech2, YouTube SG sets).
  Prompt = the set's own instruction (or `prompt:` in the YAML); BLEU is sacrebleu (zh tokenizer for Chinese
  targets); judge metrics call the AudioBench Llama-3-70B judge functions.

ASR `wer` / `cer` (every set, both processors): the team's AudioBench-SEA language normalisers and WER / CER metrics
(third_party/audiobench_sea), chosen by the set's `lang` (en, sgen = Singlish, zh, ms, ta, id, th, vi, tl, my, km,
lo). As there, CER is the main metric for zh / th / lo / km / my.

Subsets: `shuffle(seed=42).select(range(n))` as AudioBench's `number_of_samples`, so a tier's n picks the same
samples as AudioBench with that number.
"""
import importlib
import io
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from . import compat

TASK_TOKENS = {'ASR': 384, 'ST': 384, 'SQA': 512, 'SDS': 768, 'PQA': 256}
# output-token budget per second of audio (transcripts / translations grow with the clip); 0 = fixed budget
TASK_TOKENS_PER_S = {'ASR': 15, 'ST': 15}
MAX_NEW_TOKENS_CAP = 16384
# clips longer than this are transcribed / translated in chunks (no reasoning across the recording needed); speech QA,
# summary and paralinguistic sets always get the whole recording. A set can override with `chunk_s` (null = never).
TASK_CHUNK_S = {'ASR': 30, 'ST': 30}
JUDGE_METRICS = ('llama3_70b_judge', 'llama3_70b_judge_binary')
ASR_METRICS = ('wer', 'cer')
SPEAKER_TAG = re.compile(r'<\s*speaker\s*\d+\s*(,\s*[mf])?\s*>\s*:?', re.IGNORECASE)


@dataclass
class EvalSet:
    name: str
    path: str
    task: str
    metrics: List[str]
    processor: str = 'audiobench'
    n: int = -1                      # samples for the chosen tier (-1 = all)
    max_new_tokens: int = 384        # at least this many; ASR / ST get more for long clips (tokens_per_s)
    tokens_per_s: float = 0
    prompt: Optional[str] = None     # generic only: replaces the set's instruction
    lang: Optional[str] = None       # language of the answer: ASR normaliser, BLEU tokenizer
    strip_speaker_tags: bool = False  # drop "<SPEAKER1,M>:" from reference and prediction before WER / BLEU
    max_audio_s: Optional[float] = None  # drop longer clips (by the set's audio_length / audio_duration) before subsetting
    chunk_s: Optional[float] = None  # longer clips are cut into chunks of this length, transcribed one by one, joined
    min_ref_rate: Optional[float] = None  # score only samples whose reference has >= this fraction of the set's median
                                          # characters per audio second (drops truncated / empty transcripts)
    extra: Dict = field(default_factory=dict)

    def token_budget(self, duration_s: float) -> int:
        return min(MAX_NEW_TOKENS_CAP, max(self.max_new_tokens, math.ceil(self.tokens_per_s * duration_s)))


def load_suite(path: str, tier: str = 'quick', only: Optional[List[str]] = None) -> List[EvalSet]:
    cfg = yaml.safe_load(open(path))
    roots = cfg.get('roots', {})
    tiers = cfg.get('tiers', {'quick': 200, 'full': -1})
    if tier not in tiers:
        raise ValueError(f'tier {tier!r} not in {list(tiers)}')
    tokens = {**TASK_TOKENS, **cfg.get('max_new_tokens', {})}
    out = []
    for d in cfg['datasets']:
        d = dict(d)
        name = d.pop('name')
        if only and name not in only:
            continue
        p = d.pop('path')
        root, _, rest = p.partition('/')
        p = os.path.join(roots[root], rest) if root in roots else p
        metrics = d.pop('metric')
        metrics = [metrics] if isinstance(metrics, str) else list(metrics)
        task = d.pop('task')
        # per-tier override: `full: 2000` (sample count) or `quick: {n: 200, max_audio_s: 60}`
        over = d.pop(tier, None)
        for t in tiers:
            d.pop(t, None)
        over = {'n': over} if isinstance(over, int) else dict(over or {})
        d.update({k: v for k, v in over.items() if k != 'n'})
        n = over.get('n', tiers[tier])
        processor = d.pop('processor', None)
        if processor is None:
            processor = 'audiobench' if os.path.exists(
                os.path.join(compat.AUDIOBENCH, 'dataset_src', f'{name}.py')) else 'generic'
        s = EvalSet(name=name, path=p, task=task, metrics=metrics, processor=processor, n=n,
                    max_new_tokens=d.pop('max_new_tokens', tokens.get(task, 384)),
                    tokens_per_s=d.pop('tokens_per_s', TASK_TOKENS_PER_S.get(task, 0)), prompt=d.pop('prompt', None),
                    lang=d.pop('lang', None), strip_speaker_tags=d.pop('strip_speaker_tags', False),
                    max_audio_s=d.pop('max_audio_s', None), chunk_s=d.pop('chunk_s', TASK_CHUNK_S.get(task)),
                    min_ref_rate=d.pop('min_ref_rate', None), extra=d)
        if n == 0:  # `<tier>: {n: 0}`: not in this tier
            continue
        if task == 'ASR' and any(m in ASR_METRICS for m in metrics) and not s.lang:
            raise ValueError(f'{name}: ASR sets need `lang` (normaliser)')
        out.append(s)
    if only:
        missing = set(only) - {s.name for s in out}
        if missing:
            raise ValueError(f'not in the suite: {sorted(missing)}')
    return out


def _load_rows(s: EvalSet) -> List[Dict]:
    """The selected rows as plain dicts; audio stays encoded ({'bytes', 'path'}), nothing is decoded."""
    from datasets import load_from_disk
    ds = load_from_disk(s.path)
    if hasattr(ds, 'keys'):  # DatasetDict
        ds = ds['test'] if 'test' in ds else ds[list(ds.keys())[0]]
    if s.max_audio_s:
        col = next(c for c in ('audio_length', 'audio_duration') if c in ds.column_names)
        keep = [i for i, x in enumerate(ds[col]) if x is not None and x <= s.max_audio_s]
        ds = ds.select(keep, keep_in_memory=True)
    if s.n != -1 and s.n < len(ds):
        ds = ds.shuffle(seed=42, keep_in_memory=True).select(range(s.n), keep_in_memory=True)
    return ds.with_format('arrow')[:].to_pylist()


def _audiobench_processor(s: EvalSet, rows):
    compat.install()
    module_name = s.extra.get('module', s.name)
    mod = importlib.import_module(f'dataset_src.{module_name}')
    if 'cls' in s.extra:
        cls = getattr(mod, s.extra['cls'])
    else:  # usually <name>_dataset, but e.g. imda_gr_sentence -> imda_gr_sentence_test_dataset
        (cls,) = [v for k, v in vars(mod).items() if k.endswith('_dataset') and isinstance(v, type)]
    return cls(rows, -1)  # rows are already the subset


def _text(v):
    return v.get('text') if isinstance(v, dict) else v


def _audio(v):
    return v['audio'] if isinstance(v, dict) and 'audio' in v else v


def audio_duration(a) -> float:
    """Seconds, from the encoded header (cheap); falls back to 16 kHz 16-bit PCM size."""
    import soundfile as sf
    try:
        info = sf.info(io.BytesIO(a) if isinstance(a, bytes) else a)
        return info.frames / info.samplerate
    except Exception:
        return len(a) / 32000 if isinstance(a, bytes) else 30.0


def split_audio(a, chunk_s: float, min_tail_s: float = 1.0) -> List[bytes]:
    """Fixed-length chunks (AudioBench-SEA's default `simple` chunking) as 16-bit WAV bytes at the native rate; a tail
    shorter than min_tail_s is added to the previous chunk."""
    import soundfile as sf
    wav, sr = sf.read(io.BytesIO(a) if isinstance(a, bytes) else a, dtype='int16', always_2d=False)
    step = int(chunk_s * sr)
    bounds = list(range(0, len(wav), step))
    if len(bounds) > 1 and len(wav) - bounds[-1] < min_tail_s * sr:
        bounds.pop()
    out = []
    for k, b in enumerate(bounds):
        e = bounds[k + 1] if k + 1 < len(bounds) else len(wav)
        buf = io.BytesIO()
        sf.write(buf, wav[b:e], sr, format='WAV', subtype='PCM_16')
        out.append(buf.getvalue())
    return out


def build_items(s: EvalSet) -> List[Dict]:
    """[{idx, text, answer, task_type, duration, audio: bytes, ...}] in a fixed order (same on every rank)."""
    rows = _load_rows(s)
    random.seed(42)
    if s.processor == 'audiobench':
        items = _audiobench_processor(s, rows).prepare_model_input()
    else:
        items = []
        for r in rows:
            item = {'audio': _audio(r['context']), 'text': s.prompt or _text(r['instruction']),
                    'answer': _text(r['answer']), 'task_type': s.task}
            for k in ('language', 'language_target'):
                if k in r:
                    item[k] = r[k]
            items.append(item)
    out = []
    for i, it in enumerate(items):
        a = it.pop('audio')
        a = a.get('bytes') or a.get('path') if isinstance(a, dict) else a
        if not a:
            raise ValueError(f'{s.name}: sample {i} has no audio')
        out.append({'idx': i, **it, 'duration': round(audio_duration(a), 2), 'audio': a})
    return out


# ---------------------------------------------------------------- scoring

_SEA_NORMALIZERS = {}


def sea_normalize(texts, lang):
    """= audiobench.normalizer.language_normalizer.normalize_text_for_language, but one normaliser per language and
    process (it builds a new one per call; NeMo's English grammars take minutes to build)."""
    compat.install()
    from audiobench.normalizer.language_normalizer import LanguageNormalizerFactory
    if lang not in _SEA_NORMALIZERS:
        _SEA_NORMALIZERS[lang] = LanguageNormalizerFactory.get_normalizer(lang, False)
    return [_SEA_NORMALIZERS[lang].normalize(t) for t in texts]


def _asr_sea(preds, refs, lang, metric):
    """WER / CER with the AudioBench-SEA language normaliser + metric (corpus level: errors / reference units)."""
    compat.install()
    from audiobench.metrics.asr_metrics import CERMetric, WERMetric
    ref = sea_normalize(refs, lang)
    hyp = sea_normalize(preds, lang)
    value, details = (WERMetric if metric == 'wer' else CERMetric)(lang).compute(hyp, ref)
    if details.get('method') != 'jiwer':  # their fallback is an approximation (e.g. jiwer API missing): refuse
        raise RuntimeError(f'AudioBench-SEA {metric} fell back to {details.get("method")!r}')
    return {metric: value, 'language': lang, 'normalizer_libs': compat.sea_optional_libs(),
            'details': {k: v for k, v in details.items() if k != 'sample_scores'}}


def _generic_bleu(preds, refs, lang):
    import evaluate
    compat.install()
    tok = 'zh' if (lang or '').lower() in ('zh', 'chinese', 'yue', 'cantonese') else '13a'
    res = evaluate.load('sacrebleu').compute(predictions=preds, references=[[r] for r in refs], tokenize=tok)
    return {'bleu': res['score'], 'tokenize': tok}


def _generic_judge(metric, questions, refs, preds):
    compat.install()
    from dataset_src.eval_methods.eval_llama3_70b import llama3_70b_as_judge, llama3_70b_as_judge_binary
    fn = llama3_70b_as_judge_binary if metric == 'llama3_70b_judge_binary' else llama3_70b_as_judge
    res, details = fn('meta-llama/Meta-Llama-3-70B-Instruct', [questions, refs, preds])
    return {metric: res, 'details': details}


def complete_refs(s: EvalSet, records: List[Dict]):
    """(kept records, dropped idx): with min_ref_rate, drop samples whose reference is much shorter than the audio
    (reference chars per second < min_ref_rate x the set's median). Depends on the references only, so every model is
    scored on the same samples."""
    if not s.min_ref_rate or not records:
        return records, []
    import statistics
    rate = [len(r['answer'] or '') / max(r['duration'], 1e-3) for r in records]
    cut = s.min_ref_rate * statistics.median(rate)
    keep = [r for r, x in zip(records, rate) if x >= cut]
    return keep, [r['idx'] for r, x in zip(records, rate) if x < cut]


def score(s: EvalSet, records: List[Dict], metric: str) -> Dict:
    """AudioBench-style result dict; the headline value is result[metric] (a number, or {'judge_score': ...})."""
    records, dropped = complete_refs(s, records)
    result = _score(s, records, metric)
    if s.min_ref_rate:
        result['dropped_incomplete_refs'] = dropped
    return result


def _score(s: EvalSet, records: List[Dict], metric: str) -> Dict:
    preds = [r['model_prediction'] or '' for r in records]
    refs = [r['answer'] or '' for r in records]
    if s.strip_speaker_tags or metric in ASR_METRICS:
        preds = [SPEAKER_TAG.sub(' ', p).strip() for p in preds]
        refs = [SPEAKER_TAG.sub(' ', x).strip() for x in refs]
    if metric in ASR_METRICS:
        return _asr_sea(preds, refs, s.lang, metric)
    if s.processor == 'audiobench':
        compat.install()
        proc = _audiobench_processor(s, [])
        if metric == 'wer_ab':  # AudioBench's own WER (MERaLiON-3 numbers)
            return {'wer_ab': proc.compute_score([dict(r) for r in records], metrics='wer')['wer']}
        return proc.compute_score([dict(r) for r in records], metrics=metric)
    lang = s.lang or (records[0].get('language_target') or records[0].get('language') if records else None)
    if metric == 'bleu':
        return _generic_bleu(preds, refs, lang)
    if metric in JUDGE_METRICS:
        return _generic_judge(metric, [r['text'] for r in records], refs, preds)
    raise ValueError(f'{s.name}: unsupported metric {metric!r} for the generic processor')


def headline(result: Dict, metric: str) -> float:
    v = result[metric]
    return float(v['judge_score'] if isinstance(v, dict) else v)


HIGHER_IS_BETTER = {'wer': False, 'cer': False, 'wer_ab': False, 'bleu': True, 'meteor': True,
                    'llama3_70b_judge': True, 'llama3_70b_judge_binary': True}
RATE_METRICS = ('wer', 'cer', 'wer_ab')  # fractions, shown in %
