"""Parity and format tests for omni_mds.sea_text (PIPELINE_PLAN.md §6, phase 1).

Runs inside the training container (needs zstandard, soundfile, jiwer, regex, more_itertools, librosa); CPU only,
works on the login node, ~3 min:
    cd toolkits/qwen3_omni_speechllm
    apptainer exec --cleanenv -B /scratch --env PYTHONPATH=$PWD \
        /scratch/prj0000000234/sailorhb/container/swift_megatron_cu128.sif python tests/test_sea_text.py
The container has no pytest; the test_* functions are pytest-compatible if it is installed elsewhere.

The parity test feeds the same real MDS rows and the same random seed to the old InstructDataCollator
(imported file by file from toolkits/multimodal_trainer, without its package __init__, which pulls in model
code) and to the new build_row, and requires identical prompts and targets.
"""
import importlib
import io
import os
import random
import sys
import types
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from omni_mds.mds_io import iter_zstd_shard_head, list_subdirs, load_index  # noqa: E402
from omni_mds.mix import load_mix  # noqa: E402
from omni_mds.sea_text import AUDIO_TAG, RowConfig, build_row, resolve_languages  # noqa: E402

OLD_MODULES = '/scratch/prj0000000234/sailorhb/toolkits/multimodal_trainer/modules'
BASE = '/scratch/prj0000000234/zoux/datasets/datasets_mosaic_stage_AudioLLM_v2.1/datasets_multimodal'
ST_MIX = os.path.join(ROOT, 'mixes', 'st_v0.yaml')
ROWS_PER_DATASET = int(os.environ.get('ROWS_PER_DATASET', 50))

# ASR sets from the v4.3 mix, chosen to hit every language branch of _standardize_response
# (en, codeswitch, id upper-case, th, vi, zh, ms, ta, yue).
ASR_DATASETS = [
    'ASR_IMDA_PART3_conv_en_30_ASR', 'ASR_IMDA_PART4_conv_codeswitch_30_ASR', 'ASR_gigaspeech2_id_30_ASR',
    'ASR_gigaspeech2_th_30_ASR', 'ASR_gigaspeech2_vi_30_ASR', 'ASR_wenetspeech_zh_30_ASR',
    'ASR_MOBI_Malay_ms_30_ASR', 'ASR_i2r_tamil_phase_I_ta_30_ASR', 'ASR_King-ASR-423-1_yue_30_ASR',
    'ASR_SEAME_conversation_codeswitch_30_ASR',
]

# One or two datasets per remaining task (PQA accent/gender/emotion, SDS, CPQA, AQA, AC). AQA/AC are not in the
# v4.3 mix (commented out there) but exist on disk.
OTHER_TASK_DATASETS = [
    ('PQA/IMDA_PART3_conv_30_AR', 'PQA'), ('PQA/IMDA_PART3_conv_30_GR', 'PQA'), ('PQA/MELD_30_ER', 'PQA'),
    ('PQA/iemocap_30_ER', 'PQA'), ('SDS/IMDA_PART3_conv_30_SDS', 'SDS'), ('CPQA/ytb_qq_CPQA', 'CPQA'),
    ('AQA/clotho_30_AQA', 'AQA'), ('AC/AudioCaps_30_AC', 'AC'),
]

PARITY_CONFIGS = [
    dict(asr_prompt_type='random', asr_normalize_text=True, augment_prompt=False),   # old default config
    dict(asr_prompt_type='random', asr_normalize_text=False, augment_prompt=False),
    dict(asr_prompt_type='random_multilingual', asr_normalize_text=False, augment_prompt=True),
    dict(asr_prompt_type='dataset', asr_normalize_text=False, augment_prompt=False),
]


# ------------------------------------------------------------------------------------------------ helpers

@lru_cache(maxsize=None)
def old_collator_class():
    """Import the old InstructDataCollator without running modules/__init__ or data_collators/__init__."""
    pkg = 'oldmt'
    for name, path in [(pkg, OLD_MODULES), (f'{pkg}.data_collators', f'{OLD_MODULES}/data_collators'),
                       (f'{pkg}.text_normalizers', f'{OLD_MODULES}/text_normalizers')]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            sys.modules[name] = mod
    return importlib.import_module(f'{pkg}.data_collators.instruct_collator').InstructDataCollator


def make_old(cfg: dict):
    cls = old_collator_class()
    return cls(speech_feature_extractor=None, llm_tokenizer=None, is_train=True,
               asr_normalize_text=cfg['asr_normalize_text'], asr_prompt_type=cfg['asr_prompt_type'],
               ac_prompt_type=cfg['asr_prompt_type'], augment_prompt=cfg['augment_prompt'])


@lru_cache(maxsize=None)
def head_rows(dataset_dir: str, n: int):
    sub = list_subdirs(dataset_dir)[0]
    return tuple(iter_zstd_shard_head(sub, load_index(sub)[0], n))


def st_cases():
    mix = load_mix(ST_MIX)
    for split in ('train', 'validation'):
        for spec in mix[split]:
            for i, s in enumerate(head_rows(spec.path, ROWS_PER_DATASET)):
                yield f'{split}/{spec.name}#{i}', spec.meta(), s


def asr_cases():
    for name in ASR_DATASETS:
        path = f'{BASE}/train/ASR/{name[len("ASR_"):]}'
        if not os.path.isdir(path):
            continue
        for i, s in enumerate(head_rows(path, ROWS_PER_DATASET)):
            yield f'train/{name}#{i}', {'name': name, 'task': 'ASR'}, s


def other_cases():
    for rel, task in OTHER_TASK_DATASETS:
        path = f'{BASE}/train/{rel}'
        if not os.path.isdir(path):
            continue
        for i, s in enumerate(head_rows(path, ROWS_PER_DATASET)):
            yield f'train/{rel}#{i}', {'name': rel.split('/')[1], 'task': task}, s


def old_outputs(old, sample, meta, seed):
    example = dict(sample)
    example['language'] = resolve_languages(sample, meta)['language']
    example.setdefault('task', meta['task'])
    random.seed(seed)
    prompt = old.get_prompt(example)['text']
    response = old.get_response(example)['text']
    return prompt, response


def new_outputs(cfg, sample, meta, seed):
    # The old code used asr_prompt_type for AC too; the port uses ac_prompt_type, so set them equal here.
    row = build_row(sample, meta, RowConfig(**cfg, ac_prompt_type=cfg['asr_prompt_type']), random.Random(seed),
                    is_train=True)
    user, assistant = row['messages'][0]['content'], row['messages'][1]['content']
    assert user.startswith(AUDIO_TAG)
    return user[len(AUDIO_TAG):], assistant


# ------------------------------------------------------------------------------------------------ tests

def _parity(cases):
    checked, mismatches = 0, []
    for cfg in PARITY_CONFIGS:
        old = make_old(cfg)
        for k, (case_id, meta, sample) in enumerate(cases):
            seed = 1000 * k + 7
            want = old_outputs(old, sample, meta, seed)
            got = new_outputs(cfg, sample, meta, seed)
            checked += 1
            if want != got:
                mismatches.append((cfg, case_id, want, got))
    return checked, mismatches


def test_st_parity_with_old_collator():
    checked, mismatches = _parity(list(st_cases()))
    assert checked and not mismatches, mismatches[:3]


def test_asr_parity_with_old_collator():
    checked, mismatches = _parity(list(asr_cases()))
    assert checked and not mismatches, mismatches[:3]


def test_other_tasks_parity_with_old_collator():
    checked, mismatches = _parity(list(other_cases()))
    assert checked and not mismatches, mismatches[:3]


def test_ac_uses_ac_prompt_type():
    """Deliberate fix: AC prompts follow ac_prompt_type (the old code used asr_prompt_type)."""
    sample = {'context_audio': b'RIFF', 'answer_text': 'a dog barks', 'instruction_text': 'Caption it.', 'task': 'AC'}
    cfg = RowConfig(asr_prompt_type='random', ac_prompt_type='dataset')
    row = build_row(sample, {'name': 'AC_x_30_AC', 'task': 'AC'}, cfg, random.Random(0))
    assert row['messages'][0]['content'] == AUDIO_TAG + 'Caption it.'


def test_other_tasks_row_format():
    for case_id, meta, sample in other_cases():
        row = build_row(sample, meta, RowConfig(), random.Random(0))
        user, assistant = (m['content'] for m in row['messages'])
        if meta['task'] != 'AC':                                                # AC gets a random caption prompt
            assert user == AUDIO_TAG + sample['instruction_text'], case_id
        assert assistant == sample['answer_text'], case_id                      # non-ASR targets unchanged
        ctx = sample.get('context_text') or ''
        assert not ctx.strip() or ctx not in user, case_id                      # ER transcripts never in the prompt
        assert len(row['audios']) == 1 and row['audios'][0], case_id


def test_st_row_format():
    import soundfile as sf
    cfg = RowConfig()
    for case_id, meta, sample in st_cases():
        row = build_row(sample, meta, cfg, random.Random(0), with_meta=True)
        user, assistant = (m['content'] for m in row['messages'])
        assert [m['role'] for m in row['messages']] == ['user', 'assistant'], case_id
        assert user == AUDIO_TAG + sample['instruction_text'], case_id           # dataset instruction, as before
        assert assistant == sample['answer_text'], case_id                      # ST targets unchanged
        ctx = sample.get('context_text') or ''
        assert not ctx or ctx not in user, case_id                              # transcript never in the prompt
        assert len(row['audios']) == 1 and row['audios'][0] is sample['context_audio'], case_id
        assert sf.info(io.BytesIO(row['audios'][0])).samplerate == 16000, case_id
        assert row['_meta']['src_lang'] == meta['src_lang'] and row['_meta']['tgt_lang'] == meta['tgt_lang']


def test_language_parsing_matches_yaml():
    """The dataset-name fallback must agree with the explicit src/tgt in the ST mix."""
    for split, specs in load_mix(ST_MIX).items():
        for spec in specs:
            parsed = resolve_languages({}, {'name': spec.name})
            assert (parsed['src_lang'], parsed['tgt_lang']) == (spec.src_lang, spec.tgt_lang), spec.name
    assert resolve_languages({}, {'name': 'ASR_gigaspeech2_th_30_ASR'})['language'] == 'th'
    assert resolve_languages({}, {'name': 'ASR_IMDA_PART4_conv_codeswitch_30_ASR'})['language'] == 'codeswitch'
    assert resolve_languages({'language': 'ms'}, {'name': 'ASR_gigaspeech2_th_30_ASR'})['language'] == 'ms'


def test_instruction_dropout(tmp_path=None):
    import json
    import tempfile
    path = os.path.join(tmp_path or tempfile.mkdtemp(), 'text_instr.json')
    with open(path, 'w') as f:
        json.dump([{'prompt': 'What is 2+2?', 'response': '4'}], f)
    sample = {'context_audio': b'RIFF', 'answer_text': 'x', 'instruction_text': 'y', 'task': 'ST'}
    meta = {'name': 'ST_a_en_id_30_ST', 'task': 'ST'}
    # default = old behaviour: prompt/response swapped, the audio stays
    cfg = RowConfig(instruction_dropout_rate=1.0, text_instructions_path=path)
    row = build_row(sample, meta, cfg, random.Random(0), is_train=True)
    assert row['audios'] == [b'RIFF'] and row['messages'][0]['content'] == AUDIO_TAG + 'What is 2+2?'
    assert row['messages'][1]['content'] == '4'
    # text-only variant
    cfg = RowConfig(instruction_dropout_rate=1.0, text_instructions_path=path, dropout_keep_audio=False)
    row = build_row(sample, meta, cfg, random.Random(0), is_train=True)
    assert 'audios' not in row and row['messages'][0]['content'] == 'What is 2+2?'
    # never at eval time
    row = build_row(sample, meta, cfg, random.Random(0), is_train=False)
    assert row['audios'] == [b'RIFF'] and row['messages'][1]['content'] == 'x'


def test_dropout_parity_with_old_collator(tmp_path=None):
    """Same seed -> the old validate_and_parse_examples and build_row pick the same replacement pair."""
    import json
    import tempfile
    pairs = [{'prompt': f'q{i}', 'response': f'a{i}'} for i in range(20)]
    path = os.path.join(tmp_path or tempfile.mkdtemp(), 'text_instr.json')
    with open(path, 'w') as f:
        json.dump(pairs, f)
    cfg = dict(asr_prompt_type='random', asr_normalize_text=False, augment_prompt=False)
    old = make_old(cfg)
    old.text_instructions, old.instruction_dropout_rate = pairs, 0.5
    old.get_context = lambda ex: {'text': '', 'audio': {'array': None}}  # skip audio decoding
    new_cfg = RowConfig(**cfg, instruction_dropout_rate=0.5, text_instructions_path=path)
    cases = list(st_cases())[:40] + list(asr_cases())[:40]
    for k, (case_id, meta, sample) in enumerate(cases):
        example = dict(sample)
        example['language'] = resolve_languages(sample, meta)['language']
        random.seed(k)
        _, prompts, responses, _, _ = old.validate_and_parse_examples([example])
        row = build_row(sample, meta, new_cfg, random.Random(k), is_train=True)
        assert row['messages'][0]['content'] == AUDIO_TAG + prompts[0]['text'], case_id
        assert row['messages'][1]['content'] == responses[0]['text'], case_id


def test_augmentation_output():
    import soundfile as sf
    from omni_mds.audio.augment import AudioAugmenter
    random.seed(0)
    aug = AudioAugmenter(crowd_prob=1.0, change_pitch_prob=0.3, change_speed_prob=0.3, apply_rir_prob=0.3,
                         apply_noise_prob=1.0)
    for case_id, meta, sample in list(st_cases())[:30]:
        row = build_row(sample, meta, RowConfig(), random.Random(0), is_train=True, augment=aug)
        info = sf.info(io.BytesIO(row['audios'][0]))
        assert info.samplerate == 16000 and info.format == 'WAV' and 0 < info.duration <= 30.0, case_id
        assert row['messages'][1]['content'] == sample['answer_text'], case_id   # repeat_audio_prob=0: text unchanged
    row = build_row(sample, meta, RowConfig(), random.Random(0), is_train=False, augment=aug)
    assert row['audios'][0] is sample['context_audio']                          # never at eval time


def test_empty_audio_is_skipped():
    assert build_row({'context_audio': b'', 'task': 'ST'}, {'name': 'x', 'task': 'ST'}, RowConfig(),
                     random.Random(0)) is None


if __name__ == '__main__':
    st, asr, other = list(st_cases()), list(asr_cases()), list(other_cases())
    print(f'rows: ST {len(st)} from {len({c[0].split("#")[0] for c in st})} dataset splits, '
          f'ASR {len(asr)} from {len({c[0].split("#")[0] for c in asr})} datasets, '
          f'other tasks {len(other)} from {len({c[0].split("#")[0] for c in other})} datasets')
    for label, cases in (('ST', st), ('ASR', asr), ('PQA/SDS/CPQA/AQA/AC', other)):
        checked, mismatches = _parity(cases)
        print(f'{label} parity: {checked - len(mismatches)}/{checked} identical ({len(PARITY_CONFIGS)} configs)')
        for cfg, case_id, want, got in mismatches[:5]:
            print('  MISMATCH', cfg, case_id, '\n    old:', want, '\n    new:', got)
    for fn in (test_ac_uses_ac_prompt_type, test_other_tasks_row_format, test_st_row_format, test_language_parsing_matches_yaml, test_instruction_dropout,
               test_dropout_parity_with_old_collator, test_augmentation_output, test_empty_audio_is_skipped):
        fn()
        print('ok', fn.__name__)
    row = build_row(st[0][2], st[0][1], RowConfig(), random.Random(0), with_meta=True)
    print('example row:', {**row, 'audios': [f'<{len(row["audios"][0])} bytes>']})
