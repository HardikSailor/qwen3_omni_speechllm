"""Per-sample text logic for Qwen3-Omni rows: `build_row` turns one raw MDS sample into an ms-swift row.

Ported from the old MERaLiON trainer (`toolkits/multimodal_trainer/modules/data_collators/`):
  - `standardize_response`  <- instruct_collator.py `InstructDataCollator._standardize_response`
  - `get_prompt`            <- instruct_collator.py `InstructDataCollator.get_prompt`
                               + prompt_utils.py `_get_prompt` / `get_asr_prompt` / `get_ac_prompt`
  - instruction dropout     <- instruct_collator.py `validate_and_parse_examples`
Behaviour matches the old code for the same inputs and random seed (tests/test_sea_text.py), except:
  - randomness comes from an explicit `random.Random` (per-sample, reproducible) instead of the global `random`;
  - AC prompts use `ac_prompt_type` (the old code passed `asr_prompt_type` for AC too);
  - an unknown task in the instruction list raises ValueError (the old `raise (f"...")` raised TypeError);
  - `asr_normalize_text` defaults to False (old config default: true), see RowConfig;
  - the language falls back to the mix YAML / dataset name when the MDS column is empty or missing.
Dropped on purpose: the MERaLiON system prompt, the `<SpeechHere>`/`<TextHere>` query template and the
left/right prompt split (the ms-swift Qwen3-Omni template does the chat formatting), and `context_text`
(for ST/ER it is the transcript and would leak the answer).
"""
import json
import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .instructions_lib import get_instructions_list
from .text_normalizers.wer_text_normalizer.normalizer import (
    add_speaker_asr,
    cast_lower_case,
    correct_words,
    preprocess_text_asr,
    remove_hash_tag,
    remove_no_sound_elements,
    standardize_apos,
    standardize_chinese,
    standardize_fillers,
    standardize_thai,
)

AUDIO_TAG = '<audio>'  # ms-swift placeholder; the Qwen3-Omni template expands it to audio tokens (13/s)
PROMPT_TYPES = (None, 'dataset', 'random', 'random_multilingual')


@dataclass
class RowConfig:
    # ASR/AC prompt: None = no prompt, 'dataset' = the MDS instruction_text,
    # 'random' = random English instruction, 'random_multilingual' = English or the sample's language.
    asr_prompt_type: Optional[str] = 'random'
    ac_prompt_type: Optional[str] = 'random'
    english_prompt_weight: float = 0.5   # P(English) for 'random_multilingual' (old code: hard-coded 0.5)
    augment_prompt: bool = False         # Vietnamese format hints (no punctuation / lower / upper case); train only
    # Old config default was true: it lowercases and strips punctuation from ASR targets. The model copies
    # target style (see the LibriSpeech smoke test), so off by default until the target convention is agreed.
    asr_normalize_text: bool = False
    instruction_dropout_rate: float = 0.0   # train only: replace the row by a text-only instruction pair
    text_instructions_path: Optional[str] = None  # JSON list of {"prompt": ..., "response": ...}
    # The old collator kept the (now unrelated) audio when it swapped in a text instruction pair; keep that
    # by default. False gives a text-only row (no <audio>, no audios).
    dropout_keep_audio: bool = True
    system_prompt: Optional[str] = None     # None = Qwen3-Omni default (no system prompt)

    def __post_init__(self):
        for name in ('asr_prompt_type', 'ac_prompt_type'):
            value = getattr(self, name)
            if value is not None and value.lower() not in PROMPT_TYPES[1:]:
                raise ValueError(f'{name}={value!r}; expected one of {PROMPT_TYPES}')
        self._text_instructions = None
        if self.text_instructions_path:
            with open(self.text_instructions_path) as f:
                self._text_instructions = json.load(f)


# --------------------------------------------------------------------------------------------------
# language

_NAME_LANG = re.compile(r'_(codeswitch|[a-z]{2,3}(?:-[A-Z]{2})?)_(?:\d+_)?(?:ASR|AC)$')
_NAME_PAIR = re.compile(r'_([a-z]{2,3}(?:-[A-Z]{2})?)_([a-z]{2,3}(?:-[A-Z]{2})?)_(?:\d+_)?ST$')


def _base_lang(code: str) -> str:
    return code.split('-')[0] if code else code


def resolve_languages(sample: Dict, meta: Dict) -> Dict[str, str]:
    """language / src_lang / tgt_lang for a sample.

    Order: the MDS `language` column (if non-empty), then the mix YAML (`language`, `src_lang`, `tgt_lang`),
    then the dataset name (`…_th_30_ASR`, `…_en_zh-CN_30_ST`). Empty string when unknown.
    """
    name = meta.get('name', '')
    src = meta.get('src_lang') or ''
    tgt = meta.get('tgt_lang') or ''
    m = _NAME_PAIR.search(name)
    if m:
        src = src or _base_lang(m.group(1))
        tgt = tgt or _base_lang(m.group(2))
    lang = (sample.get('language') or '').strip() or meta.get('language') or src
    if not lang:
        m = _NAME_LANG.search(name)
        lang = _base_lang(m.group(1)) if m else ''
    return {'language': lang, 'src_lang': src, 'tgt_lang': tgt}


# --------------------------------------------------------------------------------------------------
# response and prompt (ported)

def is_all_upper(text: str) -> bool:
    has_letters = False
    for char in text:
        if char.isalpha():
            has_letters = True
            if not char.isupper():
                return False
    return has_letters


def is_all_lower(text: str) -> bool:
    has_letters = False
    for char in text:
        if char.isalpha():
            has_letters = True
            if not char.islower():
                return False
    return has_letters


def contains_punctuation(text: str) -> bool:
    import unicodedata
    return any(unicodedata.category(c).startswith('P') for c in text)


def standardize_response(answer_text: str, task: str, lang_code: str, asr_normalize_text: bool) -> str:
    """Target text. Non-ASR tasks (ST, SQA, SDS, AC, …) are returned unchanged, as in the old code."""
    response_text = answer_text or ''
    if task != 'ASR':
        return response_text

    response_text = remove_no_sound_elements(response_text)
    response_text = correct_words(response_text)
    response_text = standardize_chinese(response_text)
    response_text = standardize_thai(response_text)

    if lang_code in ['id'] and is_all_upper(response_text):  # lower-case gigaspeech2 id
        response_text = cast_lower_case(response_text)
    if lang_code in ['codeswitch', 'en']:
        response_text = remove_hash_tag(response_text)
        response_text = standardize_apos(response_text)  # standardise gigaspeech
    if lang_code in ['codeswitch', 'en', 'ms', 'id']:
        response_text = standardize_fillers(response_text)

    response_text = add_speaker_asr(response_text)

    if asr_normalize_text:
        response_text = preprocess_text_asr(response_text)
    return response_text


def _random_prompt(rng: random.Random, prompt_type: Optional[str], dataset_instruction: str, lang_code: str,
                   task: str, eng_weights: float, response_text: str, augment_prompt: bool) -> str:
    """Port of prompt_utils._get_prompt; consumes `rng` in the same order as the old code used `random`."""
    if prompt_type is None:
        return ''
    if prompt_type.lower() == 'dataset':
        return dataset_instruction
    if prompt_type.lower() == 'random_multilingual':
        rand_lang_code = rng.choices(['en', lang_code], weights=(eng_weights, 1 - eng_weights), k=1)[0]
    else:
        rand_lang_code = 'en'

    instructions = get_instructions_list(lang_code=rand_lang_code)
    if task not in instructions:
        raise ValueError(f'{task} not in instructions list')
    prompt = rng.choice(instructions[task])

    if augment_prompt and lang_code.lower() in ['vi'] and response_text is not None:
        prompt += rng.choice(['\n', ' '])
        additional = []
        if not contains_punctuation(response_text):
            additional.append(rng.choice(instructions['no_punctuation']))
        if is_all_lower(response_text):
            additional.append(rng.choice(instructions['lowercase']))
        elif is_all_upper(response_text):
            additional.append(rng.choice(instructions['uppercase']))
        for i, extra in enumerate(rng.sample(additional, len(additional))):
            delimiters = ['.', '\n', '\n\n', ' '] if i == len(additional) - 1 else ['.', ',', '\n', ' ']
            prompt += extra + rng.choice(delimiters)
    return prompt


def get_prompt(instruction_text: str, answer_text: str, task: str, lang_code: str, cfg: RowConfig,
               rng: random.Random, is_train: bool) -> str:
    """Instruction for the user turn. ASR and AC may get a random (multilingual) prompt; all other tasks use
    the dataset's own instruction_text, as in the old code."""
    prompt = instruction_text or ''
    augment_prompt = cfg.augment_prompt and is_train and lang_code in ['vi']
    common = dict(dataset_instruction=prompt, lang_code=lang_code, eng_weights=cfg.english_prompt_weight,
                  response_text=answer_text or '', augment_prompt=augment_prompt)
    if task.strip()[:3] == 'ASR':
        prompt = _random_prompt(rng, cfg.asr_prompt_type, task='asr', **common)
    if task == 'AC':
        prompt = _random_prompt(rng, cfg.ac_prompt_type, task='ac', **common)
    return prompt


# --------------------------------------------------------------------------------------------------
# row

def build_row(sample: Dict, meta: Dict, cfg: RowConfig, rng: random.Random, is_train: bool = True,
              with_meta: bool = False,
              augment: Optional[Callable[[bytes, str, str], Tuple[bytes, str]]] = None) -> Optional[Dict]:
    """One raw MDS sample -> one ms-swift row, or None if the sample has no input audio.

    sample: decoded MDS columns (answer_text, context_audio, instruction_text, language, task, …).
    meta:   the dataset's entry in the mix YAML (name, task, and optional language / src_lang / tgt_lang).
    Returns {"messages": [user, assistant], "audios": [bytes]} (+ "_meta" if with_meta), where the user
    content is "<audio>" followed by the instruction. Audio stays as the original encoded bytes; the
    ms-swift template decodes and resamples it.
    augment: optional per-sample audio augmentation (omni_mds.audio.augment.AudioAugmenter), train only.
    """
    audio = sample.get('context_audio') or b''
    if not audio:
        return None
    task = (sample.get('task') or meta.get('task') or '').strip()
    langs = resolve_languages(sample, meta)
    lang = langs['language']
    answer_raw = sample.get('answer_text') or ''

    prompt = get_prompt(sample.get('instruction_text') or '', answer_raw, task, lang, cfg, rng, is_train)
    response = standardize_response(answer_raw, task, lang, cfg.asr_normalize_text)
    if augment is not None and is_train:  # old order: augment after prompt/response were built
        audio, response = augment(audio, response, task)
    user = AUDIO_TAG + prompt
    audios: List[bytes] = [audio]

    if is_train and cfg._text_instructions and cfg.instruction_dropout_rate > 0.0:
        if rng.random() < cfg.instruction_dropout_rate:
            replacement = rng.choice(cfg._text_instructions)
            response = replacement['response']
            if cfg.dropout_keep_audio:
                user = AUDIO_TAG + replacement['prompt']
            else:
                user, audios = replacement['prompt'], []

    messages = []
    if cfg.system_prompt:
        messages.append({'role': 'system', 'content': cfg.system_prompt})
    messages += [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': response}]
    row = {'messages': messages}
    if audios:
        row['audios'] = audios
    if with_meta:
        row['_meta'] = {'dataset': meta.get('name', ''), 'task': task, **langs}
    return row
