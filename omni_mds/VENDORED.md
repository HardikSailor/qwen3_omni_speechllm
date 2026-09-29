# Vendored files

Copied unchanged from `toolkits/multimodal_trainer/modules` on 2026-09-28 (see PIPELINE_PLAN.md §1, principle 4).
Do not edit these copies; put changes in `sea_text.py` instead so the copies stay diff-able against the originals.

| Copy | Source | md5 of source at copy time |
|---|---|---|
| `omni_mds/instructions_lib/burmese.py` | `data_collators/instructions_lib/burmese.py` | `395251f13abe` |
| `omni_mds/instructions_lib/chinese.py` | `data_collators/instructions_lib/chinese.py` | `d6b2681ce9b4` |
| `omni_mds/instructions_lib/english.py` | `data_collators/instructions_lib/english.py` | `7fe0e7fb6ac9` |
| `omni_mds/instructions_lib/indonesian.py` | `data_collators/instructions_lib/indonesian.py` | `95d11e348595` |
| `omni_mds/instructions_lib/__init__.py` | `data_collators/instructions_lib/__init__.py` | `3315bdd3d4c6` |
| `omni_mds/instructions_lib/japanese.py` | `data_collators/instructions_lib/japanese.py` | `061cd20750d4` |
| `omni_mds/instructions_lib/korean.py` | `data_collators/instructions_lib/korean.py` | `268b50e186cc` |
| `omni_mds/instructions_lib/lao.py` | `data_collators/instructions_lib/lao.py` | `87b38009bbe7` |
| `omni_mds/instructions_lib/malay.py` | `data_collators/instructions_lib/malay.py` | `f183e7610aa5` |
| `omni_mds/instructions_lib/nepali.py` | `data_collators/instructions_lib/nepali.py` | `9a8b0f9335ea` |
| `omni_mds/instructions_lib/tagalog.py` | `data_collators/instructions_lib/tagalog.py` | `b13746150085` |
| `omni_mds/instructions_lib/tamil.py` | `data_collators/instructions_lib/tamil.py` | `e4029d9e2d31` |
| `omni_mds/instructions_lib/thai.py` | `data_collators/instructions_lib/thai.py` | `fcde3864b478` |
| `omni_mds/instructions_lib/vietnamese.py` | `data_collators/instructions_lib/vietnamese.py` | `5a1a93003208` |
| `omni_mds/text_normalizers/__init__.py` | `text_normalizers/__init__.py` | `d41d8cd98f00` |
| `omni_mds/text_normalizers/wer_text_normalizer/basic.py` | `text_normalizers/wer_text_normalizer/basic.py` | `aa94d6156f96` |
| `omni_mds/text_normalizers/wer_text_normalizer/english.json` | `text_normalizers/wer_text_normalizer/english.json` | `33bb184e238f` |
| `omni_mds/text_normalizers/wer_text_normalizer/__init__.py` | `text_normalizers/wer_text_normalizer/__init__.py` | `d41d8cd98f00` |
| `omni_mds/text_normalizers/wer_text_normalizer/normalizer.py` | `text_normalizers/wer_text_normalizer/normalizer.py` | `31b60f28232c` |
| `omni_mds/text_normalizers/wer_text_normalizer/whisper_english.py` | `text_normalizers/wer_text_normalizer/whisper_english.py` | `17bdd2d74704` |

## Added 2026-09-28 (everything the other tasks need, not only ST)

| Copy | Source (under `toolkits/multimodal_trainer/`) | md5 of source at copy time | What it is |
|---|---|---|---|
| `omni_mds/metrics/` | `modules/metrics/` | `3b830ddac5f0` | Metrics: WER/MER (jiwer), BLEU (sacrebleu), METEOR, `get_metric_fn` per dataset path, local HF metric scripts |
| `omni_mds/audio/data_augmentation.py` | `modules/data_collators/data_augmentation.py` | `1444659aff33` | Audio augmentation (pitch, speed, RIR, crowd noise). Batch-based; the per-sample wrapper goes in `omni_mds/audio/augment.py` |
| `omni_mds/language_mapping.py` | `modules/data_collators/language_mapping.py` | `f3304c3e7c6e` | Language → id table (only needed for language tokens, guide §5.7) |
| `omni_mds/prompt_utils_orig.py` | `modules/data_collators/prompt_utils.py` | `0e7cb6f1bc10` | Reference copy. Ported into `sea_text.py`; also holds `form_mm_prompt` (used by the old interleave collator for text context) |
| `tools/vendored/convert_hf_to_mds.py` | `scripts/mds_helper_scripts/convert_hf_to_mds.py` | `1deea36c01ff` | How the MDS shards were made from HF datasets (needs mosaicml-streaming). Use it to add new sets in the same format |
| `tools/vendored/convert_mds_to_mds.py` | `scripts/mds_helper_scripts/convert_mds_to_mds.py` | `c6c217780167` | Re-writing MDS shards |
| `tools/vendored/generate_dataset_config.py` | `utils/generate_dataset_config.py` | `28b61f360053` | Builds a mix YAML from a dataset folder tree |
| `tools/vendored/data_sampling/` | `utils/data_sampling/` | `5e52f224eff0` | Sample counts and UniMax sampling used to set `choose` in the old mixes |

Not copied, on purpose: the collators' tokenisation / `<SpeechHere>` split, `batch_processor.py`, `trainer.py`, `distributed/`, `models/`, `train.py` (replaced by ms-swift). `instruct_collator_interleave.py` differs from `InstructDataCollator` only in feeding `context_text` as a text context via `form_mm_prompt` (kept in `prompt_utils_orig.py`); `text_summarizer_collator.py` and `eval_dynamic_superb.py` belong to the Dynamic-SUPERB eval, to be ported with the eval script.

## Local patches to vendored files

| File | Change | Why |
|---|---|---|
| `omni_mds/metrics/wer.py` | `compute_measures` falls back to `jiwer.process_words` (same S/D/I/H counts) | The container has jiwer 4.x, which removed `compute_measures` |
