"""Write mixes/mv4_v0.yaml: the first MERaLiON-4 (Qwen3-Omni LoRA) training mix. ASR, ST, emotion, speech QA.

    python tools/make_mix_mv4_v0.py            # needs read access to the data; prints the per-task budget

Source of the dataset list and paths: the MERaLiON-3 mix
multimodal_trainer/config/dataset/meralion2_ctm_2611_with_new_data_with_sampling_v4.9.yaml (v4.7 = v4.9 minus 2 AQA sets and
smaller `choose` on a few SQA sets). Paths are relative to /data/projects/13003558; old-container paths `/root/scratch/data/` are
mapped to `/scratch/users/astar/ares/lewiswon/data/`. Datasets whose shard files are gone are left out and listed in
docs/mix_mv4_v0.md (most SQA sets and all AC sets, 2026-10-02).

Budget per epoch (samples): `choose` = samples drawn per epoch (without replacement when <= size), `repeat` = multiple of the size.
Weights follow the user's brief (2026-10-02): improve ST, speech QA and emotion while keeping ASR (a little ASR loss is
acceptable); ASR mostly Singapore + Southeast Asian languages, very little public English (LibriSpeech etc.). Stage 1 of a
planned 2-stage schedule (docs/mix_mv4_v0.md).
"""
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OLD = '/home/users/astar/ares/sailorhb/scratch/git_repos/multimodal_trainer/config/dataset/' \
      'meralion2_ctm_2611_with_new_data_with_sampling_v4.9.yaml'
BASE = '/data/projects/13003558'
REMAP = ('/root/scratch/data/', '/scratch/users/astar/ares/lewiswon/data/')
MDS = 'zoux/datasets/datasets_mosaic_stage_AudioLLM_v2.1/datasets_multimodal'
EXTRA = {   # not in v4.9, or moved (these replace the v4.9 entry of the same name)
    'PQA_iemocap_30_ER': (f'{MDS}/train/PQA/iemocap_30_ER', 'PQA'),
    'mixing_tamil_eng_codeswitch_30_ASR': ('/scratch/users/astar/ares/sailorhb/datasets_v3/mixing_tamil_eng_codeswitch_30_ASR', 'ASR'),
}

# ------------------------------------------------------------------ train: (name in v4.9 without the task prefix) -> choose | ('repeat', r)
# None = all samples once.
ALL = None
ASR_SG = {   # Singapore: IMDA NSC (English, code-switch), SEAME, local code-switch sets, SG telephony
    'IMDA_PART1_mono_en_30_ASR': 60_000, 'IMDA_PART2_mono_en_30_ASR': 60_000, 'IMDA_PART3_mono_en_30_ASR': 50_000,
    'IMDA_PART3_conv_en_30_ASR': 60_000, 'IMDA_PART4_conv_codeswitch_30_ASR': 60_000, 'IMDA_PART4_mono_codeswitch_30_ASR': 60_000,
    'IMDA_PART5_conv_en_30_ASR': 60_000, 'IMDA_PART5_mono_en_30_ASR': 30_000, 'IMDA_PART6_conv_en_30_ASR': 60_000,
    'IMDA_PART6_mono_en_30_ASR': 30_000,
    'SEAME_conversation_codeswitch_30_ASR': ALL, 'SEAME_interview_codeswitch_30_ASR': ALL,
    'Bilingual_Mandarin_and_English_HuiTing_ePR9004473_codeswitch_30_ASR': 30_000,
    'ch_eng_code_switching_Ascend_codeswitch_30_ASR': ALL,
    'TelephonyNetworkText_T023_2014_SG_en_30_ASR': 25_000, 'Fhios_Data_en_30_ASR': 20_000,
    'child-speech-chinese-english_en_30_ASR': 20_000,
    'en_zh_2_speakers': ALL, 'en_vi_2_speakers': ALL, 'en_ms_2_speakers': ALL, 'en_th_2_speakers': ALL, 'en_ta_2_speakers': ALL,
    'random_speakers_codeswitch': 35_000, 'mixing_tamil_eng_codeswitch_30_ASR': ALL,
}
ASR_SEA = {  # Singapore's other official languages (Malay, Tamil, Mandarin), dialects, and Southeast Asian languages
    # Malay
    'king223_v2_ms_30_ASR': 60_000, 'MOBI_Malay_ms_30_ASR': 36_000, 'telephoney_speech_ms_MY_4700145868_ms_30_ASR': 24_000,
    'malay_speech_data_mobile_ms_30_ASR': 18_000, 'malay-conversational-speech-corpus_ms_30_ASR': ('repeat', 2),
    'fleurs_malay_ms_30_ASR': ('repeat', 2), 'MagicHub_ms_30_ASR': ALL, 'smaldusc_ms_30_ASR': ALL,
    'sea_speech_and_annotation_4700161249_ms_30_ASR': ALL,
    # Tamil
    'ASR_Tamil_ta_30_ASR': 36_000, 'shrutilipi_tamil_ta_30_ASR': 30_000, 'i2r_tamil_phase_I_ta_30_ASR': 24_000,
    'i2r_tamil_phase_II_ta_30_ASR': ALL, 'i2r_tamil_phase_III_ta_30_ASR': ALL, 'mile_tamil_asr_corpus_new_ta_30_ASR': 18_000,
    'microsoftspeechcorpus_ta_30_ASR': 12_000, 'tamil_speech_data_4700024041and4700024043_ta_30_ASR': ALL,
    'willdata_PO4700201205_IVR_ta_30_ASR': 12_000, 'willdata_PO4700201205_SM_ta_30_ASR': 12_000,
    'fleurs_tamil_ta_30_ASR': ('repeat', 2), 'google_resource_crowdsourced_ta_30_ASR': ALL,
    # Mandarin
    'chinese_asr_new_zh_30_ASR': 24_000, 'wenetspeech_zh_30_ASR': 24_000, 'AIShell_zh_30_ASR': 12_000,
    'King-ASR-118_zh_30_ASR': 12_000, 'Corpus_chinese_zh_30_ASR': 10_000, 'chinese_callcenter_datatang_150hrs_zh_30_ASR': 10_000,
    'chinese_callcenter_datatang_500hrs_zh_30_ASR': 10_000, 'commonvoice_chinese_zh_30_ASR': 5_000,
    'sea_speech_and_annotation_4700161249_zh_30_ASR': ALL,
    # Hokkien, Cantonese
    'MDT2020S002MinnanDialectScriptedSpeechCorpus_hok_30_ASR': 15_000,
    'MDT2020S033MinnanDialectScriptedSpeechCorpus_hok_30_ASR': 15_000, 'King-ASR-427_hok_30_ASR': 10_000,
    'King-ASR-423-1_yue_30_ASR': 10_000, 'King-ASR-423-2_yue_30_ASR': 10_000,
    # Indonesian
    'gigaspeech2_id_30_ASR': 24_000, 'King-ASR-224_id_30_ASR': 18_000, 'indon_speech_data_PO4700081623_id_30_ASR': 15_000,
    'telephoney_speech_id_ID_4700145868_id_30_ASR': 12_000, 'sea_speech_and_annotation_4700161249_id_30_ASR': ALL,
    # Thai
    'gigaspeech2_th_30_ASR': 24_000, 'King-ASR-226_th_30_ASR': 18_000, 'telephoney_speech_th_TH_4700145868_th_30_ASR': 12_000,
    'LOTUS_Thai_th_30_ASR': ALL, 'sea_speech_and_annotation_4700161249_th_30_ASR': ALL,
    # Vietnamese
    'gigaspeech2_vi_30_ASR': 24_000, 'vietnam_speech_100hours_4700008645_vi_30_ASR': 15_000,
    'telephoney_speech_vi_VN_4700145868_vi_30_ASR': 12_000, 'willdata_PO4700201205_IVR_vi_30_ASR': 10_000,
    'sea_speech_and_annotation_4700161249_vi_30_ASR': 10_000,
    # Tagalog, Burmese, Iban, Javanese
    'Tagalog_Speech_PO4700081543_tl_30_ASR': 21_000, 'sea_speech_and_annotation_4700161249_tl_30_ASR': ALL,
    'burmese_speech_dataset_my_30_ASR': ALL, 'iban_iba_30_ASR': ALL, 'willdata_PO4700201205_SM_jv_30_ASR': 10_000,
}
ASR_PUBLIC = {  # common public / generic English: very small weight (user)
    'librispeech_clean100_en_30_ASR': 3_000, 'librispeech_clean360_en_30_ASR': 3_000, 'librispeech_clean500_en_30_ASR': 3_000,
    'common_voice_17_en_30_ASR': 10_000, 'gigaspeech_en_30_ASR': 15_000, 'peoples_speech_en_30_ASR': 10_000,
    'accented_english_chinese_PO4700184169_en_30_ASR': 10_000, 'accented_english_indian_PO4700184169_en_30_ASR': 10_000,
    'accented_english_vietnamese_PO4700184168_en_30_ASR': 10_000, 'King-ASR-136_en_30_ASR': 5_000,
    'King-ASR-139_en_30_ASR': 5_000, 'MOBI_English_en_30_ASR': 10_000, 'no_speech_unk_30_ASR': ALL,
}
# Left out on purpose: commonvoice_tamil_ipa (IPA transcripts), genshin_*/arknights (game voices), SM_ne (Nepali),
# cantonese-youtube-batch3 and tamil_cs_conv_v2 (no shard files).
ST = {
    'CoVoST2_en_id_30_ST': ALL, 'CoVoST2_en_zh-CN_30_ST': ALL, 'CoVoST2_en_ta_30_ST': ALL,
    'gigaspeech_en_zh_30_ST': 200_000, 'peoples_speech_en_ms_30_ST': 200_000,
    'CoVoST2_zh-CN_en_30_ST': ('repeat', 4), 'CoVoST2_id_en_30_ST': ('repeat', 4), 'CoVoST2_ta_en_30_ST': ('repeat', 4),
}
EMOTION = {  # emotion / sentiment recognition (PQA task in the data); small sets, repeated
    'MELD_30_ER': ('repeat', 4), 'MELD_30_SR': ('repeat', 4), 'iemocap_30_ER': ('repeat', 4),
}
SPEECH_QA = {  # questions about the spoken content (CPQA), dialogue summaries (SDS), questions about sounds (AQA),
              # questions about the speaker's accent / gender (PQA AR / GR; IMDA = Singapore speakers)
    'ytb_qq_CPQA': ('repeat', 3),
    'IMDA_PART3_conv_30_SDS': ALL, 'IMDA_PART4_conv_30_SDS': ALL, 'IMDA_PART5_conv_30_SDS': ALL, 'IMDA_PART6_conv_30_SDS': ALL,
    'WavCaps_30_AQA': 150_000, 'AudioCaps_30_AQA': ALL, 'clotho_30_AQA': ('repeat', 2),
    'IMDA_PART3_conv_30_AR': 15_000, 'IMDA_PART4_conv_30_AR': 15_000, 'IMDA_PART5_conv_30_AR': 15_000,
    'IMDA_PART2_mono_30_AR': 15_000, 'VoxCeleb1_30_AR': 10_000,
    'IMDA_PART3_conv_30_GR': 10_000, 'IMDA_PART4_conv_30_GR': 10_000, 'IMDA_PART5_conv_30_GR': 10_000, 'VoxCeleb1_30_GR': 10_000,
}
# Emotional YouTube (Malaysia / Singapore) sets from wangq2 (user, 2026-10-02): CPQA = contextual paralinguistic QA,
# CPSUMMARY = summaries of the speech and how it is said, PQA = emotion / paralinguistic QA. en / ms / ta / zh, 30 s and 60 s
# clips. Held-out dev splits exist and are used for validation. (Their AC/yt_sea_hf captions are not used in this run.)
WANGQ2 = '/scratch/users/astar/ares/wangq2/datasets_multimodal_mosaic'
WANGQ2_REPEAT = 2.0


def wangq2_specs(split):
    out = []
    for task in ('CPQA', 'CPSUMMARY', 'PQA'):
        root = os.path.join(WANGQ2, split, task)
        for d in sorted(os.listdir(root)):
            if os.path.isfile(os.path.join(root, d, 'index.json')):
                out.append((f'{task}_{d}', os.path.join(root, d), task))
    return out


GROUPS = [('ASR Singapore', 'ASR', ASR_SG), ('ASR SEA + SG languages', 'ASR', ASR_SEA), ('ASR public', 'ASR', ASR_PUBLIC),
          ('ST', 'ST', ST), ('Emotion', 'PQA', EMOTION), ('Speech QA', None, SPEECH_QA),
          ('Emotional YTB CPQA/CPSUMMARY/PQA', None, 'wangq2')]

# ------------------------------------------------------------------ validation (all from the v4.9 validation list; size set by --val_choose)
VAL = ['ASR_IMDA_PART1_mono_en_30_ASR', 'ASR_IMDA_PART2_mono_en_30_ASR', 'ASR_IMDA_PART3_conv_en_30_ASR',
       'ASR_IMDA_PART4_conv_codeswitch_30_ASR', 'ASR_SEAME_conversation_codeswitch_30_ASR', 'ASR_SEAME_interview_codeswitch_30_ASR',
       'ASR_cna_en_30_ASR', 'ASR_parliament_short_en_30_ASR', 'ASR_mediacorp_short_en_30_ASR', 'ASR_idpc_short_en_30_ASR',
       'ASR_ytb_batch1_en_30_ASR', 'ASR_ytb_batch2_en_30_ASR', 'ASR_singlish_name_sentence_en_30_ASR',
       'ASR_ytb_asr_batch3_malay_en_30_ASR', 'ASR_ytb_asr_batch3_tamil_en_30_ASR',
       'ASR_malay_conversational_speech_corpus_ms_30_ASR', 'ASR_microsoftspeechcorpus_ta_30_ASR', 'ASR_shrutilipi_tamil_ta_30_ASR',
       'ASR_common_voice_17_ta_30_ASR', 'ASR_AIShell_zh_30_ASR', 'ASR_chinese_asr_new_zh_30_ASR',
       'ASR_gigaspeech2_id_30_ASR', 'ASR_gigaspeech2_th_30_ASR', 'ASR_gigaspeech2_vi_30_ASR', 'ASR_common_voice_17_id_30_ASR',
       'ASR_common_voice_17_th_30_ASR', 'ASR_common_voice_17_vi_30_ASR',
       'ASR_librispeech_clean_en_30_ASR', 'ASR_librispeech_other_en_30_ASR', 'ASR_common_voice_17_en_30_ASR',
       'en_ms_2_speakers', 'en_ta_2_speakers', 'en_zh_2_speakers',
       'ST_CoVoST2_en_id_30_ST', 'ST_CoVoST2_en_zh-CN_30_ST', 'ST_CoVoST2_en_ta_30_ST', 'ST_CoVoST2_id_en_30_ST',
       'ST_CoVoST2_zh-CN_en_30_ST', 'ST_CoVoST2_ta_en_30_ST',
       'SQA_public_speech_sg_test_30_SQA', 'SQA_cn_colledge_entrance_english_test_30_SQA']


def resolve(p):
    if p.startswith(REMAP[0]):
        p = REMAP[1] + p[len(REMAP[0]):]
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def size(p):
    with open(os.path.join(p, 'index.json')) as f:
        return sum(s['samples'] for s in json.load(f)['shards'])


def has_shards(p):
    with open(os.path.join(p, 'index.json')) as f:
        sh = json.load(f)['shards']
    return all(any(os.path.exists(os.path.join(p, (s.get(k) or {}).get('basename', '#'))) for k in ('zip_data', 'raw_data'))
               for s in (sh[0], sh[len(sh) // 2], sh[-1]))


def main():
    with open(OLD) as f:
        old = yaml.safe_load(f)
    by_name = {}
    for split in ('train', 'validation'):
        for e in old[split]:
            by_name[(split, e['name'])] = (resolve(e['path']), e['task'])
    for n, (p, t) in EXTRA.items():
        by_name[('train', n)] = (resolve(p), t)

    def find(short):
        for key in (short, *(f'{t}_{short}' for t in ('ASR', 'ST', 'PQA', 'CPQA', 'SDS', 'AQA'))):
            if ('train', key) in by_name:
                return key, *by_name[('train', key)]
        sys.exit(f'not in v4.9 train: {short}')

    train, budget, total = [], [], 0
    for label, _, table in GROUPS:
        n_group = 0
        items = ([(n, ('repeat', WANGQ2_REPEAT), (n, p, t)) for n, p, t in wangq2_specs('train')] if table == 'wangq2'
                 else [(short, rule, None) for short, rule in table.items()])
        for short, rule, direct in items:
            name, path, task = direct or find(short)
            if not has_shards(path):
                sys.exit(f'{name}: no shard files at {path}')
            n = size(path)
            e = {'name': name, 'path': path, 'task': task}
            if isinstance(rule, tuple):
                e['repeat'] = float(rule[1]); per_epoch = int(n * rule[1])
            elif rule is not None and rule < n:
                e['choose'] = rule; per_epoch = rule
            else:
                per_epoch = n
            e['_comment'] = f'{label}: {n:,} samples, {per_epoch:,} per epoch'
            train.append(e); n_group += per_epoch
        budget.append((label, n_group)); total += n_group

    val = []
    for name in VAL:
        path, task = by_name[('validation', name)]
        if not has_shards(path):
            sys.exit(f'validation {name}: no shard files at {path}')
        val.append({'name': name, 'path': path, 'task': task, 'weight': 1.0, '_comment': f'{size(path):,} samples'})
    for name, path, task in wangq2_specs('dev'):
        if not has_shards(path):
            sys.exit(f'validation {name}: no shard files at {path}')
        val.append({'name': name, 'path': path, 'task': task, 'weight': 1.0, '_comment': f'{size(path):,} samples (held-out dev)'})

    lines = ['# MERaLiON-4 first LoRA run (Qwen3-Omni): ASR, ST, emotion, speech QA. GENERATED by tools/make_mix_mv4_v0.py,',
             '# edit that script, not this file. Sources, left-out datasets and the reasoning: docs/mix_mv4_v0.md.',
             f'# Samples per epoch: {total:,} = ' + ', '.join(f'{l} {n:,} ({100 * n / total:.1f}%)' for l, n in budget),
             '# Validation: the old v4.9 validation sets that still have data; size per set from --val_choose.', '']
    for split, rows in (('train', train), ('validation', val)):
        lines.append(f'{split}:')
        for e in rows:
            lines.append(f'  # {e.pop("_comment")}')
            lines.append(f'  - name: {e["name"]}')
            for k in ('path', 'task', 'choose', 'repeat', 'weight'):
                if k in e:
                    lines.append(f'    {k}: {e[k]}')
        lines.append('')
    out = os.path.join(REPO, 'mixes', 'mv4_v0.yaml')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f'wrote {out}: train {len(train)} datasets, {total:,} samples/epoch; validation {len(val)} datasets')
    for label, n in budget:
        print(f'  {label:25s} {n:>10,d}  {100 * n / total:5.1f}%')


if __name__ == '__main__':
    main()
