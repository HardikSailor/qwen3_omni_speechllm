# MERaLiON-4: how to push the key areas (draft, 2026-10-02)

Target areas (user): speech translation, spoken QA (incl. CPQA), speech summarisation, emotion understanding, and above all
**cultural understanding and reasoning** with speech, for Singapore and Southeast Asia, while keeping ASR (Singapore languages
incl. Singlish, and SEA languages) close to its current level. Stage 1 (`docs/mix_mv4_v0.md`) is a broad multi-task LoRA run;
this document is the plan for what comes after it. Items are proposals to discuss, not decisions.

## 0. First: measure (needed before any stage 2 choice)

Eval: `eval_omni.py` + `pbs/eval_omni.pbs` (AudioBench-style suite, Llama-3-70B judge), added 2026-10-02; see `docs/EVAL.md`.
- **Per-task, per-language scores** for the base model and every stage-1 checkpoint (every 1,000 steps): WER / MER for ASR
  (IMDA parts, SEAME, Malay, Tamil, Mandarin, id / th / vi), BLEU / chrF / COMET for ST (CoVoST2 6 directions), an LLM judge
  (rubric 1-5) for CPQA / CPSUMMARY / SQA, accuracy / macro-F1 for emotion (wangq2 PQA dev, MELD), and the public
  benchmarks the team already uses (AudioBench, SEA-AudioBench in `/data/projects/13003558/sea_audiobench_datasets`,
  `dynamic_superb_hf`). Old metric code is vendored in `omni_mds/metrics/`.
- Implementation: vLLM on a merged copy per checkpoint (D22: `swift export --merge_lora`, TP=2 on H100), or HF + PEFT for a
  quick in-training check. One PBS job per checkpoint on a single node of the reserved queue.
- **Track the base model's general ability too** (text-only instructions, refusal, English reasoning): LoRA fine-tunes of
  strong base models often lose instruction following. A small text-only replay set protects it (see 5).

## 1. Speech translation

- Data now: CoVoST2 en<->id/zh/ta, GigaSpeech en->zh, People's Speech en->ms (machine-translated targets, numbers spelled out).
  Missing: en->vi, en->th, ms/vi/th/tl->en, Singlish->Mandarin/Malay, and any human-quality SEA references.
- **Cascade-to-E2E distillation:** transcribe our SEA ASR sets (gold transcripts already exist) and translate the transcripts
  with a strong text LLM (SEA-LION / Gemma 3 / Qwen3 text) into en / zh / ms / ta / id / th / vi. That turns ~70 M ASR samples
  into ST pairs in every direction we care about, with real local accents. Filter with back-translation + COMET-QE.
- **Chain-of-thought ST** as an auxiliary format: "transcribe, then translate" in one answer, at a small weight. It usually helps
  low-resource directions and keeps ASR alive.
- Style / register: Singlish and code-switched speech translated into standard English / Mandarin / Malay (needs a small
  human-checked set for evaluation).

## 2. Spoken QA (incl. CPQA) and speech summarisation

- Restore or regenerate the MERaLiON-3 SQA sets (their shards were deleted from lewiswon / suns1 scratch; ~127 M samples incl.
  `mds_qaed_cleaned_reasoning` and `mds_qa_sea_synthesized`). Ask the owners first; otherwise regenerate (below).
- **Generation pipeline** (text LLM over gold transcripts + metadata): for each long-form clip (IMDA conversations, SG / MY
  YouTube, parliament, CNA, podcasts) produce (a) extractive QA, (b) inferential / multi-hop QA, (c) MCQ with distractors,
  (d) summaries at 3 lengths and in 2-3 languages, (e) "who said what" / speaker-attributed questions. Answers must be checked
  against the transcript (LLM-as-judge + string checks); drop questions answerable without the audio.
- **Longer context:** CPQA_60 and CPSUMMARY_60 exist (60 s clips). Qwen3-Omni handles long audio; add 2-5 min clips
  (audio ~13 tokens/s, so 5 min = ~3.9 k tokens: raise `max_length` to 6-8 k for those sets only, or a separate stage).
- Multi-turn spoken dialogue QA (follow-up questions about the same clip) to teach grounding.

## 3. Emotion understanding

- Now: MELD ER / SR, IEMOCAP ER (29 k unique, English acted / TV), wangq2 Emotional-YTB PQA / CPQA / CPSUMMARY (MY / SG,
  en / ms / ta / zh). The SEA part is the valuable one; the rest is small and Western.
- **Label harmonisation:** map all sets to one emotion taxonomy (e.g. 7 classes + valence / arousal) and ask in several forms:
  classification, open description, "why do you think so" (evidence from tone vs. words), and contrast questions
  ("the words are positive but the tone is ..."). This is where paralinguistic understanding beats a text model.
- **Pseudo-labels at scale:** run emotion2vec+ (already in `hardik/analytics/emotion_evaluation`) and a text sentiment model
  over IMDA / YouTube speech; keep segments where audio and text models agree with high confidence, and also keep
  "disagreement" cases as sarcasm / mismatch questions with human review.
- Evaluate on the wangq2 PQA dev sets, the SG YouTube 480 eval set (`hardik/SG_YouTube_evalset_en_480`), MELD test.

## 4. Cultural understanding and reasoning (the main gap)

The current mix teaches perception (what was said, how) but almost no **local knowledge or reasoning about it**. Proposals:
- **Singapore / SEA cultural knowledge QA over speech:** questions whose answer needs local knowledge plus the audio: food
  (hawker dishes, "kopi-o kosong"), places (MRT lines, HDB estates), institutions (CPF, NS, MOE, PSLE), festivals (Hari Raya,
  Deepavali, CNY, Vesak), etiquette, Singlish particles and their pragmatics ("lah", "leh", "meh", "sia"), Hokkien / Malay /
  Tamil loanwords in Singlish. Source: generate spoken questions with TTS in local accents (IMDA-style voices; MERaLiON TTS
  if available), or attach knowledge questions to real clips that mention these topics (find them via ASR transcripts).
- **Pragmatics and implied meaning:** indirect requests, politeness, sarcasm, code-switching as a social signal (why did the
  speaker switch to Mandarin here?). Built from IMDA conversations + LLM annotation + human review.
- **Reasoning:** keep the "think then answer" format for a share of QA (MERaLiON-3 had `mds_qaed_cleaned_reasoning`);
  Qwen3-Omni already has a thinking mode in its text model, so training short rationales in the answer (or a `<think>` block)
  for complex questions helps; reward-based tuning (GRPO / DPO with an LLM judge on cultural correctness) after SFT.
- **Text-only cultural instruction data** (SEA-LION style, SG-specific instruction sets) mixed in at a small weight: keeps
  the knowledge in the LLM and supports the speech tasks.
- Evaluation: build a held-out Singapore cultural speech QA set (~1-2 k items, human-checked) early, so every stage is scored
  on it. SEA-HELM / SEA-AudioBench style tasks where available.

## 5. Training plan

- **Stage 1 (running, `mv4_lora_v0`):** broad multi-task LoRA, 1 epoch, 4.7 M samples, ASR 36% / ST 24% / QA + summary +
  emotion 40%.
- **Stage 2 (proposal):** start from the best stage-1 adapter (or merge it and start a new adapter), LR ~5e-5, warmup again,
  1 epoch of a mix with: new ST pairs (1), regenerated / restored SQA and summaries (2), harmonised emotion data (3), cultural
  QA + reasoning (4), plus **replay**: ~15-20% ASR (SG + SEA) and ~5% text-only instructions to hold ASR and instruction
  following. Same eval suite after every 1,000 steps.
- **Stage 3 (optional):** preference tuning (DPO / GRPO) on cultural QA and summarisation with an LLM judge, small LR.
- **Model capacity:** if LoRA on attention saturates, try (a) higher rank on the audio encoder and LLM attention, (b) LoRA on
  the MoE experts through Megatron-SWIFT (expert LoRA; EP=8 is already tested here), (c) full fine-tuning of the audio
  encoder (as MERaLiON-3 did) with LoRA on the LLM.
- **Hygiene:** the eval sets must not overlap training (CoVoST2 test was used as validation in the old mixes; keep it eval
  only); dedupe generated QA against eval sources; keep every mix generated by a script (as `tools/make_mix_mv4_v0.py`).
