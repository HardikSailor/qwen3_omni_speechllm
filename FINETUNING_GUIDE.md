# Adapting Qwen3-Omni-30B-A3B for multi-task audio understanding in Southeast Asia and Singapore

Tasks in scope: **ASR, speech translation (ST), speech emotion recognition (SER), spoken QA (SQA), acoustic event classification/captioning (AEC)**, and similar tasks.

Target setting: **Southeast Asian and Singapore languages**, including code-switching and local cultural knowledge. That means:
- Singapore English and Singlish; Mandarin; Malay; Tamil; English–Mandarin, English–Malay and similar code-switching; Hokkien, Cantonese and Malay loanwords;
- regional languages: Indonesian, Thai, Vietnamese, Filipino, Burmese, Khmer, Lao.

This guide covers four things:
1. a solid multi-task fine-tuning baseline;
2. your two-stage / two-track plan, with refinements;
3. what changes for SEA languages, code-switching and cultural grounding (§4, which is referenced throughout);
4. further ideas on architecture, training objectives, RL and inference.

The roadmap at the end (§11) turns them into a sequence of experiments.

Local context this guide assumes (see the scripts in this folder):
- Env: `venvs/swift_omni` (ms-swift over nemo_env). Model: `Qwen3-Omni-30B-A3B-Instruct`. Run with `ENABLE_AUDIO_OUTPUT=0`, so the talker and code2wav are not loaded.
- Smoke-test cost on 1x H200: LoRA r16 on thinker attention plus audio `proj1`/`proj2` (13.4M params) used 61.6 GiB peak memory and took 4.4 s/step (bs 1, grad-accum 4).
- Two lessons already learned:
  - Target style gets copied. Raw LibriSpeech uppercase text made WER *worse* (2.98 → 3.97).
  - PEFT regex targets leak into the audio encoder. Use explicit module lists (`lora_targets_*.txt`).

---

## 0. The model, as seen by a fine-tuner

The numbers below come from the local snapshot's `config.json`, the safetensors headers (parameter counts), `preprocessor_config.json`, and `_get_feat_extract_output_lengths` in the transformers code.

```
audio 16 kHz, 128-bin mel, hop 160 (100 frames/s)
  └─► AuT audio encoder: 32 layers, d_model 1280, 20 heads, 0.65B params
        3 stride-2 convs per 100-frame block → 13 audio tokens per second
        variable length (no Whisper-style 30 s padding); self-attention is block-local over 8 s windows (104 tokens)
        └─ proj1/proj2 → 2048-d ─┐
                                 ├─► Thinker MoE LLM: 48 layers, hidden 2048, 32 q-heads / 4 kv-heads (GQA),
text prompt ─────────────────────┘     128 routed experts, top-8, NO shared expert,
                                        1.54B non-expert + 29.0B expert params ≈ 30.5B total, ≈3.3B active/token
                                        router_aux_loss_coef = 0.001 (config default)
(talker 3.3B + code2wav 0.2B, vision 0.54B: not used for these tasks)
```

What this means for fine-tuning:
- **The encoder is strong at speech in its supported languages.** Per the model card, speech input officially covers 19 languages. Relevant to you: **English, Chinese (Mandarin), Cantonese, Malay, Indonesian, Vietnamese**. **Tamil, Thai, Filipino, Burmese, Khmer, Lao, Hokkien and Teochew are NOT in the supported list.** For those languages, test whether the failure is acoustic before spending on encoder adaptation; the official list alone does not establish the bottleneck.
- **For supported languages**, most of the headroom for ASR, ST and SQA is on the **LLM side**: output style, Singlish and local vocabulary, code-switch fidelity, instruction following. For AEC and SER, the representation itself often limits accuracy, which puts the limit on the **encoder side**.
- **The thinker is a pure MoE.** Every MLP is a routed expert, and there is no shared expert or dense MLP to put cheap LoRA on. Attention LoRA is cheap and safe. Expert LoRA over all 48 × 128 experts is expensive, and tuning the router is risky. See §5.4.
- **Output is free-form text.** Classification tasks (SER, AEC) should therefore use a *fixed, canonical label vocabulary*, or you pay for label paraphrasing at eval time. Better still, use log-likelihood scoring (§7.1).

---

## 1. Before training: baseline and data hygiene (do not skip)

1. **Zero-shot eval of the base model on every task, language and test set**, with 3–5 prompt variants each. Break results down by language and by monolingual vs. code-switched speech. The gap between supported (ms/id/vi/zh) and unsupported (ta/th/fil/my) languages tells you where the budget should go. If a slice is already at ceiling (clean US English ASR, for example), use it only as replay data so it does not regress.
2. **Normalise targets to the style you want the model to produce**, not to the style of the corpus:
   - ASR: natural casing and punctuation. Score with a normaliser. For code-switching and scripts, see §4.3 for conventions.
   - ST: consistent punctuation and script, one reference style per language.
   - SER / AEC: one canonical label string per class (`"happy"`, never `"Happy"`, `"happiness"` or `"joy"` mixed together). Put the label set in the prompt so the model knows the closed set.
   - AEC multi-label: a deterministic order (alphabetical or by onset) and a fixed separator (`"dog_bark; siren"`).
   - SQA: short answers when the benchmark uses exact match or F1. Longer answers only if an LLM judge scores them.
3. **Prompt diversity.** Use 10–30 paraphrased instructions per task, sampled at random. A single fixed prompt makes the model brittle to prompt changes at test time, and it lets the model infer the task from the audio instead of the instruction. That causes cross-task leakage, such as transcribing when asked for emotion. Also write **some prompts in the local languages** (Chinese, Malay, Tamil, Indonesian, and so on), not only English, because users will prompt that way.
4. **Deduplicate and check leakage** between train and test. Split by speaker, not by utterance. This matters especially for SER corpora and for NSC, where the same speakers read many prompts.
5. **Track loss per task and per language.** Add a `"channel"` field to each JSONL row and pass `--enable_channel_loss true`, so ms-swift logs loss per channel. Use `task` or `task:lang` as the channel value, e.g. `asr:ta`, `asr:cs-en-zh`, `ser:en`.

JSONL row format (the one `make_swift_asr_data.py` already uses), extended:
```json
{"id": "seame_0001", "channel": "asr:cs-en-zh",
 "messages": [{"role": "user", "content": "<audio>Transcribe this audio. The speaker may mix English and Mandarin; keep each word in its original language."},
              {"role": "assistant", "content": "我今天 very tired 啦，不想 go out 了。"}],
 "audios": ["/path/to/a.wav"]}
```

---

## 2. Baseline: direct multi-task LoRA SFT

This is the reference point that every other idea must beat.

**What to train**
- LoRA r=16–64 on thinker attention (q/k/v/o) plus `audio_tower.proj1/proj2`. This is the current `lora_targets_thinker_attn_audio_proj.txt`.
- **For languages outside the official speech list (Tamil, Thai, Filipino, …), compare attention/projector tuning with encoder adaptation** (top AuT layers, §5.1). If the base model fails on phonetic contrasts even with correct output conventions, encoder-side updates become a stronger candidate.
- Add `lm_head` only if you introduce new tokens. Avoid expert LoRA in the baseline.

**Data mixing: balance both tasks and languages**
- **Temperature sampling** over (task × language) buckets: sample bucket *b* with probability ∝ n_b^(1/T), with T≈2–3 (T≈3–5 for languages, to lift low-resource ones like Burmese and Khmer). Otherwise the largest buckets (usually English and Mandarin ASR) dominate. ms-swift supports `--dataset a.jsonl#N` to sub-sample or over-sample each file to *N* rows, so compute N per file from the formula.
- **Replay/regularisation (5–15 %)**:
  - general audio instructions and text-only chat produced *by the base model itself* (self-distillation);
  - **text-only SEA/Singapore cultural QA** (§4.5).

  This keeps instruction following and SQA reasoning intact.

**Loss balance**
- Classification targets are 1–3 tokens and ASR/ST targets are 20–200 tokens. With token-level averaging, long-target tasks dominate the gradient.
- **The tokenizer adds a language bias on top of this.** Measured with this model's tokenizer, the same one-sentence question takes:
  - **en 13** tokens, **zh 9**, **vi 19**, **ms/id 20**, **fil 28**;
  - **th 32**, **ta 63**, **my (Burmese) 76**.

  Tamil and Burmese targets are therefore 5–6× longer than English for the same content. With token averaging they dominate the loss per example. They also decode 5–6× slower and are more prone to runaway repetition.
- Options:
  - (a) Use per-sample loss averaging, which neutralises both the task and the script bias.
  - (b) Set per-bucket loss weights. Channel loss gives visibility; the weighting itself needs a small custom `loss_scale` or trainer plugin.
  - (c) Oversample the short-target tasks.

**Hyperparameters to start from** (1x H200, LoRA): lr 1e-4 (5e-5 if r≥64), cosine schedule, warmup 3–5 %, 1–3 epochs, effective batch 32–64, `max_length` 2048–4096. Raise `max_length` if you have long Tamil or Burmese targets. Evaluate each task and language every few hundred steps with *generation*, not just val loss.

**Sketch** (extend `train_lora_asr_smoke.sh`):
```bash
$PY/swift sft --model "$MODEL" --model_type qwen3_omni_moe \
  --dataset data/asr_sg_en.jsonl#30000 data/asr_cs.jsonl#25000 data/asr_ms.jsonl#10000 \
            data/asr_ta.jsonl#15000 data/asr_sea.jsonl#20000 data/st.jsonl#25000 \
            data/ser.jsonl#15000 data/sqa.jsonl#15000 data/aec.jsonl#20000 \
            data/replay_general.jsonl#6000 data/replay_culture_text.jsonl#6000 \
  --enable_channel_loss true \
  --tuner_type lora --target_modules $(cat lora_targets_thinker_attn_audio_proj.txt) \
  --lora_rank 32 --lora_alpha 64 --learning_rate 1e-4 --num_train_epochs 2 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
  --gradient_checkpointing true --attn_impl flash_attn --torch_dtype bfloat16 \
  --output_dir outputs/mt_lora_stage1 ...
```

---

## 3. Your plan: joint stage, then two tracks (speech vs. acoustic events)

The plan is reasonable. Speech tasks (ASR/ST/SER/SQA) and non-speech tasks (AEC) pull the model in different directions:
- Speech tasks want **linguistic** features and long, token-heavy outputs.
- AEC wants **spectral/temporal** features, multi-label outputs, and robustness to non-speech input.

Most negative transfer shows up between these two groups. For the SEA setting, add **Stage 0** in front of the plan.

### 3.0 Stage 0: language/acoustic adaptation (new, SEA-specific)
Before multi-task training, run a short, **ASR-heavy stage** on the unsupported and under-represented languages and on code-switched speech:
- data: Tamil, Thai, Filipino, Burmese, Khmer, Singlish, and English–Mandarin/Malay code-switching;
- trainable parts: encoder LoRA + projector + attention LoRA.

ASR is the densest supervision signal for teaching the encoder new phonetics. Every downstream task (ST, SER, SQA in those languages) benefits. Keep 20–30 % supported-language ASR in the mix so the encoder does not drift. Stage 1 then starts from this adapter.

### 3.1 Make stage 2 train different parameters, not just different data
With the same LoRA and different data, stage 2 is plain continued fine-tuning and will partly forget the other track. Instead:
- **Speech track**: continue the LLM-side LoRA (attention and, optionally, a few experts, §5.4). Keep the stage-0 encoder LoRA at a low lr; do not freeze it back if Tamil and Thai rely on it.
- **AEC track**: adapt the **encoder side**: more encoder LoRA or partial unfreeze of the top AuT layers, plus the projector. Use a light LLM LoRA or none. Use `--vit_lr`/`--aligner_lr` for lower encoder learning rates (e.g. 1e-5 encoder, 5e-5 projector, 1e-4 LLM LoRA).

### 3.2 Initialise both tracks from stage 1, and keep replay
Load the stage-1 adapter (`--adapters outputs/mt_lora_stage1/checkpoint-xxx`) and add a **10–20 % replay** of the *other* track plus general data. Otherwise the speech-track model loses the ability to say "there is no speech here", and vice versa. Keep all languages in the speech-track replay, or low-resource languages regress first.

### 3.3 Decide how the two tracks are served
Two adapters mean you need a rule at inference that picks between them. Options, from simplest to most elaborate:
1. **Route by task prompt.** The task is known from the instruction, so load adapter A or B. This is easy with PEFT adapter switching. **Note (2026-09-29):** vLLM 0.17.1 has no LoRA support for Qwen3-Omni, so vLLM multi-LoRA serving is not available; in vLLM this means two merged models (2× GPU memory). See `DEPLOYMENT.md` §2.3. Good for benchmark-style evaluation.
2. **Route by audio content.** A tiny speech/non-speech classifier (a VAD or the stage-1 model itself) picks the adapter.
3. **Merge the adapters back into one model** with TIES / DARE / task-arithmetic merging, tuning the merge weights on dev sets. You usually recover 90–100 % of each specialist's gains in a single model.
4. **LoRA-MoE / mixture of LoRA experts.** Keep both (or more) adapters as experts and learn a gate. See §5.3.

### 3.4 Consider other track splits
- **Emotion** is paralinguistic and behaves more like AEC than ASR: it needs prosody, not words. A three-way split often works better: **{ASR, ST, SQA} / {SER, speaker attributes} / {AEC, captioning}**.
- **A language-family split** is also plausible for the speech track, e.g. {English/Singlish/Chinese/Malay/Indonesian} vs. {Tamil/Thai/Burmese/Khmer}. Unsupported, token-heavy languages can crowd out the rest.
- Decide empirically: fine-tune one adapter per task (and per language group) for a few hundred steps, compute pairwise gradient cosine similarity or transfer gains, and cluster.

### 3.5 Stage 3: unified instruction tuning (optional)
After the tracks, a short, low-lr pass over a small, high-quality **mixed-task instruction set** restores general chat ability and cross-task composition. Examples: "transcribe this Singlish clip and tell me the speaker's attitude"; "what sound interrupts the speaker, and what did they say before it?". This is also where culturally grounded conversational SQA (§4.5) fits best.

---

## 4. Southeast Asia and Singapore specifics

### 4.1 Where the data can come from (check licences for your use)
| Need | Candidate sources |
|---|---|
| Singapore English / Singlish ASR | IMDA **National Speech Corpus (NSC)**: read, conversational and code-switched parts |
| English–Mandarin code-switching | **SEAME** (Singapore/Malaysia conversational), NSC code-switch parts, MERLIon CCS (child-directed) |
| Malay, Tamil, Indonesian, Thai, Vietnamese, Filipino, Burmese, Khmer, Lao | Common Voice, **FLEURS** (all of these languages, parallel across languages, so it also works for ST), GigaSpeech 2 (th/id/vi), in-house data |
| ST | FLEURS (any→any via parallel sentences), CoVoST 2 (id/ta→en), MT pseudo-labels of the ASR sets above (§6.2) |
| Benchmarks / references | **AudioBench** and **MERaLiON** (A*STAR's Singapore-focused AudioLLM work) for tasks, prompts and comparisons; SEA-HELM / SeaEval for text-side cultural knowledge |
| Local AEC | collect or label local classes (MRT door chime and announcements, hawker-centre ambience, void-deck and HDB-corridor sounds, tropical rain and thunder, koel bird, festival drums and firecrackers, lion dance). AudioSet/ESC-50 do not cover these well. |

### 4.2 Unsupported languages (Tamil, Thai, Filipino, Burmese, Khmer, Lao, Hokkien)
- **Test encoder adaptation when acoustic errors persist**: compare the projector/Thinker baseline against LoRA or selective unfreezing of upper AuT layers, with more ASR data than for the other tasks (Stage 0).
- **Tokenizer cost** (measured in §2): Tamil and Burmese use ~5–6× the tokens of English. Options:
  - accept the cost, and use per-sample loss and larger `max_length`;
  - later, **vocabulary extension** with new subword tokens for Tamil, Thai and Burmese. This needs embedding/`lm_head` training on text first and is risky with LoRA-only training, so it is not a first step.
- **Metrics**: use **CER** for Thai, Burmese, Khmer, Lao and Chinese, which have no word spaces or unreliable segmentation. Use WER for Tamil but report CER too, because agglutination inflates WER.
- **Hokkien and Teochew** have no standard written form. Decide the target up front: Hanzi, romanisation (POJ / Tâi-lô), or translation into Mandarin or English (then it is really an ST task). Mixing these conventions in training data is the fastest way to a confused model.

### 4.3 Code-switching
- **Transcription conventions** (document them and apply them everywhere):
  - write each language in its own script: Mandarin in Hanzi (simplified, as in Singapore), English in Latin script;
  - no spaces around Chinese characters, spaces between English words;
  - keep Singlish discourse particles as spoken (*lah, leh, lor, meh, hor, sia, ah*);
  - keep common loanwords in their conventional local spelling (*makan, shiok, paiseh, kiasu, alamak, tabao*).
- **The main failure mode is "language flattening".** The model silently translates the minority-language segment into the matrix language, e.g. writes "go out" as 出去, or turns Singlish into standard English. Mitigations:
  - an explicit prompt ("keep each word in its original language");
  - negative/rejected examples in DPO (§8);
  - a code-switch-fidelity reward term.
- **Metric: mixed error rate (MER)**, i.e. CER on Chinese characters plus WER on Latin-script words. Also track the **error rate around switch points**, where most errors concentrate.
- **Synthetic code-switched data** (code-switched speech is scarce):
  - (a) generate code-switched text with an LLM from monolingual transcripts, respecting matrix-language grammar, then synthesise it with multilingual TTS in local-accented voices;
  - (b) splice real monolingual segments from the same speaker at word boundaries using forced alignments;
  - (c) use the stage-1 model to pseudo-label untranscribed local conversational audio, filtered by agreement (§6.2).
- **Language hints in prompts.** Train with the expected language set sometimes stated and sometimes not ("may contain English, Mandarin and Malay"). The model then uses the hint when available without depending on it.
- **ST from code-switched speech** (e.g. into standard English or Malay) benefits most from the transcript-then-translate CoT format (§6.3).

### 4.4 Singapore / SEA named entities and contextual biasing
Local place names (Toa Payoh, Jurong, Tampines), foods (char kway teow, roti prata), organisations and acronyms (HDB, CPF, NS, MRT, ERP), and personal names across Chinese, Malay, Indian and Eurasian naming conventions are a large share of ASR errors.
For a detailed experiment design covering candidate-list construction, hard distractors, acoustic verification, false-insertion metrics, and further model/training ideas, see `TRAINING_AND_LOCAL_VOCAB_EXPERIMENTS.md`.
- **Prompt biasing**: sweep small, shuffled lists (for example 0, 5, 10 and 20 terms) under a "possible spellings; use only if heard" prompt. Include true phrases, hard distractors, distractor-only lists and no-entity audio so the model learns not to insert terms from the list. At inference, fill a short list from available context such as the domain, meeting agenda or contact list; measure false insertions alongside entity recall.
- A **local lexicon** with canonical spellings and aliases (e.g. "Tekka"/"Tekka Centre") is useful both for normalisation and for synthetic data.

### 4.5 Cultural understanding (SQA, SER, conversation)
The thinker is a text LLM, so **much of the cultural knowledge can be injected through text**. The audio pathway then only needs to ground to it.
- **Text-only cultural replay**: local-knowledge QA and instructions about Singapore and SEA history, festivals (CNY, Hari Raya, Deepavali, Vesak), food, transport, government services, etiquette and local humour. SEA-focused instruction datasets can be sources, as can LLM-generated QA from curated local documents, verified by native speakers. Mix at 5–10 % during all stages.
- **Culturally grounded SQA**:
  - generate questions whose answers need local context (e.g. "Where does the speaker say they are going to *tabao* dinner from?", "What festival is being prepared for?");
  - synthesise the passages as speech in local accents and code-switched style, or record them;
  - have native speakers verify a subset.
  - include **implicature**, which is where cultural understanding shows: Singlish "can lah" (agreement) vs. "can meh?" (doubt); indirect refusals; politeness strategies.
- **SER is culture-dependent.**
  - Emotion display rules and prosodic cues differ across communities, and Singlish particles carry stance and emotion ("wah lau", "sian", "sia"). Western acted corpora (IEMOCAP, RAVDESS) transfer poorly.
  - Prioritise locally annotated data with local annotators, and report agreement (κ).
  - Consider **dimensional labels** (arousal/valence) in addition to categories; they are more robust across cultures.
  - Use text+audio CoT ("particles and wording suggest…, prosody suggests…") for SER on Singlish, where lexical cues matter.
- **Evaluate cultural competence explicitly**: a held-out set of locally written SQA items (with Singlish, code-switching and local entities), scored by exact match plus an LLM judge that is given local reference notes.

---

## 5. Architecture ideas

> **Deployment cost (2026-09-29, `DEPLOYMENT.md` §5):** a partner can run our model with stock vLLM only if the architecture stays identical to the public Qwen3-Omni (LoRA merged into the weights, or full fine-tuning). These ideas add modules stock vLLM doesn't have, so each would need a patched vLLM or custom serving code at every deployment site: layer-weighted features (§5.1), dual encoder (§5.2), MoLE (§5.3), CTC head (§5.5), new soft tokens (§5.7). Top-N encoder LoRA (§5.1) is fine: it merges.

### 5.1 Tune the encoder where it matters (high value for unsupported languages, AEC and SER)
- **LoRA on the top-N AuT layers** (of 32; list module paths explicitly and check them after training, per the PEFT regex pitfall). Use N=8 and r=16 for AEC/SER. For new languages, use N=16 or more, r=32, plus the projector.
  - *Implemented 2026-09-29:* `train_omni.py --audio_lora_layers top8 --audio_lora_modules attn_mlp` (also `all`, `bottom<K>`, `a-b`, `i,j,k`; modules `attn`/`mlp`/`attn_mlp`), for both `sft` and `megatron`.
    - Keep the LLM targets in an explicit `--target_modules` list; `all-linear` is refused.
    - Tested in job 150442 (`PIPELINE_PLAN.md` §10.7). `tests/summarize_smoke.py` prints the layers that got LoRA and whether they trained.
- **Layer-weighted features.** Emotion and event cues are stronger in *middle* encoder layers, while the top layers are ASR-shaped. Add a learnable softmax-weighted sum over encoder hidden states before `proj1` (SUPERB-style). This adds only 33 scalars. Initialise it to a one-hot on the last layer so step 0 equals the base model. It often gives several points on SER and AEC.

### 5.2 Add a complementary audio encoder (dual-encoder, SALMONN-style)
Fuse a specialised encoder alongside AuT:
- **AEC**: BEATs, EAT or a CLAP audio encoder.
- **SER**: emotion2vec+.
- **Languages AuT does not cover**: an MMS or w2v-BERT 2.0 encoder, which were trained on 1000+ languages including Tamil, Thai and Burmese. This can be cheaper than heavy AuT adaptation when data for a language is small.

Fusion options:
- (a) Concatenate the extra encoder's tokens (after a small projector) to the AuT tokens.
- (b) Q-Former / Perceiver pooling to 8–32 tokens per clip. This suits clip-level tasks.
- (c) Gated cross-attention.

Train the new projector first, then the projector plus LoRA. This is the most engineering-heavy idea (it requires modifying the ms-swift template/model code). Do it only if §5.1 is not enough.

### 5.3 Mixture of LoRA experts (MoLE / LoRA-MoE)
Rather than 2 hard tracks, use *K* LoRA experts on attention with a small learned gate, conditioned on (task-prompt embedding, pooled audio embedding). The gate naturally learns splits by task *and* by language. Add a load-balancing loss.

### 5.4 Exploit the MoE thinker (128 experts, top-8, no shared expert)
- **Profile expert routing per task *and per language* first.** Hook the router on each dev set and count activations per layer. Expect language-specific experts (e.g. Chinese vs. Tamil tokens) as well as task-specific ones.
- **Targeted expert LoRA.** Add LoRA only to the top-*k* experts per layer used by the slice you want to improve (e.g. Tamil ASR, code-switched ASR, SER). This is the only cheap way to add MLP capacity, because there is no shared or dense MLP in this model.
- **Keep the router frozen**, or train it with a very low lr and `--router_aux_loss_coef 1e-3` (the config default). An unregularised router tends to collapse onto a few experts after narrow-domain SFT.
- Full expert fine-tuning needs multi-GPU with expert parallelism (Megatron-SWIFT). Treat it as a last resort.

### 5.5 CTC auxiliary head for ASR
Add a small CTC head on the encoder or projector output, trained jointly with the LLM loss (λ≈0.1–0.3), with a character/subword vocabulary covering your scripts. Considerations and possible benefits:
- Check the time-axis constraint before implementing: CTC requires at least as many encoder steps as target units, while the final AuT output is only about 13 tokens/s. An earlier encoder state or coarser target units may be needed for character-heavy scripts.
- It gives the encoder a strong alignment signal. This is especially valuable for new languages, and it reduces hallucination on long or noisy audio.
- At inference, CTC n-best hypotheses can be fed back as hints (§5.6), or used for joint rescoring.
- CTC can provide a more acoustically constrained ASR hypothesis for checking language flattening; measure whether it helps on actual switch-point errors.

### 5.6 Hypothesis-conditioned prompts (generative error correction)
For hard slices (accented, code-switched, noisy, rare entities): put an external ASR's n-best in the prompt (a Whisper/MMS model fine-tuned on NSC and SEAME, or the CTC head). The LLM combines the acoustics, the candidates and its local-entity knowledge.

### 5.7 Task / language / domain soft tokens
Learnable prefix tokens per task and per language or domain. They are cheap and reduce cross-task leakage and language confusion.

---

## 6. Training-objective and data ideas

### 6.1 Augmentation (per task)
| Task | Augmentations |
|---|---|
| ASR / ST / SQA | speed perturb 0.9–1.1, MUSAN noise at 5–20 dB SNR, RIR reverb, **local noise beds** (hawker centre, MRT, traffic, rain), codec/telephony simulation (8 kHz, Opus/MP3) matching deployment |
| SER | mild noise/reverb only. Avoid pitch shift and heavy speed perturbation, which alter emotion cues. |
| AEC | **mixing two clips and taking the union of their labels**, random crop, gain, masking, background-noise mixing at varied SNR |

### 6.2 Synthetic and pseudo-labelled data
- **SQA**: TTS-synthesised passages and questions (multi-speaker, local accents, code-switched style), with answers generated and verified by a text LLM. Also LLM-generated QA from existing local transcripts (NSC conversations, for example).
- **ST**: machine-translate transcripts with a strong MT model, filtered with COMET-QE plus back-translation consistency. For Singlish and code-switched speech, have the MT model translate into *standard* English/Malay/Chinese, and spot-check the results with native speakers.
- **AEC captions**: turn label sets plus timestamps into natural captions with an LLM.
- **Self-training**: pseudo-label unlabelled in-domain audio (local radio/podcasts or call data, where licensing permits). Keep outputs with low token entropy and agreement across prompts/augmentations, then retrain. This is the best way to scale code-switched and low-resource data.

### 6.3 Chain-of-thought / cascaded outputs inside one model
- **ST**: "transcript → translation" in one response. This gives the largest gain for code-switched and low-resource source speech.
- **SQA**: "transcribe the relevant part, then answer".
- **SER**: "note lexical/particle cues and prosody, then give the label". This is more useful for Singlish than for other speech.
- Compare zero-shot against `Qwen3-Omni-30B-A3B-Thinking` on SQA and cultural reasoning before committing to the Instruct model.

### 6.4 Curriculum
- Start with short, clean, monolingual, single-task examples.
- Then move to long, noisy, code-switched, multi-label and compositional ones.
- For new languages: read speech first, then conversational.

### 6.5 Class imbalance
SER and AEC labels are imbalanced. Oversample rare classes (or use class-balanced sampling). Consider label smoothing, and report macro-F1 / UAR.

---

## 7. Inference-time improvements (no retraining)

### 7.1 Score classes instead of generating them (SER, AEC)
For a closed label set, compute log p(label | audio, prompt) for every label and take the argmax (or apply per-class thresholds tuned on dev for multi-label AEC). This removes invalid or paraphrased outputs and gives calibrated scores, so you can report mAP.

### 7.2 Prompt ensembling and test-time augmentation
Average label log-probs over 3–5 prompt paraphrases (including local-language prompts) and 2–3 mild augmentations.

### 7.3 ASR decoding
Use beam search (4–8) with repetition/length guards; these matter more for token-heavy Tamil and Burmese. Add CTC rescoring (§5.5) and contextual biasing lists (§4.4). The processor does not truncate audio (`truncation=False`); it only pads to the longest clip in the batch, and the encoder drops that padding. Very long audio is still worth chunking with overlap, to bound LLM sequence length and hallucination.

---

## 8. Reinforcement learning / preference optimisation

Do RL **after** SFT has converged. RL sharpens and aligns; it does not teach new skills or new languages.

### 8.1 Reward per task
| Task | Reward |
|---|---|
| ASR | `1 − min(ER, 1)`, using **MER for code-switching, CER for zh/th/my/km/lo, WER otherwise**. Add a penalty for length ratio > 1.5 (hallucination). |
| ASR code-switch fidelity | a bonus when the hypothesis's per-segment language/script matches the reference (penalises language flattening) |
| ST | COMET (or chrF++, which is more robust for Tamil and Thai than BLEU) |
| SER | +1 correct, 0 wrong, −0.5 for an invalid/out-of-set label |
| AEC | per-sample F1 between predicted and gold label sets, with an invalid-label penalty |
| SQA | exact match / token-F1, or an LLM-as-judge score with local reference notes for cultural items |
| all | a format reward; a wrong-output-language penalty (e.g. answering in English when asked in Malay) |

### 8.2 Methods, cheapest first
1. **Rejection sampling fine-tuning (RFT)**: sample N=8, keep the best by reward, and run SFT on them. See `examples/train/rft`.
2. **DPO with n-best pairs.** Chosen = best by metric. Rejected = worst, or a *targeted failure*: language-flattened transcript, standard-English "corrected" Singlish, hallucinated continuation, paraphrased label. Use `swift rlhf --rlhf_type dpo` with a `rejected_response` field. This is the most direct fix for code-switch flattening.
3. **GRPO** with the rewards above: `swift rlhf --rlhf_type grpo --external_plugins my_rewards.py --reward_funcs ...`, modelled on `examples/train/grpo/qwen2_5_omni/grpo.sh` and the ORM classes in `examples/train/grpo/plugin/plugin.py`. GRPO pays off most on SQA (cultural reasoning), ST and CoT formats. Rollouts with a 30B MoE through HF generate are slow, so check whether vLLM rollout works for qwen3_omni_moe first, and start with LoRA and `num_generations` 4–8.
4. **MWER-style sequence training for ASR**: minimum-MER over n-best, as a custom loss if GRPO is too slow.

Minimal reward plugin sketch:
```python
# my_rewards.py
import re, jiwer
from swift.rewards import ORM, orms   # check the import path in this ms-swift version (see plugin.py)

CJK = re.compile(r'[一-鿿]')

def mer(ref, hyp):
    """Mixed error rate: CJK chars as tokens, other words as tokens."""
    tok = lambda s: re.findall(r'[一-鿿]|[^\s一-鿿]+', s.lower())
    r, h = ' '.join(tok(ref)), ' '.join(tok(hyp))
    return jiwer.wer(r, h) if r else float(bool(h))

class AsrMer(ORM):
    def __call__(self, completions, solution, **kwargs):
        return [1.0 - min(mer(ref, hyp), 1.0) for hyp, ref in zip(completions, solution)]

orms['asr_mer'] = AsrMer
```

Guardrails: a KL penalty to the SFT model, a watch on output length (especially for Tamil and Burmese), and evaluating *all* tasks and languages after RL. Use mixed-task, mixed-language batches.

---

## 9. Evaluation protocol

Use one script that runs every task × language dev/test set for a checkpoint or adapter and writes a single table:

| Task | Metric | Slices |
|---|---|---|
| ASR | WER / CER / **MER** | per language; Singlish vs. standard SG English; monolingual vs. code-switched; switch-point error; entity error rate on local names |
| ST | chrF++, COMET, BLEU | per language pair; code-switched source vs. monolingual |
| SER | UA, WA, macro-F1 (plus arousal/valence CCC if used) | per language/community; acted vs. natural |
| SQA | EM / F1, LLM-judge | general vs. **culturally grounded** items; per language |
| AEC | mAP (with §7.1 scoring), micro/macro-F1 | standard classes vs. **local** classes |
| General | general audio-instruction and text-chat set, plus local-language prompts | catches forgetting and wrong-output-language behaviour |

Always compare against **base zero-shot**, the **stage-1 joint model**, **per-task single-task LoRA** (an upper bound), and an external reference where you have one (e.g. MERaLiON on AudioBench-style tasks).

---

## 10. Practical notes for this setup

- **Memory.** Attention LoRA fits on 1 H200. Encoder LoRA plus a longer `max_length` may need 2–4 GPUs with DeepSpeed ZeRO-2. Full encoder unfreeze needs ZeRO-3/FSDP.
- **Sequence length.**
  - Audio: 13 audio tokens/s (30 s ≈ 390 tokens, 5 min ≈ 3900). There is no fixed 30 s window as in Whisper. The encoder handles variable-length audio with no fixed pad or cap: it processes 1 s chunks and attends within 8 s windows (§0).
  - Text: the targets for Tamil and Burmese are ~5–6× longer than English.
  - For long SQA, keep `max_length` at 8k or more and use packing / padding-free.
- **Always inspect adapter tensor names after training** (the regex leak into the audio tower). Save a `target_modules` file per experiment.
- **Seed and log everything.** Run at least 2 seeds on small sets (SER dev and low-resource-language tests are noisy).

---

## 11. Suggested roadmap

| # | Experiment | Effort | Why |
|---|---|---|---|
| 1 | Zero-shot eval of all tasks × languages; decide normalisation and code-switch conventions (§4.3) | S | find headroom; supported vs. unsupported languages |
| 2 | Single-task (and per-language-group) LoRA, short runs | S | upper bounds + interference clustering (§3.4) |
| 3 | **Stage 0**: encoder+projector+attention LoRA, ASR-heavy on ta/th/fil/my/km, Singlish and code-switching | M | acoustic foundation for unsupported languages |
| 4 | **Stage 1**: joint multi-task LoRA, task×language temperature sampling, per-sample loss, cultural text replay | M | the main baseline |
| 5 | Log-likelihood scoring for SER/AEC; contextual biasing prompts for local entities | S | free gains |
| 6 | ST / SQA CoT formats; synthetic code-switched + cultural SQA data | S–M | largest gains for code-switched ST and cultural SQA |
| 7 | **Stage 2**: speech track (LLM-side) vs. AEC track (encoder LoRA + layer-weighted features) | M | your plan with parameter separation |
| 8 | Merge the track adapters vs. prompt routing | S | single-model vs. multi-adapter deployment |
| 9 | Routing profile per task/language + targeted expert LoRA | M | extra MLP capacity where it matters |
| 10 | DPO on targeted failures (flattening, Singlish over-correction), then GRPO for SQA/ST | M–L | code-switch fidelity, hallucination, cultural reasoning |
| 11 | Dual encoder (MMS/w2v-BERT for new languages, BEATs, emotion2vec) or LoRA-MoE | L | only if languages, AEC or SER still lag after #3/#7 |

---

## 12. Training stack: HF `swift sft` vs. Megatron (checked 2026-09-24)

**Raw Megatron-LM** (`toolkits/Megatron-LM`, megatron-core 0.20.0, commit 7de072a) is **not a practical path for this model.**
- It has no Qwen3-Omni model definition, no AuT encoder and no HF→Megatron weight converter for it.
- I found no support in core for Qwen3's interleaved multimodal RoPE (`mrope_interleaved`, sections [24, 20, 20]).
- Its MIMO framework (`examples/mimo`: HF encoder wrapper + GPT/MoE language model) could host it in principle. That would be a porting project: model spec, weight conversion, data pipeline and chat template.
- Its RL stack (`train_rl.py`) is text-only.

**Megatron-SWIFT** (`megatron sft` / `megatron rlhf` in ms-swift) **is the practical path.**
- Its docs list Qwen3-Omni for SFT, DPO, KTO, RM and GRPO. mcore-bridge loads HF safetensors directly and saves HF safetensors or PEFT LoRA back (`--save_safetensors true`), so the eval and serving code stays HF.
- Reference script: `ms-swift/examples/megatron/multimodal/omni/moe.sh` (LoRA, EP=2, packing, 2 × ~50 GiB).
- In Megatron-SWIFT, "vit" flags cover the **audio_tower** too.

**Where the speed comes from:**
- expert parallelism with grouped GEMM over the 128 experts;
- **packing/padding-free**, which matters a lot for variable-length short audio clips;
- fused cross-entropy and the distributed optimizer.

ms-swift's own benchmark for Qwen3-30B-A3B full fine-tuning on 16×A800 at 8K context: Megatron **9.6 s/it**, DeepSpeed ZeRO-3 **91.2 s/it**, ZeRO-2 out of memory. For a dense 14B model the gain was only ~12 %. For attention-only LoRA the gain will be smaller than for full or expert tuning, but the forward/backward pass through the experts dominates either way. With transformers ≥5, `swift sft --experts_impl grouped_mm` narrows the gap. **Benchmark both on your own data before switching.**

**Requirements and environment:**
- megatron-core **>=0.16,<0.20** (so the local 0.20.0 checkout is out of range). 0.17+ needs Python ≥3.12, so our Python 3.11 container uses 0.16.1;
- Transformer Engine ≥2.3, apex, mcore-bridge ≥1.3; recommended torch 2.8 / CUDA 12.8.

nemo_env (torch 2.6+cu124) has none of these installed. Do not install them into nemo_env (see the CUDA-13 wheel breakage). **Built and tested (2026-09-24):** `/scratch/prj0000000234/sailorhb/container/swift_megatron_cu128.sif` (12.6 GB). It holds ms-swift 4.6.0.dev0, megatron-core 0.16.1, Transformer Engine 2.13, torch 2.10/CUDA 12.8 and vLLM 0.17.1. Launch it with `run_container.sh`; see the README there. In a 1-GPU Megatron-SWIFT LoRA smoke test it ran at ~4.2 s/step with a 63.5 GiB peak.

**Multi-node feasibility: good.**
- Hardware: partitions `h200n` and `h200n-long` (up to 7 days), 13 nodes × 8 H200, 8 × 400 Gb/s NDR InfiniBand per node with GPU–NIC PIX affinity.
- Suggested layouts:
  - **LoRA:** EP=8 within a node, data-parallel across nodes, TP=1. Hidden size 2048 is small, and 4 KV heads cap TP at 4.
  - **Full or expert fine-tuning:** EP=8–16 plus the distributed optimizer (Adam needs ~490 GB of state + weights for 30.5B params).
  - The 0.65B audio encoder is replicated on every rank.
- Launch with `srun` per node and ms-swift's `NNODES / NODE_RANK / MASTER_ADDR` env vars (see `ms-swift/examples/megatron/multi-node`).

**Division of labour:**
- Prototype data, prompts and **architecture changes** (layer-weighted encoder features, CTC head, dual encoder) in HF `swift sft`, where the model is plain PyTorch.
- Run the large Stage 0–2 runs, expert LoRA / full fine-tuning and GRPO in Megatron-SWIFT. There the audio encoder stays the HF implementation, so encoder-side changes carry over easily. Changes to the thinker (e.g. a CTC head on its outputs, new losses) must be written inside megatron-core modules.

**Deployment: Megatron does not help directly.**
- Megatron-LM's inference engine has no Qwen3-Omni.
- Instead, export merged HF weights and serve with vLLM. **Checked 2026-09-29:** vLLM 0.17.1 serves the base Qwen3-Omni thinker with audio input, but *not* LoRA adapters for it, so deployment uses merged weights and multi-LoRA serving is not available (`DEPLOYMENT.md`).
- FP8 weight quantization on H200 is the main deployment-efficiency lever.
