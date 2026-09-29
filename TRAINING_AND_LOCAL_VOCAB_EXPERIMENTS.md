# Grounded training and local-vocabulary experiments for Qwen3-Omni

Date: 2026-09-28. Scope: text outputs from Qwen3-Omni's Thinker for Southeast Asian and Singapore ASR, speech translation, spoken QA, emotion, and sound events. These are **proposals to test**, not measured improvements on this model.

The current folder contains a working LoRA smoke test on 1,000 LibriSpeech training utterances and a 20-utterance `dev-other` generation check. Clean WER on that check went from 2.98% (base) to 3.97% (LoRA trained on raw uppercase targets). That is evidence that target format matters; it is too small and too different from local conversational speech to rank the ideas below. The MDS adapter in `MDS_DATA_PIPELINE.md` is a design, not an implemented data loader in this folder.

## First, define the failures

Use a held-out set by speaker and recording source, with answerable and unanswerable items. Keep the reference transcript's language and spelling convention explicit. Report results by language, code-switch boundary, entity type, acoustic condition, and whether a candidate list was supplied. Alongside WER/CER/MER and QA accuracy, count **speech inserted on silence or non-speech**, **unsupported named entities**, **false answers to absent-evidence questions**, and **correct-answer coverage at a fixed abstention rate**. Use the same decoding settings in every ablation.

Separate three kinds of grounding:

1. **Acoustic:** did the answer come from the recording, including which word or event was heard?
2. **Contextual spelling:** did a candidate list help spell a word that was actually heard?
3. **External factual:** does a claimed local fact have support in a supplied, dated source? A lexicon is a spelling aid, not a factual source.

## Model and training experiments, ordered by cost

| Experiment | Change | Primary test and stop condition |
|---|---|---|
| A. Balanced SFT and acoustic abstention | Per-example loss normalization or explicit task weights; replay; supervised silence, noise-only, absent-speaker and unanswerable QA cases | Reduce false speech/answers without lowering answerable QA accuracy or minority-language ASR. Start here. |
| B. Audio counterfactual training | Keep question and prompt fixed; replace audio with a closely matched clip that changes one name, event, count, or language span. Train both correct outputs. Optionally add the paired loss below. | Answer must change with audio; preserve ordinary ASR/ST accuracy. |
| C. Selective audio adaptation | Compare projector-only training, upper 4/8 AuT layers with LoRA, and partial unfreezing of those layers at a lower learning rate; keep Thinker training fixed. | Run only where the base model's errors are demonstrably acoustic. Check supported and unsupported languages and forgetting. |
| D. Evidence output | For spoken QA, train a short answer plus a verbatim supporting phrase or time interval; for absent evidence, output `insufficient_audio_evidence`. For factual QA, require a supplied source ID. | Human-check support precision, not just answer accuracy. Avoid long free-form rationales that can themselves fabricate evidence. |
| E. Small acoustic verifier | Add a pooled-audio binary head for speech presence or candidate-term presence, or train the same decisions as short auxiliary text tasks. Gate term biasing on its calibrated score. | Worth keeping only if it cuts false insertions at matched entity recall. The text-task version needs no model code. |
| F. CTC alignment head | Try only after A-C if ASR still inserts words or loses switch points. Attach to an encoder state with enough time steps for the chosen character/subword targets. | The current guide estimates about 13 *output* audio tokens/s; CTC on that compressed sequence may violate its required `T >= target length`, especially for character-level Tamil/Burmese. Verify lengths first. |

For B, the least complex implementation is ordinary supervised fine-tuning on both members of each pair. A second ablation can add a length-normalized margin on the gold answer `y`:

`L = L_SFT + lambda * max(0, margin - score(y | matching_audio, q) + score(y | mismatched_audio, q))`, where `score` is the mean gold-token log probability.

The mismatched clip must make `y` false, and should match duration, speaker/accent, and recording quality where possible; otherwise the model can solve the pair from artifacts. Keep a small percentage of paired examples in each batch and compare against **the same examples without the margin** to isolate the objective. For generated wrong responses, preference tuning is another ablation, but text-only preference cues may let a multimodal model ignore audio; explicitly test audio swaps. [AHA](https://arxiv.org/abs/2512.24052) motivates audio counterfactual negatives, while [mDPO](https://arxiv.org/abs/2406.11839) identifies the analogous modality-ignoring problem in multimodal preference training. The exact margin above is a proposed experiment here, not a reported Qwen3-Omni result.

For C, compare on the **same data and update budget**, and use a smaller encoder learning rate than the projector/Thinker. Official language coverage is a useful prior, not proof that a particular layer is the bottleneck. A separate [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) baseline or teacher is useful for languages it covers (including Thai and Filipino), but verify its coverage for each target language; it does not replace local annotations.

For multitask interference, log per-task/language generated metrics and update contributions. If one task regresses, first rebalance sampling and per-example losses; only then try gradient conflict methods or separate adapters. More trainable parameters will not fix inconsistent transcript rules or a biased task mix.

## Local vocabulary: a concrete contextual-biasing study

The question is **whether a list helps recognize a term that was spoken without inserting a plausible term that was not spoken**. This is especially important for Singapore places, food, agencies, schools, personal names, dialect words, and English–Mandarin/Malay/Tamil switching. Prompt-only biasing is the first baseline; a gated term verifier is the small model-side extension.

### 1. Build a versioned lexicon

Each entry should have `entity_id`, canonical written form, type, language/script, spelling aliases, localized names, pronunciation or romanization if available, locality/domain, and provenance. A localized name may refer to the same place but is not automatically a valid transcription of speech in another language. Keep aliases separate from genuinely different entities. Example:

```json
{"entity_id":"place:toa_payoh","canonical":"Toa Payoh","type":"place","language":"en-SG","spelling_aliases":[],"localized_names":{"zh-SG":"大巴窑"},"domain":"transit","source":"reviewed_local_lexicon_v1"}
```

Native reviewers should settle when the audio warrants an English form, Hanzi, or a loanword spelling. Do not convert a minority-language span into English merely because the English form appears in the candidate list. Protect evaluation by splitting speakers, recordings, and candidate-list templates; also maintain a set of **unseen entities** to test whether the method can use a new list at inference.

### 2. Construct lists without answer leakage

At training time, sample list sizes `0, 5, 10, 20` and shuffle order. Use four conditions, with the same audio and gold transcript wherever possible:

- **True term plus distractors:** include the spoken term and several real alternatives of the same type and locale.
- **Distractors only:** the spoken term is absent from the list; the transcript still says what was heard.
- **No local entity:** a plausible list accompanies speech that contains none of its terms.
- **No list:** ordinary ASR replay, so the model remains usable without context.

Mine hard distractors by pronunciation, syllable shape, spelling, or code-switch context, then have a reviewer check that the candidate is truly absent. Include confusable local pairs and acoustic reductions, not just unrelated place names. Do not use test transcripts to build candidate lists. Synthetic speech can expand coverage, but report a real-speech test separately.

Prompt contract:

> Possible spellings for this recording: Toa Payoh; Tiong Bahru; Tampines. Use a spelling only if supported by the audio. Transcribe the speech in its original languages; the list may contain no spoken term.

The word **possible** matters: the list is not a claim that any term occurs. Mix prompt languages and vary list order during training. At inference, candidate lists should come from *available* context (meeting agenda, route, approved contact list, domain metadata), not a reference transcript.

### 3. Add training signal in stages

1. **Prompt-only SFT:** ordinary transcript loss across the four conditions. Compare with no-list SFT at the same update budget.
2. **Entity-span weighting:** as a separate ablation, give gold transcript tokens inside reviewed entity spans a modest extra loss weight (for example, test 1x versus 2x). Keep the distractor-only and no-entity cases; monitor false insertions because overweighting names can teach the model to invent them. A [keyword-aware biasing study on Singapore NSC Part 2](https://www.isca-archive.org/interspeech_2025/kwok25b_interspeech.html) motivates the objective but used a different ASR architecture and cautions about synthetic-audio artifacts.
3. **Verification as an auxiliary task:** for `(audio, candidate)` train `heard` / `not heard`, using transcript-aligned entity spans when available. A short text answer is simplest; a pooled-audio classification head is the small architecture variant. Train with near-homophones and candidates from a wrong locale/domain. Calibrate a threshold on held-out data and pass only accepted terms to the ASR prompt.
4. **Counterfactual term pairs:** same prompt/list, with two matched clips speaking different candidates. Add the paired audio loss from experiment B or targeted preference pairs after SFT. This tests whether the model actually listens rather than copying a frequent name.
5. **Large lexicon retrieval, only if needed:** use a lightweight acoustic term retriever to shortlist `k` candidates from the lexicon, then verify and pass only a small list to Omni. A first-pass transcript or domain lookup is a cheaper baseline, but can miss exactly the entity that needs rescue. [BR-ASR](https://www.isca-archive.org/interspeech_2025/gong25_interspeech.html) is an example of speech-to-bias retrieval at scale; it is a research reference, not an implementation already present here.

Prompt lists get brittle as they grow. A [2024 SpeechLLM study](https://www.isca-archive.org/interspeech_2024/gong24b_interspeech.html) reported hallucination with larger lists, and a [2026 comparison](https://arxiv.org/abs/2608.05759) found sensitivity to distractor count and order. The guide now calls for a measured list-size sweep rather than assuming a long list helps. Prefer a calibrated shortlist over dumping the full local lexicon into the prompt.

### 4. Evaluate benefit and harm together

Report entity recall and entity error rate by type/language, plus **false entity insertions per audio hour** when the term is absent. Also report overall WER/CER/MER, switch-boundary error, and unchanged-word WER. Evaluate each of: no list, true term included, distractors only, no-term audio, shuffled list order, unseen entities, and non-speech. Plot recall against false insertion as verifier threshold or shortlist size changes; compare methods at a matched false-insertion rate. For a factual downstream answer, separately check whether the entity was heard and whether any outside claim is source-supported.

Minimum ablation: base Omni; no-list SFT; prompt-only list SFT; entity-weighted SFT; prompt plus verification; then optional audio-paired loss. This isolates gains from the data format, weighted objective, verifier, and paired training. Use the same candidate generator at train and deployment, or report the mismatch explicitly.

## Suggested first run

1. Curate a small, human-reviewed real-speech evaluation set for local entities, negative-list cases, and unanswerable audio questions. Establish base Omni and Qwen3-ASR baselines where supported.
2. Implement the lexicon and four list conditions in the data adapter; train prompt-only SFT with fixed transcription rules. The MDS design's `row_fn` is the intended hook once implemented.
3. Add the auxiliary `heard/not heard` task, then the paired-audio experiment. Keep each ablation's data, steps, decoding, and evaluation slices fixed.
4. Only if errors remain acoustic, test selective AuT adaptation or a small verifier head. Keep a supported-language replay slice and check general audio tasks for forgetting.

The contribution to aim for is **acoustically verified contextual biasing for code-switched local entities**, measured by rare-word recall *and* false insertions, with paired counterfactuals proving that the audio controls the answer. That is more focused and cheaper than adding a second large encoder or modifying the MoE router.
