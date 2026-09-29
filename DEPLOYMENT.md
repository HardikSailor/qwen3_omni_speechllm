# Deploying the fine-tuned Qwen3-Omni: serving, eval and partner hand-off

Written 2026-09-29. Scope: what a partner company needs to run our fine-tuned model on its own servers, how hard that is, and the constraints this puts on training choices *now*. Items marked **(untested)** are plans, not measurements.

## 1. Key finding: vLLM 0.17.1 cannot serve LoRA adapters for Qwen3-Omni

Checked 2026-09-29 inside the container (`vllm.model_executor.models.interfaces.supports_lora`):

| Model class (vLLM 0.17.1) | `supports_lora` |
|---|---|
| `Qwen3OmniMoeThinkerForConditionalGeneration` | **False** |
| `Qwen2_5OmniThinkerForConditionalGeneration` (for comparison) | True |

- The Qwen3-Omni thinker class does not implement `SupportsLoRA`, so `vllm serve --enable-lora` / `swift infer --infer_backend vllm --adapters ...` will not load our adapters. This holds for **every** LoRA we train (LLM attention, experts, audio encoder), not only the new audio-encoder LoRA.
- vLLM does have *experimental* LoRA for multimodal towers (`enable_tower_connector_lora`), and the Qwen3-Omni class has the hook for it (`get_num_mm_encoder_tokens`). It is still unusable while `SupportsLoRA` is missing.
- Base-model serving works: vLLM 0.17.1 transcribed 3/3 LibriSpeech clips exactly (container smoke test, 2026-09-24).
- Newer vLLM releases may add LoRA for this model. **Not checked.** Re-check before relying on it (`supports_lora(...)` one-liner above).

**What was wrong before:** `PIPELINE_PLAN.md` (§2, §3.3, §5, §10.5) and `DECISION_LOG.md` (D18, §6) assumed "vLLM loads the adapters". Those places are now corrected and point here (D22).

## 2. Consequences

### 2.1 For deployment: ship a merged model

Merge the LoRA into the base weights once, at the end: `swift export --adapters <ckpt> --merge_lora true`.
- The result is a plain Hugging Face model folder with **exactly the architecture of the public `Qwen/Qwen3-Omni-30B-A3B-Instruct`**. Merging only changes weight values, including audio-encoder LoRA.
- The partner needs no ms-swift, Megatron, PEFT, mosaic or any of our code: stock vLLM (or stock transformers) loads it.

(untested) Before the first hand-off, check the following:
- the merge of an adapter from each trainer (HF and Megatron adapters are both PEFT format);
- whether the merged folder keeps the talker / code2wav weights. We train with `ENABLE_AUDIO_OUTPUT=0`, so the talker is not loaded. For text-only output the thinker is enough, but the folder must still load in vLLM and transformers;
- merged size (the base HF folder is 66 GB with the talker);
- that vLLM output on the check set (§4) matches transformers + PEFT on the unmerged adapter.

### 2.2 For our own evaluation (phase 5, `eval_omni.py`)

The plan "vLLM + adapter per checkpoint" does not work. Options:

| Option | Cost | Use for |
|---|---|---|
| transformers + PEFT (`swift infer --infer_backend transformers --adapters ...`) | slower generation; no extra disk | frequent checks during training, small val sets |
| merge → vLLM → delete merged copy | ~66 GB disk and a few minutes per checkpoint; fast generation | final numbers on full test sets, best-checkpoint selection |
| newer vLLM with Qwen3-Omni LoRA | unknown; would need a container rebuild | re-check later |

Default: **transformers + PEFT for in-training eval, merge + vLLM for final eval.** Merged copies are temporary; delete them after scoring (disk policy, `PIPELINE_PLAN.md` §10.5). (The eval script doesn't exist yet.)

### 2.3 For the two-track plan (`FINETUNING_GUIDE.md` §3.3)

"Load adapter A or B per request" (multi-LoRA serving) is not available in vLLM for this model.
- **Alternative 1:** two merged models, which costs 2× GPU memory.
- **Alternative 2:** one model trained on both tracks.
- **Alternative 3:** HF + PEFT adapter switching, which is slow.

Take this into account before committing to separate tracks.

## 3. What the partner needs

| Item | Detail |
|---|---|
| GPUs | See §3.1 |
| Host | Linux, NVIDIA driver with CUDA 12.x support. Our H200 nodes run driver 565.57 (CUDA 12.7) with a CUDA 12.8 image through minor-version compatibility; check the chosen image's requirement. Docker (or Podman / Apptainer) + NVIDIA Container Toolkit |
| Serving software | Official `vllm/vllm-openai` image, **pinned to the exact version we validated** (0.17.1 today). For air-gapped sites: `docker save` → tarball → `docker load` |
| Model | The merged model folder (§2.1), mounted into the container; `HF_HUB_OFFLINE=1` so nothing is downloaded |
| Start command | `vllm serve /models/omni-sea --served-model-name omni-sea --tensor-parallel-size <N> --max-model-len <L>` → OpenAI-compatible HTTP API; audio sent as `input_audio` (base64) or `audio_url` content parts |
| Client | Any OpenAI-compatible client. We ship per-task examples (§4) |

We should *not* ship our training container: it is 12.6 GB and contains Megatron, DeepSpeed and ms-swift, which the partner doesn't need, and it runs under Apptainer, which many companies don't use.

**Difficulty:** low (install Docker + driver, load image, mount model, one command) *if* the rules in §5 hold. It becomes hard as soon as the model needs code that stock vLLM does not have.

### 3.1 Hardware sizing

| Precision | Weights (thinker) | Fits on | Status |
|---|---|---|---|
| bf16 | ~63 GB | 1× H200 141 GB; 2× H100/A100 80 GB (`--tensor-parallel-size 2`) | Base model tested on 1× H200 |
| bf16 on 1× H100 80 GB | ~63 GB | Loads, but leaves ~10 GB for the KV cache: fine for single requests, not for concurrent production load | (untested) |
| FP8 (`--quantization fp8`, Hopper / Ada GPUs) | ~32 GB | 1× H100, 1× L40S 48 GB | **(untested)**: quality must be measured on our eval sets per task and language |
| 4-bit AWQ / GPTQ | ~17 GB | 1× L40S / A6000 48 GB; possibly 24 GB cards | **(untested)**: needs a quantization pass and support for this MoE + audio model in the quantizer and in vLLM |

Throughput and concurrency need a load test on the partner's target GPU before we promise numbers.

## 4. The hand-off package

1. **Model folder**: merged weights, config, tokenizer, processor / chat template, plus `README` and `LICENSE` / `NOTICE` (§6).
2. **Serving recipe**: pinned image tag (or tarball), `docker run` / `docker-compose.yml`, recommended flags per GPU type (§3.1).
3. **Prompts**: the chat template and the exact task instructions used in training (`omni_mds/instructions_lib`, system prompt, output conventions such as casing, punctuation and code-switch format). Wrong prompts don't cause errors; they only lower quality, so this is essential.
4. **Client examples**: one per task (ASR, ST, SER, SQA, AEC), with audio format rules: sample rate (16 kHz in our data) and maximum length. Our training used `max_length 2048` tokens, and audio costs 13 tokens/s.
5. **Check set**: ~20 clips across tasks and languages with our expected outputs and a script that compares. The partner runs it after install to confirm their setup reproduces ours. Use greedy decoding for the check.
6. **Model card**: tasks and languages supported, eval results, known weaknesses, intended use.

## 5. Rules for training, from the deployment side

1. **Only weight-level changes (LoRA → merge, or full fine-tuning).** The architecture must stay identical to the public Qwen3-Omni so stock vLLM serves it. The ideas below in `FINETUNING_GUIDE.md` §5 add modules stock vLLM does not have; each would need a patched vLLM or our own serving code at the partner. Each must justify that cost:
   - CTC head (§5.5);
   - complementary / dual encoder (§5.2);
   - layer-weighted encoder features (§5.1);
   - MoLE / LoRA-MoE (§5.3);
   - soft tokens (§5.7), unless they are tokens already in the vocabulary.
2. **No new tokenizer tokens** unless unavoidable: they change the embedding size and the tokenizer files, and must be shipped and tested.
3. **Fix the prompt format early** (system prompt, instructions, target conventions: open decision in `DECISION_LOG.md` §6). The partner must reproduce it exactly.
4. **Evaluate what we ship.** Final numbers come from the merged model in the pinned vLLM version, and from the quantized model if we ship one.

## 6. Licences and legal

- **Base model:** Qwen3-Omni-30B-A3B-Instruct is published under Apache 2.0. Include the licence and a NOTICE in the package.
- **Training data:** the licences of the datasets in the final mix may restrict commercial use of a model trained on them. To our knowledge CoVoST 2 is CC BY-NC 4.0 (non-commercial) and GigaSpeech has its own terms of use. **This needs a proper licence review of every dataset in the final mix before any commercial hand-off.** The list above is from memory, not a review. Keep a per-dataset licence table next to the data inventory (`DECISION_LOG.md` §6, next step 5).
- **Weights leave our control.** An on-prem deployment gives the partner the full weights. Protection is contractual (licence agreement, usage terms), not technical.

## 7. Open items

1. (untested) `swift export --merge_lora` for an HF adapter and a Megatron adapter, including audio-encoder LoRA. Check the merged folder loads in vLLM 0.17.1 and matches PEFT outputs.
2. Re-check newer vLLM versions for Qwen3-Omni LoRA (would allow adapter-based eval and multi-LoRA serving).
3. FP8 serving: quality per task and language vs. bf16; memory and throughput on 1× H100.
4. Build the hand-off package template (§4) once the first real model exists.
5. Licence review of the final data mix (§6).
