"""Inference / env-compatibility check for Qwen3-Omni-30B-A3B-Instruct (HF transformers).

Runs a few smoke tests and prints PASS/FAIL per test:
  1. text-only chat
  2. ASR on a local wav
  3. audio + text question
  4. (--talker) speech output via the talker, written to outputs/<env>_talker.wav

Usage (on a GPU node):
    python infer_qwen3_omni.py                     # text-only output, talker disabled (~10GB less VRAM)
    python infer_qwen3_omni.py --talker            # also test speech generation
    python infer_qwen3_omni.py --audio my.wav --attn sdpa
"""
import argparse
import importlib
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = "/scratch/prj0000000234/sailorhb/hf_models/"
MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def print_env():
    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    for m in ["torch", "transformers", "accelerate", "flash_attn", "qwen_omni_utils", "librosa", "soundfile"]:
        try:
            print(f"  {m:16s} {getattr(importlib.import_module(m), '__version__', 'ok')}")
        except Exception:
            print(f"  {m:16s} MISSING")


def pick_attn(requested):
    if requested != "auto":
        return requested
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--audio", default=os.path.join(HERE, "assets", "asr_en.wav"))
    ap.add_argument("--attn", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--talker", action="store_true", help="keep talker and test speech output")
    ap.add_argument("--speaker", default="Ethan")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    print_env()
    import torch
    import soundfile as sf
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    env_name = os.environ.get("CONDA_DEFAULT_ENV") or os.path.basename(os.path.dirname(os.path.dirname(sys.executable)))
    attn = pick_attn(args.attn)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}  attn={attn}  talker={args.talker}", flush=True)

    t0 = time.time()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation=attn,
    )
    if not args.talker:
        model.disable_talker()
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    print(f"Loaded in {time.time() - t0:.1f}s, VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    def run(conversation, return_audio=False):
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(text=text, audio=audios, images=images, videos=videos,
                           return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(model.device).to(model.dtype)
        gen_kwargs = dict(thinker_max_new_tokens=args.max_new_tokens, thinker_do_sample=False,
                          use_audio_in_video=False, return_audio=return_audio)
        if return_audio:
            gen_kwargs["speaker"] = args.speaker
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        # Returns (text_ids, audio) in 4.57; be lenient about other shapes across versions.
        text_ids, wav = (out if isinstance(out, tuple) else (out, None))
        seqs = getattr(text_ids, "sequences", text_ids)
        reply = processor.batch_decode(seqs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)[0]
        return reply, wav

    tests = [
        ("text_only", [{"role": "user", "content": [
            {"type": "text", "text": "In one sentence, what is the capital of Singapore?"}]}], False),
        ("asr", [{"role": "user", "content": [
            {"type": "audio", "audio": args.audio},
            {"type": "text", "text": "Transcribe the audio into text."}]}], False),
        ("audio_qa", [{"role": "user", "content": [
            {"type": "audio", "audio": args.audio},
            {"type": "text", "text": "Describe the speaker (gender, emotion, accent) and summarise what they say."}]}], False),
    ]
    if args.talker:
        tests.append(("talker_speech", [{"role": "user", "content": [
            {"type": "text", "text": "Say hello and introduce yourself in one short sentence."}]}], True))

    results = {}
    for name, conv, want_audio in tests:
        print(f"\n=== {name}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            reply, wav = run(conv, return_audio=want_audio)
            print(f"[{time.time() - t:.1f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB] {reply!r}")
            if want_audio:
                if wav is None:
                    raise RuntimeError("talker returned no audio")
                os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
                out_wav = os.path.join(HERE, "outputs", f"{env_name}_talker.wav")
                sf.write(out_wav, wav.reshape(-1).float().cpu().numpy(), samplerate=24000)
                print(f"speech written: {out_wav}")
            results[name] = "PASS" if reply.strip() else "FAIL (empty)"
        except Exception as e:
            traceback.print_exc()
            results[name] = f"FAIL ({type(e).__name__}: {e})"[:200]

    print(f"\n##### SUMMARY env={env_name}")
    for k, v in results.items():
        print(f"  {k:14s} {v}")
    sys.exit(0 if all(v == "PASS" for v in results.values()) else 1)


if __name__ == "__main__":
    main()
