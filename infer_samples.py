"""Run Qwen3-Omni-30B-A3B-Instruct on real sample audio and score ASR.

- LibriSpeech dev-other (clean) + the matching corruption_dev versions (additive noise / overlap speech)
- a few MUSAN noise clips for sound description (no reference)
Writes per-utterance results to outputs/samples_<tag>.jsonl and prints WER per condition.

Usage (GPU node, nemo_env):
    PYTHONNOUSERSITE=1 python infer_samples.py --n 20 --severity medium
    PYTHONNOUSERSITE=1 python infer_samples.py --adapter outputs/swift_lora_asr_smoke/<run>/checkpoint-60  # needs peft
"""
import argparse
import glob
import json
import os
import random
import re
import time

import jiwer
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = "/scratch/prj0000000234/sailorhb/hf_models/"
MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DATA = "/scratch/prj0000000234/sailorhb/data"
ASR_PROMPT = "Transcribe the English audio into text. Output only the transcription."
SOUND_PROMPT = "What sounds do you hear in this audio? Describe them briefly."


def normalize(text):
    text = text.upper().replace("’", "'")
    text = re.sub(r"[^A-Z0-9' ]+", " ", text)
    return " ".join(text.split())


def load_items(n, severity, seed):
    """Pick n utterances present in both corruption sets, keyed by utt_id."""
    conds = {}
    for group in ["additive-noise", "overlap-speech"]:
        path = f"{DATA}/corruption_dev/metadata-dev-other-realistic-{group}-{severity}.jsonl"
        with open(path) as f:
            conds[group] = {r["utt_id"]: r for r in map(json.loads, f)}
    common = sorted(set(conds["additive-noise"]) & set(conds["overlap-speech"]))
    random.Random(seed).shuffle(common)
    items = []
    for uid in common[:n]:
        a = conds["additive-noise"][uid]
        items.append(dict(utt_id=uid, condition="clean", audio=a["original_audio_path"], ref=a["original_transcript"]))
        for group, recs in conds.items():
            r = recs[uid]
            items.append(dict(utt_id=uid, condition=f"{group}-{severity}", audio=r["audio_path"],
                              ref=r["reference_transcript"], snr_db=r["corruption_parameters"].get("snr_db")))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--severity", default="medium", choices=["mild", "medium", "severe"])
    ap.add_argument("--n-sound", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--adapter", default=None, help="LoRA checkpoint dir (ms-swift/PEFT); merged into the weights")
    args = ap.parse_args()

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2")
    model.disable_talker()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        print(f"merged LoRA adapter: {args.adapter}", flush=True)
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    def generate(audio, prompt):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": audio}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        inputs = processor(text=text, audio=audios, images=images, videos=videos,
                           return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(model.device).to(model.dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, thinker_max_new_tokens=args.max_new_tokens, thinker_do_sample=False,
                                 use_audio_in_video=False, return_audio=False)
        text_ids = out[0] if isinstance(out, tuple) else out
        seqs = getattr(text_ids, "sequences", text_ids)
        return processor.batch_decode(seqs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()

    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    out_path = os.path.join(HERE, "outputs", f"samples_{args.severity}_n{args.n}{'_lora' if args.adapter else ''}.jsonl")
    items = load_items(args.n, args.severity, args.seed)
    by_cond = {}
    t_start = time.time()
    with open(out_path, "w") as fout:
        for i, it in enumerate(items):
            t = time.time()
            hyp = generate(it["audio"], ASR_PROMPT)
            it.update(hyp=hyp, sec=round(time.time() - t, 2),
                      wer=jiwer.wer(normalize(it["ref"]), normalize(hyp) or "<empty>"))
            fout.write(json.dumps(it) + "\n")
            by_cond.setdefault(it["condition"], []).append(it)
            print(f"[{i + 1}/{len(items)}] {it['condition']:26s} {it['utt_id']} WER={it['wer']:.3f} ({it['sec']}s)\n"
                  f"   REF: {normalize(it['ref'])}\n   HYP: {hyp}", flush=True)

        noise = sorted(glob.glob(f"{DATA}/musan/noise/all_musan_noise/*.wav"))
        for path in random.Random(args.seed).sample(noise, min(args.n_sound, len(noise))):
            desc = generate(path, SOUND_PROMPT)
            fout.write(json.dumps(dict(condition="musan-sound", audio=path, hyp=desc)) + "\n")
            print(f"[sound] {os.path.basename(path)}\n   {desc}", flush=True)

    print(f"\n##### ASR SUMMARY ({args.n} utts, dev-other, severity={args.severity}, "
          f"total {time.time() - t_start:.0f}s) -> {out_path}")
    for cond, rs in by_cond.items():
        corpus_wer = jiwer.wer([normalize(r["ref"]) for r in rs], [normalize(r["hyp"]) or "<empty>" for r in rs])
        print(f"  {cond:28s} corpus WER {100 * corpus_wer:6.2f}%   mean latency {sum(r['sec'] for r in rs) / len(rs):.2f}s")


if __name__ == "__main__":
    main()
