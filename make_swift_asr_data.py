"""Build ms-swift ASR JSONL (messages + audios) from LibriSpeech.

    python make_swift_asr_data.py --train-n 1000 --val-n 50
Writes data/asr_train.jsonl (train-clean-100) and data/asr_val.jsonl (dev-other).
"""
import argparse
import glob
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
LIBRI = "/scratch/prj0000000234/sailorhb/data/LibriSpeech"
PROMPT = "Transcribe the English audio into text. Output only the transcription."


def load_split(split):
    items = []
    for trans in glob.glob(f"{LIBRI}/{split}/*/*/*.trans.txt"):
        d = os.path.dirname(trans)
        with open(trans) as f:
            for line in f:
                uid, text = line.strip().split(" ", 1)
                items.append((uid, os.path.join(d, uid + ".flac"), text))
    return sorted(items)


def write(items, path):
    with open(path, "w") as f:
        for uid, audio, text in items:
            f.write(json.dumps({
                "id": uid,
                "messages": [{"role": "user", "content": "<audio>" + PROMPT},
                             {"role": "assistant", "content": text}],
                "audios": [audio],
            }) + "\n")
    print(f"{path}: {len(items)} examples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=1000)
    ap.add_argument("--val-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    rng = random.Random(args.seed)
    train = load_split("train-clean-100")
    val = load_split("dev-other")
    write(rng.sample(train, args.train_n), os.path.join(HERE, "data", "asr_train.jsonl"))
    write(rng.sample(val, args.val_n), os.path.join(HERE, "data", "asr_val.jsonl"))


if __name__ == "__main__":
    main()
