"""Per-sample wrapper around the old batch augmentation (vendored data_augmentation.py, unchanged).

The old `augment_data(batch_audios, batch_responses, batch_tasks)` ran inside the collator on a whole batch:
with probability 0.8 it mixed a "crowd" background from the other clips in the batch, then per clip applied
one of pitch / speed / RIR (tasks ASR, ST, SQA, SDS, DS) and additive crowd noise at 5–20 dB SNR.
Rows are now built one at a time in dataloader workers, so the crowd background comes from a small
reservoir of recent clips seen by the same worker. Effect probabilities are the ones `augment_data` used.

Randomness: the vendored functions use the global `random` / `np.random`, seeded per dataloader worker by
ms-swift, so augmentation is random per run, not reproducible per sample (as in the old trainer).
"""
import io
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from .data_augmentation import _augment_sample, prepare_crowd_bg_audio

SAMPLE_RATE = 16000
OVERALL_LENGTH = 30 * SAMPLE_RATE

# Same values as the old augment_data() call.
OLD_PROBS = dict(change_pitch_prob=0.01, change_speed_prob=0.05, apply_rir_prob=0.05,
                 repeat_audio_prob=0.0, apply_noise_prob=0.1)
CROWD_PROB = 0.8


class AudioAugmenter:
    def __init__(self, reservoir_size: int = 8, crowd_prob: float = CROWD_PROB, **probs):
        self.reservoir = deque(maxlen=reservoir_size)
        self.crowd_prob = crowd_prob
        self.probs = {**OLD_PROBS, **probs}

    def __call__(self, audio_bytes: bytes, response_text: str, task: str) -> Tuple[bytes, str]:
        """Returns (WAV PCM16 16 kHz bytes, response text). The text only changes when repeat_audio_prob > 0
        (old default 0) or when the clip is shorter than 500 samples (the old code then returned "EMPTY")."""
        audio = decode_16k(audio_bytes)
        self.reservoir.append(audio[:OVERALL_LENGTH])
        crowd = None
        if random.random() < self.crowd_prob:
            crowd = prepare_crowd_bg_audio(list(self.reservoir), desired_length=OVERALL_LENGTH)
        out, text = _augment_sample(audio, {'text': response_text}, task, crowd,
                                    overall_length=OVERALL_LENGTH, **self.probs)
        return encode_wav(out), text


def decode_16k(audio_bytes: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32', always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float64)  # the old code worked on sf.read's default float64


def encode_wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:  # noise / RIR can push above full scale; avoid PCM16 clipping
        audio = audio / peak
    sf.write(buf, audio, SAMPLE_RATE, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def maybe_augmenter(enabled: bool, **kwargs) -> Optional[AudioAugmenter]:
    return AudioAugmenter(**kwargs) if enabled else None
