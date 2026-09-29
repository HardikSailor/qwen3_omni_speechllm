import random
import re

import librosa
import numpy as np
from scipy.signal import fftconvolve

NORMAL_EFFECT_TASKS = ["ASR", "ST", "SQA", "SDS", "DS"]
REPEAT_EFFECT_TASKS = ["ASR"]


def normalize_audio(input_array):
    return input_array / np.max(np.abs(input_array))


def standardise_audio(input_array):
    _mean = np.mean(input_array)
    _std = np.std(input_array)
    return (input_array - _mean) / _std


def pad_audio(input_array, desired_length):
    input_array = input_array[:desired_length]
    if input_array.shape[0] < desired_length:
        pad_array = np.random.normal(0, 0.0001, desired_length - input_array.shape[0])
        input_array = np.concatenate([input_array, pad_array])
    return input_array


def add_noise(input_array, singal_power=None, noise_array=None, snr_dbs=3):
    if noise_array is None:
        noise_array = np.random.normal(0, 1, input_array.shape)
    noise_array = pad_audio(noise_array, input_array.shape[0])
    noise_array = standardise_audio(noise_array)

    if singal_power is None:
        singal_power = np.mean(input_array**2)

    desired_snr_linear = 10 ** (snr_dbs / 10)
    noise_power = singal_power / desired_snr_linear

    noise_array = noise_array * np.sqrt(noise_power)
    output_array = input_array + noise_array

    # output_array = normalize_audio(output_array)
    return output_array


def add_rir_effect(input_array, effect_length=8000, decay_coeff=-3):
    # Generate exponential decay
    decay = np.exp(decay_coeff * np.arange(effect_length) / effect_length)

    # Add randomness
    random_component = np.random.randn(effect_length)

    # Combine components to form the RIR
    rir = decay * random_component

    # Normalize
    rir /= np.max(np.abs(rir))

    output_array = fftconvolve(input_array, rir)
    # output_array = normalize_audio(output_array)

    return output_array


def change_speed(input_array, scaler=1):
    origin_dtype = input_array.dtype
    output_array = librosa.effects.time_stretch(input_array.astype("float64"), rate=scaler)

    # output_array = normalize_audio(output_array)
    return output_array.astype(origin_dtype)


def change_pitch(input_array, n_steps=2, sample_rate=16000):
    origin_dtype = input_array.dtype
    output_array = librosa.effects.pitch_shift(
        input_array.astype("float64"), sr=sample_rate, n_steps=n_steps, bins_per_octave=12
    )

    # output_array = normalize_audio(output_array)
    return output_array.astype(origin_dtype)


def apply_single_audio_effect(
    single_array,
    acceptable_length=30 * 16000,
    change_pitch_prob=0.1,
    change_speed_prob=0.1,
    apply_rir_prob=0.1,
    min_speed_scaler=0.9,
    max_speed_scaler=1.75,
    min_rir_length=4000,
    max_rir_length=12000,
    min_rir_decay_coeff=-3,
    max_rir_decay_coeff=-2,
    min_pitch_steps=-1,
    max_pitch_steps=1.5,
    epsilon=10,
):
    output_array = single_array.copy()
    prob_weights = [change_pitch_prob, change_speed_prob, apply_rir_prob]
    prob_weights.append(1 - sum(prob_weights))
    effect = random.choices(["pitch", "speed", "rir", "still"], weights=prob_weights, k=1)[0]
    # change pitch
    if effect == "pitch":
        current_pitch_steps = random.uniform(min_pitch_steps, max_pitch_steps)
        output_array = change_pitch(output_array, n_steps=current_pitch_steps)
        return output_array

    # change speed
    if effect == "speed":
        local_min_speed_scaler = output_array.shape[0] / (acceptable_length - epsilon)
        current_speech_scaler = random.uniform(
            max(min_speed_scaler, local_min_speed_scaler), max_speed_scaler
        )
        output_array = change_speed(output_array, scaler=current_speech_scaler)
        return output_array

    # add rir effect
    if effect == "rir":
        local_max_rir_length = acceptable_length - epsilon - output_array.shape[0]
        current_rir_length = random.randint(
            max(min(min_rir_length, local_max_rir_length), 10),
            max(min(local_max_rir_length, max_rir_length), 10),
        )
        current_decay_coeff = random.uniform(min_rir_decay_coeff, max_rir_decay_coeff)
        output_array = add_rir_effect(
            output_array, effect_length=current_rir_length, decay_coeff=current_decay_coeff
        )
        return output_array

    return output_array


def apply_noise_effect(
    single_array,
    singal_power=None,
    crowd_background_audio=None,
    apply_noise_prob=0.1,
    min_snr_dbs=5,
    max_snr_dbs=20,
):
    output_array = single_array.copy()
    if random.random() < apply_noise_prob:
        current_dbs = random.uniform(min_snr_dbs, max_snr_dbs)
        output_array = add_noise(
            output_array,
            singal_power=singal_power,
            noise_array=crowd_background_audio,
            snr_dbs=current_dbs,
        )
    return output_array


def prepare_crowd_bg_audio(batch_audios, desired_length=30 * 16000):
    if len(batch_audios) < 2:
        return None

    crowd_background_audio = np.zeros(desired_length)

    for _audio_array in batch_audios:
        recurrence_time = desired_length // _audio_array.shape[0]
        _audio_array = np.concatenate([_audio_array] * (recurrence_time + 1))

        for _ in range(2):
            start_index = random.randint(0, _audio_array.shape[0] - desired_length)
            _sub_array = _audio_array[start_index : start_index + desired_length]
            _sub_array = normalize_audio(_sub_array[-desired_length:])
            crowd_background_audio += _sub_array

    return crowd_background_audio


def merge_transcriptions(transcriptions):
    merged_transcriptions = []
    current_speaker = None
    current_sentence = []

    speaker_pattern = re.compile(r"^<([^>]+)>: ([\s\S]*)")

    for line in transcriptions:
        match = speaker_pattern.match(line)
        if match:
            speaker, sentence = match.groups()
            sentence = sentence.strip()
            if speaker == current_speaker:
                current_sentence.append(sentence)
            else:
                if current_sentence:
                    merged_transcriptions.append(
                        f"<{current_speaker}>: {' '.join(current_sentence)}"
                    )
                current_speaker = speaker
                current_sentence = [sentence]

    if current_sentence:
        merged_transcriptions.append(f"<{current_speaker}>: {' '.join(current_sentence)}")

    return " ".join(merged_transcriptions)


def _augment_sample(
    audio_array,
    data_response,
    data_task,
    crowd_background_audio,
    overall_length=30 * 16000,
    change_pitch_prob=0.1,
    change_speed_prob=0.1,
    apply_rir_prob=0.1,
    repeat_audio_prob=0.2,
    apply_noise_prob=0.1,
):

    original_audio = audio_array.copy()
    output_response = data_response["text"]

    if original_audio.shape[0] < 500:
        return original_audio, "EMPTY"

    if data_task in NORMAL_EFFECT_TASKS:
        audio_array = apply_single_audio_effect(
            audio_array,
            change_pitch_prob=change_pitch_prob,
            change_speed_prob=change_speed_prob,
            apply_rir_prob=apply_rir_prob,
        )

    repeat_count = 1
    while (
        (data_task in REPEAT_EFFECT_TASKS)
        and (repeat_count <= 5)
        and (audio_array.shape[0] < (overall_length - original_audio.shape[0]))
        and (random.random() < repeat_audio_prob)
    ):

        max_pad_length = overall_length - audio_array.shape[0] - original_audio.shape[0]
        pad_length = random.randint(0, max_pad_length)
        audio_array = pad_audio(audio_array, audio_array.shape[0] + pad_length)

        new_array = apply_single_audio_effect(
            original_audio,
            acceptable_length=overall_length - audio_array.shape[0],
            change_pitch_prob=change_pitch_prob,
            change_speed_prob=change_speed_prob,
            apply_rir_prob=apply_rir_prob,
        )

        audio_array = np.concatenate([audio_array, new_array])
        output_response = merge_transcriptions((output_response, output_response))
        repeat_count += 1

    # zhou yu lv tuan
    if data_task in NORMAL_EFFECT_TASKS:
        singal_power = np.mean(original_audio**2)
        audio_array = apply_noise_effect(
            audio_array,
            singal_power=singal_power,
            crowd_background_audio=crowd_background_audio,
            apply_noise_prob=apply_noise_prob,
        )

    return audio_array, output_response


def augment_data(batch_audios, batch_responses, batch_tasks):
    """
    batch_audios: list of np array
    batch_responses: list of dict("text", "audio")
    """
    overall_length = 30 * 16000

    crowd_background_audio = None
    if random.random() < 0.8:
        crowd_background_audio = prepare_crowd_bg_audio(
            batch_audios,
            desired_length=overall_length,
        )

    output_batch_audios = []
    output_batch_responses = []

    for idx in range(len(batch_audios)):
        audio_array = batch_audios[idx]
        data_response = batch_responses[idx]
        data_task = batch_tasks[idx]

        output_array, output_response = _augment_sample(
            audio_array,
            data_response,
            data_task,
            crowd_background_audio,
            overall_length=overall_length,
            change_pitch_prob=0.01,
            change_speed_prob=0.05,
            apply_rir_prob=0.05,
            repeat_audio_prob=0.0,
            apply_noise_prob=0.1,
        )

        output_batch_audios.append(output_array)
        data_response["text"] = output_response
        output_batch_responses.append(data_response)

    return output_batch_audios, output_batch_responses
