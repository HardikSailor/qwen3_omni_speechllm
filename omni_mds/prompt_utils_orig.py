import random
import unicodedata
import warnings

from .instructions_lib import get_instructions_list

"""
prompt_list = [
    "Given the following audio context: <SpeechHere>\n\nText instruction:",
    "Given the following audio context: <SpeechHere>\n\nPlease answer this question:",
    "Listen to the following audio snippet: <SpeechHere>\n\nNow, based on this audio, follow the instruction:",
    "Here's some audio context: <SpeechHere>\n\nPlease respond to the text instruction accordingly:",
    "Audio content: <SpeechHere>\n\nYour task is to interpret and respond according to the instruction below:",
    "Given the speech below: <SpeechHere>\n\nCarry out the following instruction:",
    "Consider this audio: <SpeechHere>\n\nNow respond to the following instruction:",
    "Audio content: <SpeechHere>\n\nYour task is to interpret and respond according to the instruction below:",
    "Refer to the following audio message: <SpeechHere>\n\nThen, perform the instruction:",
    "The following is an audio segment: <SpeechHere>\n\nUsing this as context, handle the instruction:",
    "Take the following audio into account: <SpeechHere>\n\nNow, respond to the instruction:",
]
"""

prompt_list = [
    "Given the following audio context: <SpeechHere>\n\nText instruction: <TextHere>",
]


def contains_punctuation(text):
    """
    Checks if the given string contains any punctuation character.
    Works with non-English (Unicode) strings as well.

    Args:
        text (str): Input string.

    Returns:
        bool: True if punctuation is found, False otherwise.
    """
    for char in text:
        if unicodedata.category(char).startswith("P"):
            return True
    return False


def is_all_upper(text):
    """
    Check if all alphabetic characters in the string are uppercase.
    Works with multilingual strings using Unicode properties.
    """
    has_letters = False
    for char in text:
        if char.isalpha():
            has_letters = True
            if not char.isupper():
                return False
    return has_letters


def is_all_lower(text):
    """
    Check if all alphabetic characters in the string are lowercase.
    Works with multilingual strings using Unicode properties.
    """
    has_letters = False
    for char in text:
        if char.isalpha():
            has_letters = True
            if not char.islower():
                return False
    return has_letters


def _get_prompt(
    prompt_type,
    dataset_instruction,
    lang_code,
    task,
    eng_weights=0.5,
    response_text=None,
    augment_prompt=True,
):

    if prompt_type is None:
        # no instruction prompt mode
        return ""
    elif prompt_type.lower() == "dataset":
        return dataset_instruction
    else:
        if prompt_type.lower() == "random_multilingual":
            rand_lang_code = random.choices(
                ["en", lang_code], weights=(eng_weights, 1 - eng_weights), k=1
            )[0]
        else:
            rand_lang_code = "en"

        instructions = get_instructions_list(lang_code=rand_lang_code)
        if task not in instructions:
            raise (f"{task} not in instructions list!")

        # get a random prompt from instruction list
        prompt = random.choice(instructions[task])

        if augment_prompt and lang_code.lower() in ["vi"]:
            if response_text is None:
                warnings.warn(
                    "response_text field must not be None when augment_prompt=True. Setting augment_prompt=False..."
                )
            else:
                prompt += random.choice(["\n", " "])
                additional_prompt_list = []

                # add additional prompts based on training data
                if not contains_punctuation(response_text):
                    additional_prompt_list.append(
                        random.choice(instructions["no_punctuation"])
                    )
                if is_all_lower(response_text):
                    additional_prompt_list.append(
                        random.choice(instructions["lowercase"])
                    )
                elif is_all_upper(response_text):
                    additional_prompt_list.append(
                        random.choice(instructions["uppercase"])
                    )

                for prompt_index, additional_prompt in enumerate(
                    random.sample(
                        additional_prompt_list, len(additional_prompt_list)
                    )
                ):
                    if prompt_index == len(additional_prompt_list) - 1:
                        delimiters = [".", "\n", "\n\n", " "]
                    else:
                        delimiters = [".", ",", "\n", " "]

                    prompt += additional_prompt + random.choice(delimiters)
        return prompt


def get_query_prompt(query_prompt):
    return prompt_list if query_prompt is None else [query_prompt]


def form_mm_prompt(text_context, has_audio, prompt):
    prompt_suffix = f"Instruction:\n{prompt}"
    dummy_audio_placeholder = "<SpeechHere>"

    if (not has_audio) and (not text_context):
        return dummy_audio_placeholder + prompt_suffix

    prefix = f"Given the following information:\n"

    if text_context:
        prefix += f"{text_context}\n\n"

    if has_audio:
        prefix += f"{dummy_audio_placeholder}\n\n"
    else:
        prefix += dummy_audio_placeholder

    return prefix + prompt_suffix


def get_asr_prompt(prompt_type, dataset_instruction, lang_code="en", **kwargs):
    return _get_prompt(
        prompt_type=prompt_type,
        dataset_instruction=dataset_instruction,
        lang_code=lang_code,
        task="asr",
        **kwargs,
    )


def get_ac_prompt(prompt_type, dataset_instruction, lang_code="en", **kwargs):
    return _get_prompt(
        prompt_type=prompt_type,
        dataset_instruction=dataset_instruction,
        lang_code=lang_code,
        task="ac",
        **kwargs,
    )
