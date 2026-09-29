from functools import partial

from .bleu import compute_bleu
from .meteor import compute_meteor
from .wer import compute_wer


def get_metric_fn(dataset_path):
    if "ASR/" in dataset_path:
        if "gigaspeech" in dataset_path.lower():
            dataset = "gigaspeech"
            target_lang = "en"
        elif "imda" in dataset_path.lower():
            dataset = "imda"
            target_lang = "en"
            if "part4" in dataset_path.lower():
                target_lang = "en_zh_ms_ta"
        elif "aishell" in dataset_path.lower():
            dataset = "aishell"
            target_lang = "zh"
        else:
            dataset = "default"
            target_lang = "en"

        compute_wer_partial = partial(compute_wer, dataset=dataset, target_lang=target_lang)
        compute_wer_partial.__metric__ = compute_wer.__metric__
        compute_wer_partial.__greater_is_better__ = compute_wer.__greater_is_better__

        return compute_wer_partial
    elif "ST/" in dataset_path:
        if "en_zh" in dataset_path:
            compute_bleu_partial = partial(compute_bleu, target_language="zh")
            compute_bleu_partial.__metric__ = compute_bleu.__metric__
            compute_bleu_partial.__greater_is_better__ = compute_bleu.__greater_is_better__
            return compute_bleu_partial
        else:
            return compute_bleu
    elif "AC/" in dataset_path:
        return compute_meteor
    elif "AQA/" in dataset_path:
        return compute_meteor            
    else:
        return None
