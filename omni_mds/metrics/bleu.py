import os
import evaluate


def compute_bleu(predictions=None, references=None, target_language="en", experiment_id=None):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sacrebleu = evaluate.load(os.path.join(script_dir, "hf_metrics/sacrebleu/sacrebleu.py"), experiment_id=experiment_id)

    if target_language == "zh":
        tokenize = "zh"
    else:
        tokenize = "13a"
    results = sacrebleu.compute(predictions=predictions, references=references, tokenize=tokenize)

    return results["score"] / 100


compute_bleu.__metric__ = "bleu"
compute_bleu.__greater_is_better__ = True
