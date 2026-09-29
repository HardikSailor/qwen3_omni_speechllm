try:
    from jiwer import compute_measures
except ImportError:  # jiwer >= 4 removed compute_measures (omni_mds patch, see VENDORED.md)
    import jiwer

    def compute_measures(truth, hypothesis):
        out = jiwer.process_words(truth, hypothesis)
        return {"substitutions": out.substitutions, "deletions": out.deletions,
                "insertions": out.insertions, "hits": out.hits}

from ..text_normalizers.wer_text_normalizer.normalizer import preprocess_text_asr


def compute_wer(predictions=None, references=None, dataset="default", target_lang="english", **kwargs):

    incorrect = 0
    total = 0
    for prediction, reference in zip(predictions, references):
        truth = preprocess_text_asr(reference, dataset=dataset, target_lang=target_lang)
        hypothesis = preprocess_text_asr(prediction, dataset=dataset, target_lang=target_lang)

        if truth:
            measures = compute_measures(
                truth=truth,
                hypothesis=hypothesis,
            )

            incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
            total += measures["substitutions"] + measures["deletions"] + measures["hits"]

    return incorrect / total if total > 0 else float("inf")


compute_wer.__metric__ = "wer"
compute_wer.__greater_is_better__ = False


