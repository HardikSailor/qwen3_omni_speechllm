# asr_metrics.py
"""
ASR-specific metrics for AudioBench-SEA.
Includes Word Error Rate (WER) and Character Error Rate (CER) metrics.

Uses the unified preprocessing system from the normalizer package to eliminate
duplications and ensure consistent text cleaning across the codebase.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# Import unified preprocessing functions
from ..normalizer.text_normalizer.preprocess_text import (
    preprocess_text_generic, PreprocessConfig, clean_asr_prediction
)

# These come from your existing codebase
from .base import BaseMetric, JIWER_AVAILABLE, _measures_to_dict, _extract_counts_from_measures

if JIWER_AVAILABLE:
    # imported in .base normally; if not, uncomment next line
    # from jiwer import compute_measures
    from jiwer import compute_measures  # type: ignore

logger = logging.getLogger(__name__)

# ========== Unified Cleaning Utilities ==========

def clean_text_for_metric(text: str, language_code: str = "en") -> str:
    """
    Preprocess text for WER/CER using the unified preprocessing system.
    
    Args:
        text: Input text to clean
        language_code: Language code for language-specific preprocessing
        
    Returns:
        Cleaned text ready for metric computation
    """
    if not text:
        return ""

    # Use the unified preprocessing system with metric-specific configuration
    preprocess_config = PreprocessConfig(
        to_lower=True,                    # Lowercase for consistent comparison
        remove_punctuation=True,          # Remove punctuation for metric computation
        keep_inword_hyphen_apostrophe=False,  # Remove ALL punctuation including apostrophes in words
        remove_speaker_tags=True,         # Remove speaker tags
        remove_bracketed=True,            # Remove bracketed content
        remove_nonspeech_meta=True,       # Remove non-speech elements
        remove_urls_emails=True,          # Remove URLs and emails
        normalize_phones=True,            # Normalize phone numbers
        collapse_whitespace=True,         # Collapse multiple spaces
        strip_edges=True,                 # Strip leading/trailing whitespace
        use_jiwer=False,                  # Don't use jiwer here (we'll use it in metric computation)
        debug=False
    )
    
    # Apply unified preprocessing
    cleaned_text = preprocess_text_generic(text, preprocess_config)
    
    return cleaned_text


# ========== WER ==========

class WERMetric(BaseMetric):
    """Word Error Rate metric for ASR tasks with language-specific normalization using jiwer"""

    def __init__(self, language: str = "en"):
        super().__init__("WER", f"Word Error Rate ({language})")
        self.language = str(language).lower() if language else "en"

    def compute(self, predictions: List[str], references: List[str],
                questions: Optional[List[str]] = None) -> Tuple[float, Dict[str, Any]]:
        if not isinstance(predictions, list) or not isinstance(references, list):
            raise ValueError("predictions and references must be lists")

        if not predictions or not references:
            logger.warning("Empty predictions or references provided")
            return 0.0, {
                "S": 0, "I": 0, "D": 0, "N": 0,
                "total_errors": 0, "total_words": 0,
                "sample_scores": [], "std_dev": 0.0,
                "language": self.language,
                "warning": "Empty input data"
            }

        total_S = total_I = total_D = total_N = 0
        sample_scores: List[float] = []
        sample_details: List[Dict[str, Any]] = []
        processed_count = 0

        n = min(len(predictions), len(references))

        for idx in range(n):
            try:
                raw_pred = "" if predictions[idx] is None else str(predictions[idx])
                raw_ref  = "" if references[idx]   is None else str(references[idx])

                pred = clean_text_for_metric(raw_pred, self.language)
                ref  = clean_text_for_metric(raw_ref, self.language)

                # Word tokens (whitespace-delimited). For Chinese, this may be 1 token per line,
                # but it's consistent with a language-agnostic baseline.
                ref_words  = ref.split()
                pred_words = pred.split()

                if not ref.strip() and not pred.strip():
                    sample_scores.append(0.0)
                    sample_details.append({
                        "sample_id": idx,
                        "reference": raw_ref,
                        "prediction": raw_pred,
                        "normalized_reference": ref,
                        "normalized_prediction": pred,
                        "wer": 0.0,
                        "S": 0, "I": 0, "D": 0, "N": 0,
                        "method": "empty_input",
                        "ref_words": ref_words,
                        "pred_words": pred_words,
                    })
                    continue

                has_measures = False
                measures_dict: Dict[str, Any] = {}
                if JIWER_AVAILABLE:
                    try:
                        m = compute_measures(ref, pred)
                        measures_dict = _measures_to_dict(m)
                        has_measures = bool(measures_dict)
                    except Exception as jiwer_error:
                        logger.debug(f"jiwer failed on sample {idx}: {jiwer_error}")

                if has_measures:
                    S, I, D, N = _extract_counts_from_measures(
                        measures_dict, ref_len_fallback=len(ref_words)
                    )
                    sample_wer = float(measures_dict.get("wer", (S + I + D) / N if N > 0 else 0.0))
                    method = "jiwer"
                else:
                    # Simple fallback approximation at word level
                    N = len(ref_words)
                    S = I = D = 0
                    if N > 0:
                        min_len = min(len(ref_words), len(pred_words))
                        for i in range(min_len):
                            if ref_words[i] != pred_words[i]:
                                S += 1
                        if len(pred_words) > len(ref_words):
                            I = len(pred_words) - len(ref_words)
                        elif len(ref_words) > len(pred_words):
                            D = len(ref_words) - len(pred_words)
                    sample_wer = (S + I + D) / N if N > 0 else 0.0
                    method = "fallback"

                total_S += S
                total_I += I
                total_D += D
                total_N += N
                sample_scores.append(float(sample_wer))

                sample_details.append({
                    "sample_id": idx,
                    "reference": raw_ref,
                    "prediction": raw_pred,
                    "normalized_reference": ref,
                    "normalized_prediction": pred,
                    "wer": float(sample_wer),
                    "S": S, "I": I, "D": D, "N": N,
                    "method": method,
                    "ref_words": ref_words,
                    "pred_words": pred_words,
                })
                processed_count += 1

            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                logger.debug(f"Sample {idx} details - pred: '{predictions[idx]}', ref: '{references[idx]}'")
                sample_scores.append(0.0)
                continue

        if processed_count == 0:
            logger.warning("No samples were processed successfully")
            return 0.0, {
                "S": 0, "I": 0, "D": 0, "N": 0,
                "total_errors": 0, "total_words": 0,
                "sample_scores": [], "std_dev": 0.0,
                "language": self.language,
                "warning": "No samples processed successfully"
            }

        overall_wer = (total_S + total_I + total_D) / total_N if total_N > 0 else 0.0
        details = {
            "S": total_S, "I": total_I, "D": total_D, "N": total_N,
            "total_errors": total_S + total_I + total_D,
            "total_words": total_N,
            "sample_scores": sample_scores,
            "std_dev": float(np.std(sample_scores)) if sample_scores else 0.0,
            "language": self.language,
            "processed_samples": processed_count,
            "total_samples": n,
            "method": "jiwer" if JIWER_AVAILABLE else "fallback",
            "sample_details": sample_details,
        }
        logger.info(f"WER computation completed: {processed_count}/{n} samples processed successfully")
        return float(overall_wer), details


# ========== CER ==========

class CERMetric(BaseMetric):
    """Character Error Rate metric for ASR tasks with language-specific normalization using jiwer"""

    def __init__(self, language: str = "en"):
        super().__init__("CER", f"Character Error Rate ({language})")
        self.language = str(language).lower() if language else "en"

    def compute(self, predictions: List[str], references: List[str],
                questions: Optional[List[str]] = None) -> Tuple[float, Dict[str, Any]]:
        if not isinstance(predictions, list) or not isinstance(references, list):
            raise ValueError("predictions and references must be lists")

        if not predictions or not references:
            logger.warning("Empty predictions or references provided")
            return 0.0, {
                "S": 0, "I": 0, "D": 0, "N": 0,
                "total_errors": 0, "total_chars": 0,
                "sample_scores": [], "std_dev": 0.0,
                "language": self.language,
                "warning": "Empty input data"
            }

        total_S = total_I = total_D = total_N = 0
        sample_scores: List[float] = []
        sample_details: List[Dict[str, Any]] = []
        processed_count = 0

        n = min(len(predictions), len(references))

        for idx in range(n):
            try:
                raw_pred = "" if predictions[idx] is None else str(predictions[idx])
                raw_ref  = "" if references[idx]   is None else str(references[idx])

                pred = clean_text_for_metric(raw_pred, self.language)
                ref  = clean_text_for_metric(raw_ref, self.language)

                if not ref.strip() and not pred.strip():
                    sample_scores.append(0.0)
                    sample_details.append({
                        "sample_id": idx,
                        "reference": raw_ref,
                        "prediction": raw_pred,
                        "normalized_reference": ref,
                        "normalized_prediction": pred,
                        "cer": 0.0,
                        "S": 0, "I": 0, "D": 0, "N": 0,
                        "method": "empty_input"
                    })
                    continue

                # Tokenize to char-level with spaces so jiwer can operate at character granularity
                ref_chars = " ".join(list(ref))
                pred_chars = " ".join(list(pred))

                has_measures = False
                measures_dict: Dict[str, Any] = {}
                if JIWER_AVAILABLE:
                    try:
                        m = compute_measures(ref_chars, pred_chars)
                        measures_dict = _measures_to_dict(m)
                        has_measures = bool(measures_dict)
                    except Exception as jiwer_error:
                        logger.debug(f"jiwer failed on sample {idx}: {jiwer_error}")

                if has_measures:
                    S, I, D, N = _extract_counts_from_measures(
                        measures_dict, ref_len_fallback=len(ref)
                    )
                    sample_cer = float(measures_dict.get("wer", (S + I + D) / N if N > 0 else 0.0))
                    method = "jiwer"
                else:
                    # Fallback approximate char-level edit distance (position-wise mismatch + length diff)
                    N = len(ref)
                    S = I = D = 0
                    if N > 0:
                        min_len = min(len(ref), len(pred))
                        for i in range(min_len):
                            if ref[i] != pred[i]:
                                S += 1
                        if len(pred) > len(ref):
                            I = len(pred) - len(ref)
                        elif len(ref) > len(pred):
                            D = len(ref) - len(pred)
                    sample_cer = (S + I + D) / N if N > 0 else 0.0
                    method = "fallback"

                total_S += S
                total_I += I
                total_D += D
                total_N += N
                sample_scores.append(float(sample_cer))

                sample_details.append({
                    "sample_id": idx,
                    "reference": raw_ref,
                    "prediction": raw_pred,
                    "normalized_reference": ref,
                    "normalized_prediction": pred,
                    "cer": float(sample_cer),
                    "S": S, "I": I, "D": D, "N": N,
                    "method": method
                })
                processed_count += 1

            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                logger.debug(f"Sample {idx} details - pred: '{predictions[idx]}', ref: '{references[idx]}'")
                sample_scores.append(0.0)
                continue

        if processed_count == 0:
            logger.warning("No samples were processed successfully")
            return 0.0, {
                "S": 0, "I": 0, "D": 0, "N": 0,
                "total_errors": 0, "total_chars": 0,
                "sample_scores": [], "std_dev": 0.0,
                "language": self.language,
                "warning": "No samples processed successfully"
            }

        overall_cer = (total_S + total_I + total_D) / total_N if total_N > 0 else 0.0
        details = {
            "S": total_S, "I": total_I, "D": total_D, "N": total_N,
            "total_errors": total_S + total_I + total_D,
            "total_chars": total_N,
            "sample_scores": sample_scores,
            "std_dev": float(np.std(sample_scores)) if sample_scores else 0.0,
            "language": self.language,
            "processed_samples": processed_count,
            "total_samples": n,
            "method": "jiwer" if JIWER_AVAILABLE else "fallback",
            "sample_details": sample_details
        }
        logger.info(f"CER computation completed: {processed_count}/{n} samples processed successfully")
        return float(overall_cer), details
