# -*- coding: utf-8 -*-
"""
Filipino/Tagalog ASR normalizer using the unified preprocessing system.
- Uses preprocess_text.py for base cleaning
- Adds Filipino-specific normalization
- Normalizes units/currency (GB/MB → lowercase; PHP/₱ → "peso")
- Collapses phone numbers and standardizes common +63 formats

Usage:
    from filipino_normaliser import FilipinoASRNormalizer, FilipinoNormalizerConfig
    norm = FilipinoASRNormalizer()
    print(norm.normalize("<spk2> Ano ba, 2GB for ₱199 — tawagan mo ako sa +63 912-345-6789 [noise]"))
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import List

# Import base normalizer and preprocessing functions
from .base_normalizer import BaseASRNormalizer, ASRConfig
from .text_normalizer.preprocess_text import preprocess_text_asr_filipino

__all__ = [
    "FilipinoASRNormalizer",
    "FilipinoNormalizerConfig",
]

logger = logging.getLogger(__name__)


@dataclass
class FilipinoNormalizerConfig(ASRConfig):
    """Filipino-specific configuration extending base ASR config."""
    normalize_currency_units: bool = True  # PHP/₱ → "peso", GB/MB → lowercase tokens
    normalize_phone_numbers: bool = True   # Standardize +63 formats


class FilipinoASRNormalizer(BaseASRNormalizer):
    """
    Filipino/Tagalog ASR normalizer using the unified preprocessing system.
    """
    
    LANGUAGE_CODE = "tl"
    LANGUAGE_NAME = "Filipino"
    
    # Filipino-specific filler words
    FILLER_WORDS = [
        "ano", "ano ba", "ano nga", "ano yun", "ano yung",
        "eh", "eh ano", "eh di", "eh kasi", "eh paano",
        "kasi", "kasi nga", "kasi naman",
        "nga", "nga naman", "nga pala",
        "naman", "naman eh", "naman kasi",
        "paano", "paano ba", "paano nga",
        "tuloy", "tuloy nga", "tuloy naman",
        "yun", "yung", "yung nga", "yung naman",
        "ba", "ba naman", "ba nga",
        "eh", "eh ba", "eh nga", "eh naman",
        "lang", "lang naman", "lang nga",
        "din", "din naman", "din nga",
        "rin", "rin naman", "rin nga",
    ]
    
    def __init__(self, config: FilipinoNormalizerConfig | None = None, enable_disfluency_removal: bool = False):
        super().__init__(config or FilipinoNormalizerConfig(), enable_disfluency_removal)
        self._compile_filipino_patterns()

    def _compile_filipino_patterns(self):
        """Compile Filipino-specific patterns."""
        # Units/currency
        self._unit_map = {
            "GB": "gb", "Mb": "mb", "MB": "mb", "TB": "tb", "KB": "kb",
            "gb": "gb", "mb": "mb", "tb": "tb", "kb": "kb",
        }
        self._unit_res = [
            (re.compile(rf"(\d+(?:\.\d+)?)\s*{re.escape(u)}\b"), t)
            for u, t in self._unit_map.items()
        ]
        self._php_re = re.compile(r"\b(PHP)\b", re.IGNORECASE)
        self._peso_symbol_re = re.compile(r"[₱]")

        # Phone numbers (PH 10-11 digits; allow +63)
        self._phone_re = re.compile(r"\b(?:\+63[\s-]?)?(?:\d[\s-]?){10,11}\b")

    def _normalize_currency_units(self, text: str) -> str:
        """Normalize Filipino currency and units."""
        if hasattr(self.config, 'normalize_currency_units') and self.config.normalize_currency_units:
            for rx, tgt in self._unit_res:
                text = rx.sub(lambda m: f"{m.group(1)} {tgt}", text)
            text = self._php_re.sub("peso", text)
            text = self._peso_symbol_re.sub(" peso ", text)
        return text

    def _normalize_phone(self, text: str) -> str:
        """Normalize Filipino phone numbers."""
        def repl(m):
            digits = re.sub(r"\D", "", m.group(0))
            if 10 <= len(digits) <= 11:
                return digits
            return m.group(0)
        return self._phone_re.sub(repl, text)

    def _language_specific_normalization(self, text: str) -> str:
        """Apply Filipino-specific normalization."""
        # Apply currency and unit normalization
        text = self._normalize_currency_units(text)
        
        # Apply phone number normalization if enabled
        if hasattr(self.config, 'normalize_phone_numbers') and self.config.normalize_phone_numbers:
            text = self._normalize_phone(text)
        
        return text
