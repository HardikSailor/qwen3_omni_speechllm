# -*- coding: utf-8 -*-
"""
Vietnamese ASR normalizer using the unified preprocessing system.
- Uses preprocess_text.py for base cleaning
- Adds Vietnamese-specific normalization
- Normalizes units/currency (GB/MB → lowercase; VND/₫ → "đồng")
- Collapses phone numbers
- Optional diacritic stripping (off by default)

Usage:
    from vietnamese_normaliser import VietnameseASRNormalizer, VietnameseNormalizerConfig
    norm = VietnameseASRNormalizer(VietnameseNormalizerConfig(strip_diacritics=False))
    print(norm.normalize("<Speaker1>: Ờm… 1.5GB dữ liệu giá 200,000₫ [nhạc]"))
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import List

# Import base normalizer and preprocessing functions
from .base_normalizer import BaseASRNormalizer, ASRConfig
from .text_normalizer.preprocess_text import preprocess_text_asr_vietnamese

__all__ = [
    "VietnameseASRNormalizer",
    "VietnameseNormalizerConfig",
]

logger = logging.getLogger(__name__)


@dataclass
class VietnameseNormalizerConfig(ASRConfig):
    """Vietnamese-specific configuration extending base ASR config."""
    normalize_currency_units: bool = True  # ₫/VND → "đồng", GB/MB → lowercase tokens
    strip_diacritics: bool = False         # keep accents by default


class VietnameseASRNormalizer(BaseASRNormalizer):
    """
    Vietnamese ASR normalizer using the unified preprocessing system.
    """
    
    LANGUAGE_CODE = "vi"
    LANGUAGE_NAME = "Vietnamese"
    
    # Vietnamese-specific filler words
    FILLER_WORDS = [
        "ờ", "ừ", "ừm", "ừhm", "ờm", "ờ hờ", "à", "ạ",
        "kiểu như", "nói chung", "kiểu", "kiểu là", "kiểu như là",
        "kiểu ấy", "kiểu vậy", "kiểu vầy",
        "ý là", "thì", "ờ thì",
        "ơ", "ờ ờ", "ừ ừ",
    ]
    
    def __init__(self, config: VietnameseNormalizerConfig | None = None, enable_disfluency_removal: bool = False):
        super().__init__(config or VietnameseNormalizerConfig(), enable_disfluency_removal)
        self._compile_vietnamese_patterns()

    def _compile_vietnamese_patterns(self):
        """Compile Vietnamese-specific patterns."""
        # Units/currency
        self._unit_map = {
            "GB": "gb", "Mb": "mb", "MB": "mb", "TB": "tb", "KB": "kb",
            "gb": "gb", "mb": "mb", "tb": "tb", "kb": "kb",
        }
        self._unit_res = [
            (re.compile(rf"(\d+(?:\.\d+)?)\s*{re.escape(u)}\b"), t)
            for u, t in self._unit_map.items()
        ]
        self._vnd_re = re.compile(r"\b(VND)\b", re.IGNORECASE)
        self._dong_symbol_re = re.compile(r"[₫]")

        # Phone numbers (VN 9–11 digits; allow +84)
        self._phone_re = re.compile(r"\b(?:\+84[\s-]?)?(?:\d[\s-]?){9,12}\b")

    def _normalize_currency_units(self, text: str) -> str:
        """Normalize Vietnamese currency and units."""
        if hasattr(self.config, 'normalize_currency_units') and self.config.normalize_currency_units:
            for rx, tgt in self._unit_res:
                text = rx.sub(lambda m: f"{m.group(1)} {tgt}", text)
            text = self._vnd_re.sub("đồng", text)
            text = self._dong_symbol_re.sub(" đồng ", text)
        return text

    def _normalize_phone(self, text: str) -> str:
        """Normalize Vietnamese phone numbers."""
        def repl(m):
            digits = re.sub(r"\D", "", m.group(0))
            if 9 <= len(digits) <= 11:
                return digits
            return m.group(0)
        return self._phone_re.sub(repl, text)

    def _strip_diacritics_if_needed(self, text: str) -> str:
        """Strip diacritics if configured."""
        if hasattr(self.config, 'strip_diacritics') and self.config.strip_diacritics:
            nfd = unicodedata.normalize("NFD", text)
            no_marks = "".join(ch for ch in nfd if not unicodedata.combining(ch))
            return unicodedata.normalize("NFC", no_marks)
        return text

    def _language_specific_normalization(self, text: str) -> str:
        """Apply Vietnamese-specific normalization."""
        # Apply currency and unit normalization
        text = self._normalize_currency_units(text)
        text = self._normalize_phone(text)
        
        # Apply diacritic stripping if configured
        text = self._strip_diacritics_if_needed(text)
        
        return text
