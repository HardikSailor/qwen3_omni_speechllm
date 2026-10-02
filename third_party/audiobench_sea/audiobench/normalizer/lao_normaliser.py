"""
ASR Normalizer for Lao (lo)
Uses the unified base normalizer to eliminate code duplication.
"""

from typing import Optional
from .base_normalizer import BaseASRNormalizer, ASRConfig


class LaoASRNormaliser(BaseASRNormalizer):
    """Lao text normalizer for ASR evaluation."""
    
    NATIVE_DIGIT_MAP = {
        '໐': '0', '໑': '1', '໒': '2', '໓': '3', '໔': '4', 
        '໕': '5', '໖': '6', '໗': '7', '໘': '8', '໙': '9'
    }
    EXTRA_PUNCTUATION = "«»""''•·…฿₭ໆ–—،؛‹›"
    FILLER_WORDS = ['ອ່າ', 'ອື່', 'ແບບ']
    IS_CASE_SENSITIVE = False
    LANGUAGE_CODE = "lo"
    LANGUAGE_NAME = "Lao"

    def _language_specific_normalization(self, text: str) -> str:
        """Lao-specific normalization steps."""
        # Add any Lao-specific normalization here
        return text


def get_normalizer(lang: str, config: Optional[ASRConfig] = None) -> LaoASRNormaliser:
    """Factory function for backward compatibility."""
    lang = (lang or "").lower()
    if lang in ['lo', 'lao']:
        return LaoASRNormaliser(config)
    raise ValueError(f"Unsupported language code for this module: {lang!r}")