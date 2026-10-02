"""
ASR Normalizer for Khmer (km)
Uses the unified base normalizer to eliminate code duplication.
"""

from typing import Optional
from .base_normalizer import BaseASRNormalizer, ASRConfig


class KhmerASRNormaliser(BaseASRNormalizer):
    """Khmer text normalizer for ASR evaluation."""
    
    NATIVE_DIGIT_MAP = {
        '០': '0', '១': '1', '២': '2', '៣': '3', '៤': '4', 
        '៥': '5', '៦': '6', '៧': '7', '៨': '8', '៩': '9'
    }
    EXTRA_PUNCTUATION = "«»""''។៕៖៍៙៚–—•·…៛฿,"
    FILLER_WORDS = ['អ៊ា', 'អឺ', 'អ៊ុំ']
    IS_CASE_SENSITIVE = False
    LANGUAGE_CODE = "km"
    LANGUAGE_NAME = "Khmer"

    def _language_specific_normalization(self, text: str) -> str:
        """Khmer-specific normalization steps."""
        # Add any Khmer-specific normalization here
        return text


def get_normalizer(lang: str, config: Optional[ASRConfig] = None) -> KhmerASRNormaliser:
    """Factory function for backward compatibility."""
    lang = (lang or "").lower()
    if lang in ['km', 'khmer']:
        return KhmerASRNormaliser(config)
    raise ValueError(f"Unsupported language code for this module: {lang!r}")
