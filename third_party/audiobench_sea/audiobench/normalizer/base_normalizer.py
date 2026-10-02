#!/usr/bin/env python3
"""
Unified Base Normalizer for AudioBench-SEA
Provides a common foundation for all language-specific normalizers.
Eliminates code duplication and ensures consistency.

This module provides:
- BaseASRNormalizer: Abstract base class for all normalizers
- ASRConfig: Configuration dataclass for normalization settings
- UniversalASRNormalizer: Fallback normalizer for unsupported languages
- create_normalizer: Factory function for creating language-specific normalizers
"""

from __future__ import annotations
import re
import unicodedata
import string
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any
from abc import ABC, abstractmethod

# Import preprocessing functions from text_normalizer
from .text_normalizer.preprocess_text import (
    preprocess_text_generic, PreprocessConfig, preprocess_text_asr_base,
    preprocess_text_asr_universal, clean_asr_prediction
)

logger = logging.getLogger(__name__)

@dataclass
class ASRConfig:
    """
    Configuration for ASR text normalization.
    
    Attributes:
        preserve_decimal_points: Whether to preserve decimal points (.)
        preserve_percentages: Whether to preserve percentage symbols (%)
        remove_punctuation: Whether to remove punctuation marks
        normalize_whitespace: Whether to normalize whitespace (collapse multiple spaces)
        lowercase_output: Whether to convert output to lowercase
        enable_debug_logging: Whether to enable debug logging
        strict_mode: Whether to use strict mode (truncate long texts)
        max_text_length: Maximum text length in strict mode
        remove_filler_words: Whether to remove filler words
        enable_disfluency_removal: Whether to enable disfluency removal
        preserve_currency: Whether to preserve currency symbols
        preserve_numbers: Whether to preserve number formatting
    """
    preserve_decimal_points: bool = True
    preserve_percentages: bool = True
    remove_punctuation: bool = True
    normalize_whitespace: bool = True
    lowercase_output: bool = False
    enable_debug_logging: bool = False
    strict_mode: bool = False
    max_text_length: int = 500_000
    remove_filler_words: bool = True
    enable_disfluency_removal: bool = False
    preserve_currency: bool = False
    preserve_numbers: bool = True


class BaseASRNormalizer(ABC):
    """
    Base class for all ASR normalizers.
    Provides common functionality and ensures consistent interface.
    
    Subclasses must implement:
    - _language_specific_normalization(): Language-specific normalization logic
    - Set class attributes: NATIVE_DIGIT_MAP, EXTRA_PUNCTUATION, FILLER_WORDS, etc.
    """
    
    # Language-specific attributes to be overridden by subclasses
    NATIVE_DIGIT_MAP: Dict[str, str] = {}
    EXTRA_PUNCTUATION: str = ""
    FILLER_WORDS: List[str] = []
    IS_CASE_SENSITIVE: bool = False
    LANGUAGE_CODE: str = ""
    LANGUAGE_NAME: str = ""

    def __init__(self, config: Optional[ASRConfig] = None, enable_disfluency_removal: bool = False):
        """
        Initialize the normalizer with configuration.
        
        Args:
            config: Configuration object for normalization settings
            enable_disfluency_removal: Whether to enable disfluency removal
        """
        self.config = config or ASRConfig()
        self.config.enable_disfluency_removal = enable_disfluency_removal
        
        # Build punctuation pattern
        self._build_punctuation_pattern()
        
        # Build filler words pattern
        self._build_filler_pattern()
        
        # Zero-width characters pattern
        self.zero_width_re = re.compile("[\u200B\u200C\u200D\u2060]")
        
        if self.config.enable_debug_logging:
            logger.debug(f"Initialized {self.__class__.__name__} for {self.LANGUAGE_NAME}")
    
    def _build_punctuation_pattern(self):
        """Build the punctuation removal pattern based on configuration."""
        base_punct = string.punctuation
        extra_punct = self.EXTRA_PUNCTUATION
        preserved = ""
        
        if self.config.preserve_decimal_points:
            preserved += "."
        if self.config.preserve_percentages:
            preserved += "%"
        if self.config.preserve_currency:
            preserved += "$€£¥₹₱₫"
            
        for ch in preserved:
            base_punct = base_punct.replace(ch, "")
            extra_punct = extra_punct.replace(ch, "")
            
        self.punct_pattern = re.compile(f"[{re.escape(base_punct + extra_punct)}]+")
    
    def _build_filler_pattern(self):
        """Build the filler words removal pattern."""
        if self.FILLER_WORDS:
            words = [re.escape(w) for w in self.FILLER_WORDS]
            if self.IS_CASE_SENSITIVE:
                self.filler_re = re.compile(rf"\b({'|'.join(words)})\b")
            else:
                self.filler_re = re.compile(rf"(?i)\b({'|'.join(words)})\b")
        else:
            self.filler_re = None

    def _unicode_normalize(self, text: str) -> str:
        """Normalize Unicode characters."""
        return unicodedata.normalize("NFC", text)

    def _strip_zero_width(self, text: str) -> str:
        """Remove zero-width characters."""
        return self.zero_width_re.sub("", text)

    def _normalize_digits(self, text: str) -> str:
        """Convert native digits to ASCII digits."""
        if not self.NATIVE_DIGIT_MAP:
            return text
        return "".join(self.NATIVE_DIGIT_MAP.get(ch, ch) for ch in text)

    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation based on configuration."""
        if not self.config.remove_punctuation:
            return text
        return self.punct_pattern.sub(" ", text)

    def _remove_fillers(self, text: str) -> str:
        """Remove filler words if enabled."""
        if not (self.config.remove_filler_words and self.filler_re):
            return text
        return self.filler_re.sub(" ", text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        if not self.config.normalize_whitespace:
            return text.strip()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _final_case(self, text: str) -> str:
        """Apply final case transformation."""
        if self.config.lowercase_output:
            return text.lower()
        return text

    @abstractmethod
    def _language_specific_normalization(self, text: str) -> str:
        """Language-specific normalization steps. Must be implemented by subclasses."""
        pass

    def normalize(self, text: str) -> str:
        """
        Main normalization method.
        Applies preprocessing first, then language-specific normalization.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            text = str(text)
            
        if not text:
            return ""
            
        if self.config.strict_mode and len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]
            logger.warning(f"Text truncated to {self.config.max_text_length} characters")

        try:
            # Step 1: Apply base preprocessing using preprocess_text.py
            preprocess_config = PreprocessConfig(
                to_lower=self.config.lowercase_output,
                remove_punctuation=self.config.remove_punctuation,
                collapse_whitespace=self.config.normalize_whitespace,
                strip_edges=True,
                debug=self.config.enable_debug_logging
            )
            text = preprocess_text_generic(text, preprocess_config)
            
            # Step 2: Apply language-specific normalization
            text = self._language_specific_normalization(text)
            
            # Step 3: Apply additional normalizer-specific processing
            text = self._normalize_digits(text)
            text = self._remove_fillers(text)
            text = self._strip_zero_width(text)
            
            # Step 4: Final processing
            text = self._normalize_whitespace(text)
            text = self._final_case(text)
            
            return text
            
        except Exception as e:
            logger.error(f"Normalization failed for {self.LANGUAGE_NAME}: {e}")
            # Fallback to basic preprocessing
            try:
                return preprocess_text_generic(text, PreprocessConfig())
            except Exception:
                return re.sub(r'\s+', ' ', str(text).strip())

    def normalize_batch(self, texts: List[str]) -> List[str]:
        """
        Normalize a batch of texts.
        
        Args:
            texts: List of texts to normalize
            
        Returns:
            List of normalized texts
        """
        return [self.normalize(text) for text in texts]

    def normalize_for_asr_eval(self, text: str) -> str:
        """
        Alias for normalize method for ASR evaluation compatibility.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        return self.normalize(text)

    def get_language_info(self) -> Dict[str, str]:
        """
        Get language information.
        
        Returns:
            Dictionary with language code, name, and class name
        """
        return {
            'code': self.LANGUAGE_CODE,
            'name': self.LANGUAGE_NAME,
            'class': self.__class__.__name__
        }


class UniversalASRNormalizer(BaseASRNormalizer):
    """
    Universal normalizer that can handle any language with basic cleaning.
    Used as fallback when language-specific normalizers are not available.
    """
    
    LANGUAGE_CODE = "universal"
    LANGUAGE_NAME = "Universal"
    
    def _language_specific_normalization(self, text: str) -> str:
        """Basic language-agnostic normalization using universal preprocessing."""
        # Use the universal preprocessor from preprocess_text.py
        return preprocess_text_asr_universal(text, self.LANGUAGE_CODE)


def safe_import_normalizer(language_code: str, config: Optional[ASRConfig] = None, 
                          enable_disfluency_removal: bool = False) -> BaseASRNormalizer:
    """
    Safe wrapper for create_normalizer that handles all edge cases.
    
    Args:
        language_code: Language code (e.g., 'zh', 'th', 'en', 'vi', 'tl')
        config: Optional configuration
        enable_disfluency_removal: Whether to enable disfluency removal
        
    Returns:
        Appropriate normalizer instance
    """
    try:
        return create_normalizer(language_code, config, enable_disfluency_removal)
    except Exception as e:
        logger.error(f"Failed to create normalizer for '{language_code}': {e}")
        logger.info("Falling back to universal normalizer")
        return UniversalASRNormalizer(config, enable_disfluency_removal)


def create_normalizer(language_code: str, config: Optional[ASRConfig] = None, 
                     enable_disfluency_removal: bool = False) -> BaseASRNormalizer:
    """
    Factory function to create normalizers.
    
    Args:
        language_code: Language code (e.g., 'zh', 'th', 'en', 'vi', 'tl')
        config: Optional configuration
        enable_disfluency_removal: Whether to enable disfluency removal
        
    Returns:
        Appropriate normalizer instance
    """
    # Handle None or empty language_code
    if not language_code or language_code is None:
        logger.warning("Language code is None or empty, using universal normalizer")
        return UniversalASRNormalizer(config, enable_disfluency_removal)
    
    language_code = language_code.lower()
    
    # Language mapping for cleaner code
    language_mappings = {
        'zh': ('chinese_normaliser', 'ChineseASRNormaliser'),
        'chinese': ('chinese_normaliser', 'ChineseASRNormaliser'),
        'th': ('thai_normaliser', 'ThaiASRNormaliser'),
        'thai': ('thai_normaliser', 'ThaiASRNormaliser'),
        'my': ('burmese_normaliser', 'BurmeseASRNormaliser'),
        'burmese': ('burmese_normaliser', 'BurmeseASRNormaliser'),
        'km': ('khmer_normaliser', 'KhmerASRNormaliser'),
        'khmer': ('khmer_normaliser', 'KhmerASRNormaliser'),
        'lo': ('lao_normaliser', 'LaoASRNormaliser'),
        'lao': ('lao_normaliser', 'LaoASRNormaliser'),
        'id': ('indonesian_normaliser', 'IndonesianASRNormaliser'),
        'indonesian': ('indonesian_normaliser', 'IndonesianASRNormaliser'),
        'ms': ('malay_normaliser', 'MalayASRNormaliser'),
        'malay': ('malay_normaliser', 'MalayASRNormaliser'),
        'ta': ('tamil_normaliser', 'TamilASRNormaliser'),
        'tamil': ('tamil_normaliser', 'TamilASRNormaliser'),
        'en': ('nemo_english', 'RefinedNeMo_English_ASR_Normaliser'),
        'english': ('nemo_english', 'RefinedNeMo_English_ASR_Normaliser'),
        'sgen': ('nemo_singlish', 'RefinedNeMo_Singlish_ASR_Normaliser'),
        'singlish': ('nemo_singlish', 'RefinedNeMo_Singlish_ASR_Normaliser'),
        'ensg': ('nemo_singlish', 'RefinedNeMo_Singlish_ASR_Normaliser'),
        'vi': ('vietnamese_normaliser', 'VietnameseASRNormalizer'),
        'vietnamese': ('vietnamese_normaliser', 'VietnameseASRNormalizer'),
        'tl': ('filipino_normaliser', 'FilipinoASRNormalizer'),
        'filipino': ('filipino_normaliser', 'FilipinoASRNormalizer'),
        'tagalog': ('filipino_normaliser', 'FilipinoASRNormalizer'),
    }
    
    # Import and create language-specific normalizers
    try:
        if language_code in language_mappings:
            module_name, class_name = language_mappings[language_code]
            # Import the module using the full package path
            full_module_name = f'audiobench.normalizer.{module_name}'
            module = __import__(full_module_name, fromlist=[class_name])
            normalizer_class = getattr(module, class_name)
            
            # Handle different constructor signatures
            try:
                # Try new BaseASRNormalizer signature first
                return normalizer_class(config, enable_disfluency_removal)
            except TypeError as e:
                logger.debug(f"Failed with new signature for {class_name}: {e}")
                try:
                    # Try NeMo normalizer signature (language, cache_dir, use_lm_context, enable_disfluency_removal, asr_config)
                    if class_name in ['RefinedNeMo_English_ASR_Normaliser', 'RefinedNeMo_Singlish_ASR_Normaliser']:
                        return normalizer_class(
                            language=language_code,
                            cache_dir=None,
                            use_lm_context=False,
                            enable_disfluency_removal=enable_disfluency_removal,
                            asr_config=config
                        )
                    else:
                        # Try old signature with only enable_disfluency_removal
                        return normalizer_class(enable_disfluency_removal)
                except TypeError as e2:
                    logger.debug(f"Failed with old signature for {class_name}: {e2}")
                    try:
                        # Try with no arguments
                        return normalizer_class()
                    except Exception as e3:
                        logger.error(f"Failed to create {class_name} with any signature: {e3}")
                        raise
        else:
            logger.warning(f"No specific normalizer for '{language_code}', using universal normalizer")
            return UniversalASRNormalizer(config, enable_disfluency_removal)
            
    except ImportError as e:
        logger.warning(f"Failed to import normalizer for '{language_code}': {e}")
        return UniversalASRNormalizer(config, enable_disfluency_removal)
    except Exception as e:
        logger.error(f"Error creating normalizer for '{language_code}': {e}")
        return UniversalASRNormalizer(config, enable_disfluency_removal)
