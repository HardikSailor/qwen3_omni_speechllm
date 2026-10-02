#!/usr/bin/env python3
"""
Language-Specific Text Normalizer for AudioBench-SEA
Provides language-aware text normalization for ASR evaluation metrics

This module provides:
- LanguageNormalizer: Base class for language-specific normalizers
- LanguageNormalizerFactory: Factory for creating language normalizers
- Individual language normalizer classes
- Utility functions for text normalization
"""

import importlib
import importlib.util
import sys
import os
import inspect
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import re

# Import preprocessing functions
from .text_normalizer.preprocess_text import (
    preprocess_text_asr_universal, clean_asr_prediction, 
    preprocess_text_generic, PreprocessConfig
)

logger = logging.getLogger(__name__)

def _instantiate_normalizer(normalizer_cls, enable_disfluency_removal: bool):
    """Instantiate with kwarg if supported; fall back to no-arg."""
    try:
        sig = inspect.signature(normalizer_cls)
        if "enable_disfluency_removal" in sig.parameters:
            return normalizer_cls(enable_disfluency_removal=enable_disfluency_removal)
        return normalizer_cls()
    except Exception:
        # Last-resort: try no-arg
        return normalizer_cls()

def safe_import_normalizer(module_name: str, class_name: str, enable_disfluency_removal: bool = False):
    """
    Robust import helper for normalizers using direct imports of specialized normalizer classes.
    
    Args:
        module_name: Name of the normalizer module
        class_name: Name of the normalizer class
        enable_disfluency_removal: Whether to enable disfluency removal
        
    Returns:
        Normalizer instance or None if import fails
    """
    try:
        # Direct import of specialized normalizer classes
        if module_name == 'chinese_normaliser':
            from .chinese_normaliser import ChineseASRNormaliser
            return _instantiate_normalizer(ChineseASRNormaliser, enable_disfluency_removal)
        elif module_name == 'thai_normaliser':
            from .thai_normaliser import ThaiASRNormaliser
            return _instantiate_normalizer(ThaiASRNormaliser, enable_disfluency_removal)
        elif module_name == 'burmese_normaliser':
            from .burmese_normaliser import BurmeseASRNormaliser
            return _instantiate_normalizer(BurmeseASRNormaliser, enable_disfluency_removal)
        elif module_name == 'khmer_normaliser':
            from .khmer_normaliser import KhmerASRNormaliser
            return _instantiate_normalizer(KhmerASRNormaliser, enable_disfluency_removal)
        elif module_name == 'lao_normaliser':
            from .lao_normaliser import LaoASRNormaliser
            return _instantiate_normalizer(LaoASRNormaliser, enable_disfluency_removal)
        elif module_name == 'indonesian_normaliser':
            from .indonesian_normaliser import IndonesianASRNormaliser
            return _instantiate_normalizer(IndonesianASRNormaliser, enable_disfluency_removal)
        elif module_name == 'malay_normaliser':
            from . import _lazy_import_normalizers
            _lazy_import_normalizers()  # Ensure normalizers are imported
            from .malay_normaliser import MalayASRNormaliser
            return _instantiate_normalizer(MalayASRNormaliser, enable_disfluency_removal)
        elif module_name == 'tamil_normaliser':
            from .tamil_normaliser import TamilASRNormaliser
            return _instantiate_normalizer(TamilASRNormaliser, enable_disfluency_removal)
        elif module_name == 'nemo_english':
            from .nemo_english import RefinedNeMo_English_ASR_Normaliser
            return _instantiate_normalizer(RefinedNeMo_English_ASR_Normaliser, enable_disfluency_removal)
        elif module_name == 'nemo_singlish':
            from .nemo_singlish import RefinedNeMo_Singlish_ASR_Normaliser
            return _instantiate_normalizer(RefinedNeMo_Singlish_ASR_Normaliser, enable_disfluency_removal)
        elif module_name == 'vietnamese_normaliser':
            from .vietnamese_normaliser import VietnameseASRNormalizer
            return _instantiate_normalizer(VietnameseASRNormalizer, enable_disfluency_removal)
        elif module_name == 'filipino_normaliser':
            from .filipino_normaliser import FilipinoASRNormalizer
            return _instantiate_normalizer(FilipinoASRNormalizer, enable_disfluency_removal)
        else:
            logger.warning(f"No direct import mapping found for module: {module_name}")
            return None
            
    except Exception as e:
        logger.debug(f"Failed to import normalizer {module_name}.{class_name}: {e}")
        return None


class LanguageNormalizer(ABC):
    """
    Base class for language-specific text normalization.
    
    This class provides a common interface for all language-specific normalizers.
    Uses preprocessing functions from preprocess_text.py first, then applies
    language-specific normalization.
    """
    
    def __init__(self, language_code: str, enable_disfluency_removal: bool = False):
        """
        Initialize the language normalizer.
        
        Args:
            language_code: ISO language code (e.g., 'zh', 'th', 'en')
            enable_disfluency_removal: Whether to enable disfluency removal
        """
        self.language_code = language_code
        self.enable_disfluency_removal = enable_disfluency_removal
    
    def normalize(self, text: str) -> str:
        """
        Normalize text for ASR evaluation.
        First applies base preprocessing, then language-specific normalization.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        try:
            # Step 1: Apply base preprocessing using preprocess_text.py
            text = preprocess_text_asr_universal(text, self.language_code)
            
            # Step 2: Apply language-specific normalization
            text = self._language_specific_normalize(text)
            
            return text
            
        except Exception as e:
            logger.error(f"Normalization failed for {self.language_code}: {e}")
            # Fallback to basic preprocessing
            try:
                return preprocess_text_generic(text, PreprocessConfig())
            except Exception:
                return text.strip()
    
    @abstractmethod
    def _language_specific_normalize(self, text: str) -> str:
        """
        Apply language-specific normalization.
        Must be implemented by subclasses.
        
        Args:
            text: Preprocessed text to normalize
            
        Returns:
            Language-specific normalized text
        """
        pass


class ChineseNormalizer(LanguageNormalizer):
    """Chinese text normalizer for ASR evaluation"""
    
    def __init__(self, enable_disfluency_removal: bool = False):
        super().__init__("zh", enable_disfluency_removal)
    
    def _language_specific_normalize(self, text: str) -> str:
        """Apply Chinese-specific normalization"""
        try:
            # Use the safe import helper function
            chinese_normalizer = safe_import_normalizer("chinese_normaliser", "ChineseASRNormaliser", self.enable_disfluency_removal)
            
            if chinese_normalizer is not None:
                return chinese_normalizer.normalize_for_asr_eval(text)
            else:
                logger.debug("Chinese normalizer not available, using basic cleaning")
                return self._basic_chinese_normalize(text)
                
        except Exception as e:
            logger.error(f"Chinese normalization failed: {e}")
            return self._basic_chinese_normalize(text)
    
    def _basic_chinese_normalize(self, text: str) -> str:
        """Basic Chinese text normalization fallback"""
        # Keep Chinese characters, alphanumeric, and spaces
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class ThaiNormalizer(LanguageNormalizer):
    """Thai text normalizer for ASR evaluation"""
    
    def __init__(self, enable_disfluency_removal: bool = False):
        super().__init__("th", enable_disfluency_removal)
    
    def _language_specific_normalize(self, text: str) -> str:
        """Apply Thai-specific normalization"""
        if not text:
            return ""
        
        try:
            # Try to use the existing Thai normalizer with multiple import strategies
            thai_normalizer = None
            
            # Strategy 1: Relative import
            try:
                from .thai_normaliser import ThaiASRNormaliser
                thai_normalizer = ThaiASRNormaliser(enable_disfluency_removal=self.enable_disfluency_removal)
                # logger.debug("Thai normalizer loaded via relative import")
            except ImportError:
                pass
            
            # Strategy 2: Absolute import
            if thai_normalizer is None:
                try:
                    from audiobench.normalizer.thai_normaliser import ThaiASRNormaliser
                    thai_normalizer = ThaiASRNormaliser(enable_disfluency_removal=self.enable_disfluency_removal)
                    # logger.debug("Thai normalizer loaded via absolute import")
                except ImportError:
                    pass
            
            if thai_normalizer is not None:
                return thai_normalizer.normalize_for_asr_eval(text)
            else:
                logger.debug("Thai normalizer not available, using basic cleaning")
                return self._basic_cleanup(text)
                
        except Exception as e:
            logger.error(f"Thai normalization failed: {e}")
            return self._basic_cleanup(text)
    
    def _basic_cleanup(self, text: str) -> str:
        """Basic text cleaning for Thai"""
        # Remove extra whitespace but preserve word boundaries
        text = re.sub(r'\s+', ' ', text.strip())
        return text


class VietnameseNormalizer(LanguageNormalizer):
    """Vietnamese text normalizer for ASR evaluation"""
    
    def __init__(self, enable_disfluency_removal: bool = False):
        super().__init__("vi", enable_disfluency_removal)
    
    def _language_specific_normalize(self, text: str) -> str:
        """Apply Vietnamese-specific normalization"""
        try:
            # Use the safe import helper function
            vietnamese_normalizer = safe_import_normalizer("vietnamese_normaliser", "VietnameseASRNormalizer", self.enable_disfluency_removal)
            
            if vietnamese_normalizer is not None:
                return vietnamese_normalizer.normalize_for_asr_eval(text)
            else:
                logger.debug("Vietnamese normalizer not available, using basic cleaning")
                return self._basic_vietnamese_normalize(text)
                
        except Exception as e:
            logger.error(f"Vietnamese normalization failed: {e}")
            return self._basic_vietnamese_normalize(text)
    
    def _basic_vietnamese_normalize(self, text: str) -> str:
        """Basic Vietnamese text normalization fallback"""
        # Keep Vietnamese characters, alphanumeric, and spaces
        text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)  # Keep Vietnamese diacritics
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class IndonesianNormalizer(LanguageNormalizer):
    """Indonesian text normalizer for ASR evaluation"""
    
    def __init__(self, enable_disfluency_removal: bool = False):
        super().__init__("id", enable_disfluency_removal)
    
    def _language_specific_normalize(self, text: str) -> str:
        """Apply Indonesian-specific normalization"""
        if not text:
            return ""
        
        # Lowercase and remove extra whitespace
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation for ASR evaluation
        text = re.sub(r'[^\w\s]', '', text)
        
        return text
    


class MalayNormalizer(LanguageNormalizer):
    """Malay text normalizer for ASR evaluation"""

    def __init__(self, enable_disfluency_removal: bool = False):
        super().__init__("ms", enable_disfluency_removal)
    
    def _language_specific_normalize(self, text: str) -> str:
        """Apply Malay-specific normalization"""
        if not text:
            return ""
        
        # Lowercase and remove extra whitespace
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation for ASR evaluation
        text = re.sub(r'[^\w\s]', '', text)
        
        return text
    


class TamilNormalizer(LanguageNormalizer):
    """Tamil text normalizer for ASR evaluation"""
    
    def __init__(self, enable_disfluency_removal: bool = False):
        super().__init__("ta", enable_disfluency_removal)
    
    def _language_specific_normalize(self, text: str) -> str:
        """Apply Tamil-specific normalization"""
        if not text:
            return ""
        
        try:
            # Try to use the existing Tamil normalizer with multiple import strategies
            tamil_normalizer = None
            
            # Strategy 1: Relative import
            try:
                from .tamil_normaliser import TamilASRNormaliser
                tamil_normalizer = TamilASRNormaliser(enable_disfluency_removal=self.enable_disfluency_removal)
                # logger.debug("Tamil normalizer loaded via relative import")
            except ImportError:
                pass
            
            # Strategy 2: Absolute import
            if tamil_normalizer is None:
                try:
                    from audiobench.normalizer.tamil_normaliser import TamilASRNormaliser
                    tamil_normalizer = TamilASRNormaliser(enable_disfluency_removal=self.enable_disfluency_removal)
                    # logger.debug("Tamil normalizer loaded via absolute import")
                except ImportError:
                    pass
            
            if tamil_normalizer is not None:
                return tamil_normalizer.normalize_for_asr_eval(text)
            else:
                logger.debug("Tamil normalizer not available, using basic cleaning")
                return self._basic_cleanup(text)
                
        except Exception as e:
            logger.error(f"Tamil normalization failed: {e}")
            return self._basic_cleanup(text)
    
    def _basic_cleanup(self, text: str) -> str:
        """Basic text cleaning for Tamil"""
        # Remove extra whitespace but preserve word boundaries
        text = re.sub(r'\s+', ' ', text.strip())
        return text


class EnglishNormalizer(LanguageNormalizer):
    """English text normalizer for ASR evaluation"""
    
    def __init__(self, enable_disfluency_removal: bool = False):
        super().__init__("en", enable_disfluency_removal)
    
    def _language_specific_normalize(self, text: str) -> str:
        """Apply English-specific normalization"""
        if not text:
            return ""
        
        try:
            # Use the safe import helper function
            english_normalizer = safe_import_normalizer("nemo_english", "RefinedNeMo_English_ASR_Normaliser", self.enable_disfluency_removal)
            
            if english_normalizer is not None:
                return english_normalizer.normalize_for_asr_eval(text)
            else:
                logger.debug("English normalizer not available, using basic cleaning")
                return self._basic_cleanup(text)
                
        except Exception as e:
            logger.error(f"English normalization failed: {e}")
            return self._basic_cleanup(text)
    
    
    def _basic_cleanup(self, text: str) -> str:
        """Basic text cleaning for English"""
        # Lowercase and remove extra whitespace
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation for ASR evaluation
        text = re.sub(r'[^\w\s]', '', text)
        
        return text


class SinglishNormalizer(LanguageNormalizer):
    """Singlish (Singaporean English) text normalizer for ASR evaluation"""
    
    def __init__(self, enable_disfluency_removal: bool = False):
        super().__init__("sgen", enable_disfluency_removal)
    
    def _language_specific_normalize(self, text: str) -> str:
        """Apply Singlish-specific normalization"""
        if not text:
            return ""
        
        try:
            # Use the safe import helper function
            singlish_normalizer = safe_import_normalizer("nemo_singlish", "RefinedNeMo_Singlish_ASR_Normaliser", self.enable_disfluency_removal)
            
            if singlish_normalizer is not None:
                return singlish_normalizer.normalize_for_asr_eval(text)
            else:
                logger.debug("Singlish normalizer not available, using basic cleaning")
                return self._basic_cleanup(text)
                
        except Exception as e:
            logger.error(f"Singlish normalization failed: {e}")
            return self._basic_cleanup(text)
    
    
    def _basic_cleanup(self, text: str) -> str:
        """Basic text cleaning for Singlish"""
        # Lowercase and remove extra whitespace
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation for ASR evaluation
        text = re.sub(r'[^\w\s]', '', text)
        
        return text


class BurmeseNormalizer(LanguageNormalizer):
    """Burmese text normalizer for ASR evaluation"""
    
    def __init__(self, enable_disfluency_removal: bool = False):
        super().__init__("my", enable_disfluency_removal)
    
    def _language_specific_normalize(self, text: str) -> str:
        """Apply Burmese-specific normalization"""
        if not text:
            return ""
        
        try:
            # Use the safe import helper function
            burmese_normalizer = safe_import_normalizer("burmese_normaliser", "BurmeseASRNormaliser", self.enable_disfluency_removal)
            
            if burmese_normalizer is not None:
                return burmese_normalizer.normalize_for_asr_eval(text)
            else:
                logger.debug("Burmese normalizer not available, using basic cleaning")
                return self._basic_cleanup(text)
                
        except Exception as e:
            logger.error(f"Burmese normalization failed: {e}")
            return self._basic_cleanup(text)
    
    
    def _basic_cleanup(self, text: str) -> str:
        """Basic text cleaning for Burmese"""
        # Remove extra whitespace but preserve character boundaries
        text = re.sub(r'\s+', ' ', text.strip())
        return text


class KhmerNormalizer(LanguageNormalizer):
    """Khmer text normalizer for ASR evaluation"""
    
    def __init__(self, enable_disfluency_removal: bool = False):
        super().__init__("km", enable_disfluency_removal)
    
    def _language_specific_normalize(self, text: str) -> str:
        """Apply Khmer-specific normalization"""
        if not text:
            return ""
        
        try:
            # Use the safe import helper function
            khmer_normalizer = safe_import_normalizer("khmer_normaliser", "KhmerASRNormaliser", self.enable_disfluency_removal)
            
            if khmer_normalizer is not None:
                return khmer_normalizer.normalize_for_asr_eval(text)
            else:
                logger.debug("Khmer normalizer not available, using basic cleaning")
                return self._basic_cleanup(text)
                
        except Exception as e:
            logger.error(f"Khmer normalization failed: {e}")
            return self._basic_cleanup(text)
    
    
    def _basic_cleanup(self, text: str) -> str:
        """Basic text cleaning for Khmer"""
        # Remove extra whitespace but preserve character boundaries
        text = re.sub(r'\s+', ' ', text.strip())
        return text


class LaoNormalizer(LanguageNormalizer):
    """Lao text normalizer for ASR evaluation"""
    
    def __init__(self, enable_disfluency_removal: bool = False):
        super().__init__("lo", enable_disfluency_removal)
    
    def _language_specific_normalize(self, text: str) -> str:
        """Apply Lao-specific normalization"""
        if not text:
            return ""
        
        try:
            # Use the safe import helper function
            lao_normalizer = safe_import_normalizer("lao_normaliser", "LaoASRNormaliser", self.enable_disfluency_removal)
            
            if lao_normalizer is not None:
                return lao_normalizer.normalize_for_asr_eval(text)
            else:
                logger.debug("Lao normalizer not available, using basic cleaning")
                return self._basic_cleanup(text)
                
        except Exception as e:
            logger.error(f"Lao normalization failed: {e}")
            return self._basic_cleanup(text)
    
    
    def _basic_cleanup(self, text: str) -> str:
        """Basic text cleaning for Lao"""
        # Remove extra whitespace but preserve character boundaries
        text = re.sub(r'\s+', ' ', text.strip())
        return text




class FilipinoNormalizer(LanguageNormalizer):
    """Filipino/Tagalog text normalizer for ASR evaluation"""
    
    def __init__(self, enable_disfluency_removal: bool = False):
        super().__init__("tl", enable_disfluency_removal)
    
    def _language_specific_normalize(self, text: str) -> str:
        """Apply Filipino-specific normalization"""
        try:
            # Use the safe import helper function
            filipino_normalizer = safe_import_normalizer("filipino_normaliser", "FilipinoASRNormalizer", self.enable_disfluency_removal)
            
            if filipino_normalizer is not None:
                return filipino_normalizer.normalize_for_asr_eval(text)
            else:
                logger.debug("Filipino normalizer not available, using basic cleaning")
                return self._basic_filipino_normalize(text)
                
        except Exception as e:
            logger.error(f"Filipino normalization failed: {e}")
            return self._basic_filipino_normalize(text)
    
    def _basic_filipino_normalize(self, text: str) -> str:
        """Basic Filipino text normalization fallback"""
        # Keep Filipino characters, alphanumeric, and spaces
        text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)  # Keep Filipino diacritics
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class LanguageNormalizerFactory:
    """Factory for creating language-specific normalizers"""
    
    _normalizers = {
        'zh': ChineseNormalizer,
        'th': ThaiNormalizer,
        'lo': LaoNormalizer,  # Lao has its own normalizer
        'km': KhmerNormalizer,  # Khmer has its own normalizer
        'my': BurmeseNormalizer,  # Burmese has its own normalizer
        'vi': VietnameseNormalizer,
        'id': IndonesianNormalizer,
        'ms': MalayNormalizer,
        'ta': TamilNormalizer,
        'tl': FilipinoNormalizer,  # Filipino/Tagalog has its own normalizer
        'en': EnglishNormalizer,
        'sgen': SinglishNormalizer,  # Singaporean English
    }
    
    @classmethod
    def get_normalizer(cls, language_code: str, enable_disfluency_removal: bool = False) -> LanguageNormalizer:
        """Get a normalizer for the specified language"""
        # Handle None or empty language_code
        if not language_code or language_code is None:
            logger.debug("Language code is None or empty, using English normalizer")
            return EnglishNormalizer(enable_disfluency_removal=enable_disfluency_removal)
        
        language_code = language_code.lower()
        
        if language_code in cls._normalizers:
            return cls._normalizers[language_code](enable_disfluency_removal=enable_disfluency_removal)
        else:
            logger.debug(f"No normalizer found for language '{language_code}', using English")
            return EnglishNormalizer(enable_disfluency_removal=enable_disfluency_removal)
    
    @classmethod
    def list_supported_languages(cls) -> List[str]:
        """List all supported language codes"""
        return list(cls._normalizers.keys())


def normalize_text_for_language(texts: List[str], language: str, enable_disfluency_removal: bool = False) -> List[str]:
    """
    Normalize a list of texts for a specific language.
    
    Args:
        texts: List of texts to normalize
        language: Language code (e.g., 'zh', 'th', 'vi', 'en')
        enable_disfluency_removal: Whether to enable disfluency removal
        
    Returns:
        List of normalized texts
    """
    normalizer = LanguageNormalizerFactory.get_normalizer(language, enable_disfluency_removal)
    return [normalizer.normalize(text) for text in texts]


def postprocess_text_for_asr(dataset_instance, texts: List[str], language: str = None, enable_disfluency_removal: bool = False) -> List[str]:
    """
    Post-process texts for ASR evaluation using language-specific normalization.
    
    This function integrates with the existing normalizer system and provides
    language-aware text normalization for ASR evaluation metrics.
    
    Args:
        dataset_instance: Dataset instance with language attribute
        texts: List of texts to normalize
        language: Language code (overrides dataset_instance.language if provided)
        enable_disfluency_removal: Whether to enable disfluency removal
        
    Returns:
        List of normalized texts
    """
    # Input validation
    if not texts:
        logger.warning("Empty text list provided")
        return []
    
    if not isinstance(texts, list):
        logger.error(f"Texts must be a list, got {type(texts)}")
        return []
    
    # Get language from parameters or dataset instance
    if language is None:
        language = getattr(dataset_instance, 'language', 'en')
    
    # Validate language code
    if not language or not isinstance(language, str):
        logger.warning(f"Invalid language code '{language}', defaulting to 'en'")
        language = 'en'
    
    language = language.lower().strip()
    
    # Normalize texts using language-specific normalizer
    try:
        normalized_texts = normalize_text_for_language(texts, language, enable_disfluency_removal)
        logger.debug(f"Normalized {len(texts)} texts for language '{language}' with disfluency_removal={enable_disfluency_removal}")
        return normalized_texts
    except Exception as e:
        logger.error(f"Language normalization failed for '{language}': {e}")
        
        # Fallback: basic text cleaning
        logger.info("Using basic text cleaning as fallback")
        cleaned_texts = []
        for text in texts:
            if text:
                # Basic cleaning: strip whitespace, normalize spaces
                cleaned = re.sub(r'\s+', ' ', str(text).strip())
                cleaned_texts.append(cleaned)
            else:
                cleaned_texts.append("")
        
        return cleaned_texts


def get_supported_languages() -> List[str]:
    """Get list of all supported language codes."""
    return LanguageNormalizerFactory.list_supported_languages()


def is_language_supported(language: str) -> bool:
    """Check if a language is supported."""
    return language.lower() in LanguageNormalizerFactory.list_supported_languages()


def get_language_family(language: str) -> str:
    """
    Get the language family for a given language code.
    
    Args:
        language: Language code
        
    Returns:
        Language family name
    """
    language = language.lower()
    
    # Language family mappings
    language_families = {
        # Sino-Tibetan
        'zh': 'Sino-Tibetan',
        'my': 'Sino-Tibetan',
        
        # Tai-Kadai
        'th': 'Tai-Kadai',
        'lo': 'Tai-Kadai',
        
        # Austroasiatic
        'km': 'Austroasiatic',
        'vi': 'Austroasiatic',
        
        # Austronesian
        'id': 'Austronesian',
        'ms': 'Austronesian',
        'tl': 'Austronesian',
        
        # Dravidian
        'ta': 'Dravidian',
        'ml': 'Dravidian',
        
        # Indo-European
        'en': 'Indo-European',
        'sgen': 'Indo-European',
    }
    
    return language_families.get(language, 'Unknown')


def get_optimal_metric_for_language(language: str) -> str:
    """
    Get the optimal primary metric for a given language.
    
    Args:
        language: Language code
        
    Returns:
        'CER' for CJK and character-based languages, 'WER' for others
    """
    language = language.lower()
    
    # Character-based languages typically use CER as primary metric
    character_based_languages = {
        'zh', 'ja', 'ko',  # CJK languages
        'th', 'lo', 'km', 'my',  # Southeast Asian scripts
        'ml', 'ta'  # Dravidian scripts
    }
    
    if language in character_based_languages:
        return 'CER'
    else:
        return 'WER'


def validate_language_code(language: str) -> str:
    """
    Validate and normalize language code.
    
    Args:
        language: Language code to validate
        
    Returns:
        Normalized language code or 'en' as fallback
    """
    if not language or not isinstance(language, str):
        return 'en'
    
    language = language.lower().strip()
    
    if is_language_supported(language):
        return language
    else:
        logger.warning(f"Unsupported language '{language}', using 'en'")
        return 'en'


# Removed duplicate function - use postprocess_text_for_asr instead


def get_normalizer_info(language: str) -> Dict[str, Any]:
    """
    Get information about available normalizers for a language.
    
    Args:
        language: Language code
        
    Returns:
        Dictionary with normalizer information
    """
    language = language.lower() if language else "en"
    
    normalizer_info = {
        'language': language,
        'language_family': get_language_family(language),
        'supported': is_language_supported(language),
        'optimal_metric': get_optimal_metric_for_language(language),
        'available_normalizers': []
    }
    
    # Check which specialized normalizers are available
    try:
        if language == 'zh':
            from .chinese_normaliser import ChineseASRNormaliser
            normalizer_info['available_normalizers'].append('ChineseASRNormaliser')
    except ImportError:
        pass
    
    try:
        if language == 'th':
            from .thai_normaliser import ThaiASRNormaliser
            normalizer_info['available_normalizers'].append('ThaiASRNormaliser')
    except ImportError:
        pass
    
    try:
        if language == 'lo':
            from .lao_normaliser import LaoASRNormaliser
            normalizer_info['available_normalizers'].append('LaoASRNormaliser')
    except ImportError:
        pass
    
    try:
        if language == 'km':
            from .khmer_normaliser import KhmerASRNormaliser
            normalizer_info['available_normalizers'].append('KhmerASRNormaliser')
    except ImportError:
        pass
    
    try:
        if language == 'my':
            from .burmese_normaliser import BurmeseASRNormaliser
            normalizer_info['available_normalizers'].append('BurmeseASRNormaliser')
    except ImportError:
        pass
    
    try:
        if language == 'ms':
            from .malay_normaliser import MalayASRNormaliser
            normalizer_info['available_normalizers'].append('MalayASRNormaliser')
    except ImportError:
        pass
    
    try:
        if language == 'id':
            from .indonesian_normaliser import IndonesianASRNormaliser
            normalizer_info['available_normalizers'].append('IndonesianASRNormaliser')
    except ImportError:
        pass
    
    try:
        if language == 'ta':
            from .tamil_normaliser import TamilASRNormaliser
            normalizer_info['available_normalizers'].append('TamilASRNormaliser')
    except ImportError:
        pass
    
    try:
        if language == 'vi':
            from .vietnamese_normaliser import VietnameseASRNormalizer
            normalizer_info['available_normalizers'].append('VietnameseASRNormalizer')
    except ImportError:
        pass
    
    try:
        if language == 'tl':
            from .filipino_normaliser import FilipinoASRNormalizer
            normalizer_info['available_normalizers'].append('FilipinoASRNormalizer')
    except ImportError:
        pass
    
    
    try:
        if language == 'en':
            from .nemo_english import RefinedNeMo_English_ASR_Normaliser
            normalizer_info['available_normalizers'].append('RefinedNeMo_English_ASR_Normaliser')
    except ImportError:
        pass
    
    try:
        if language == 'sgen':
            from .nemo_singlish import RefinedNeMo_Singlish_ASR_Normaliser
            normalizer_info['available_normalizers'].append('RefinedNeMo_Singlish_ASR_Normaliser')
    except ImportError:
        pass
    
    # Always available: custom language normalizer
    normalizer_info['available_normalizers'].append('CustomLanguageNormalizer')
    
    return normalizer_info


def get_disfluency_setting_from_config() -> bool:
    """
    Get the normalize_disfluency setting from the main configuration.
    
    Returns:
        True if disfluency normalization is enabled, False otherwise
    """
    try:
        import yaml
        import os
        
        # Try to find and load the main config
        config_paths = [
            "configs/main.yaml",
            "../configs/main.yaml",
            "../../configs/main.yaml"
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if config and 'evaluation' in config:
                        return config['evaluation'].get('normalize_disfluency', False)
        
        # Fallback: check environment variable
        import os
        env_setting = os.environ.get('NORMALIZE_DISFLUENCY', 'false').lower()
        return env_setting in ('true', '1', 'yes')
        
    except Exception as e:
        logger.warning(f"Failed to load disfluency setting from config: {e}, defaulting to False")
        return False


# Removed redundant functions - use LanguageNormalizerFactory.get_normalizer() and postprocess_text_for_asr() instead


def get_language_mappings() -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive language mappings and information.
    
    Returns:
        Dictionary with language code as key and language info as value
    """
    return {
        'zh': {
            'name': 'Chinese',
            'family': 'Sino-Tibetan',
            'script': 'Han',
            'normalizer': 'ChineseNormalizer',
            'optimal_metric': 'CER'
        },
        'th': {
            'name': 'Thai',
            'family': 'Tai-Kadai',
            'script': 'Thai',
            'normalizer': 'ThaiNormalizer',
            'optimal_metric': 'CER'
        },
        'lo': {
            'name': 'Lao',
            'family': 'Tai-Kadai',
            'script': 'Lao',
            'normalizer': 'LaoNormalizer',
            'optimal_metric': 'CER'
        },
        'km': {
            'name': 'Khmer',
            'family': 'Austroasiatic',
            'script': 'Khmer',
            'normalizer': 'KhmerNormalizer',
            'optimal_metric': 'CER'
        },
        'my': {
            'name': 'Burmese',
            'family': 'Sino-Tibetan',
            'script': 'Myanmar',
            'normalizer': 'BurmeseNormalizer',
            'optimal_metric': 'CER'
        },
        'vi': {
            'name': 'Vietnamese',
            'family': 'Austroasiatic',
            'script': 'Latin',
            'normalizer': 'VietnameseNormalizer',
            'optimal_metric': 'WER'
        },
        'id': {
            'name': 'Indonesian',
            'family': 'Austronesian',
            'script': 'Latin',
            'normalizer': 'IndonesianNormalizer',
            'optimal_metric': 'WER'
        },
        'ms': {
            'name': 'Malay',
            'family': 'Austronesian',
            'script': 'Latin',
            'normalizer': 'MalayNormalizer',
            'optimal_metric': 'WER'
        },
        'ta': {
            'name': 'Tamil',
            'family': 'Dravidian',
            'script': 'Tamil',
            'normalizer': 'TamilNormalizer',
            'optimal_metric': 'CER'
        },
        'tl': {
            'name': 'Filipino/Tagalog',
            'family': 'Austronesian',
            'script': 'Latin',
            'normalizer': 'FilipinoNormalizer',
            'optimal_metric': 'WER'
        },
        'en': {
            'name': 'English',
            'family': 'Indo-European',
            'script': 'Latin',
            'normalizer': 'EnglishNormalizer',
            'optimal_metric': 'WER'
        },
        'sgen': {
            'name': 'Singaporean English',
            'family': 'Indo-European',
            'script': 'Latin',
            'normalizer': 'SinglishNormalizer',
            'optimal_metric': 'WER'
        }
    }


def get_script_type(language: str) -> str:
    """
    Get the script type for a given language.
    
    Args:
        language: Language code
        
    Returns:
        Script type ('character-based', 'word-based', 'mixed')
    """
    language = language.lower()
    
    # Character-based scripts
    character_scripts = {'zh', 'ja', 'ko', 'th', 'lo', 'km', 'my', 'ta', 'ml'}
    
    if language in character_scripts:
        return 'character-based'
    else:
        return 'word-based'


def batch_normalize_texts(texts: List[str], language: str, enable_disfluency_removal: bool = False, 
                         batch_size: int = 1000) -> List[str]:
    """
    Normalize a large batch of texts with memory-efficient processing.
    
    Args:
        texts: List of texts to normalize
        language: Language code
        enable_disfluency_removal: Whether to enable disfluency removal
        batch_size: Number of texts to process at once
        
    Returns:
        List of normalized texts
    """
    if not texts:
        return []
    
    normalizer = LanguageNormalizerFactory.get_normalizer(language, enable_disfluency_removal)
    normalized_texts = []
    
    # Process in batches to avoid memory issues
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_normalized = [normalizer.normalize(text) for text in batch]
        normalized_texts.extend(batch_normalized)
        
        # Log progress for large batches
        if len(texts) > batch_size:
            logger.debug(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
    
    return normalized_texts


def validate_normalizer_availability() -> Dict[str, bool]:
    """
    Check which normalizers are available in the system.
    
    Returns:
        Dictionary with language code as key and availability as value
    """
    availability = {}
    
    for language in LanguageNormalizerFactory.list_supported_languages():
        try:
            normalizer = LanguageNormalizerFactory.get_normalizer(language)
            # Try to normalize a simple test string
            test_result = normalizer.normalize("test")
            availability[language] = True
        except Exception as e:
            # logger.debug(f"Normalizer for {language} not available: {e}")
            availability[language] = False
    
    return availability
