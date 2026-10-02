#!/usr/bin/env python3
"""
Enhanced Normalizer Manager for AudioBench-SEA
Provides a centralized interface for text normalization during ASR evaluation.
Combines normalizer management and preprocessing functionality.
"""

import logging
from typing import Optional, Dict, Any, List, Callable
import importlib
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

class NormalizerManager:
    """
    Central manager for handling different text normalizers for ASR evaluation.
    Supports dynamic loading of normalizers and easy extension for new languages/types.
    """
    
    def __init__(self):
        self.normalizers = {}
        self.available_normalizers = {
            'nemo_english': 'audiobench.normalizer.nemo_english',
            'nemo_singlish': 'audiobench.normalizer.nemo_singlish',
            'chinese': 'audiobench.normalizer.chinese_normaliser',
            'malay': 'audiobench.normalizer.malay_normaliser',
            'tamil': 'audiobench.normalizer.tamil_normaliser',
            'filipino': 'audiobench.normalizer.filipino_normaliser',
            'burmese': 'audiobench.normalizer.burmese_normaliser',
            'indonesian': 'audiobench.normalizer.indonesian_normaliser',
            'khmer': 'audiobench.normalizer.khmer_normaliser',
            'lao': 'audiobench.normalizer.lao_normaliser',
        }
    
    def get_normalizer(self, normalizer_name: str, enable_disfluency_removal: bool = True, **kwargs):
        """
        Get a normalizer instance by name.
        
        Args:
            normalizer_name: Name of the normalizer
            enable_disfluency_removal: Whether to enable disfluency removal
            **kwargs: Additional arguments for the normalizer
            
        Returns:
            Normalizer instance or None if not available
        """
        if normalizer_name == 'none' or normalizer_name is None:
            return None
            
        cache_key = f"{normalizer_name}_{enable_disfluency_removal}_{hash(frozenset(kwargs.items()))}"
        
        if cache_key in self.normalizers:
            return self.normalizers[cache_key]
        
        if normalizer_name not in self.available_normalizers:
            logger.warning(f"Normalizer '{normalizer_name}' not found. Available normalizers: {list(self.available_normalizers.keys())}")
            return None
        
        try:
            module_path = self.available_normalizers[normalizer_name]
            module = importlib.import_module(module_path)
            
            # Map normalizer names to their class names
            normalizer_classes = {
                'nemo_english': 'RefinedNeMo_English_ASR_Normaliser',
                'nemo_singlish': 'RefinedNeMo_Singlish_ASR_Normaliser',
                'chinese': 'ChineseASRNormaliser',
                'malay': 'MalayASRNormaliser',
                'thai': 'ThaiASRNormaliser',
                'tamil': 'TamilASRNormaliser',
                'filipino': 'FilipinoASRNormalizer',
                'burmese': 'BurmeseASRNormaliser',
                'vietnamese': 'VietnameseASRNormalizer',
                'indonesian': 'IndonesianASRNormaliser',
                'khmer': 'KhmerASRNormaliser',
                'lao': 'LaoASRNormaliser',
            }
            
            class_name = normalizer_classes.get(normalizer_name)
            if class_name and hasattr(module, class_name):
                normalizer_class = getattr(module, class_name)
            else:
                # Fallback: assume class name is TitleCase of normalizer_name
                class_name = ''.join(word.capitalize() for word in normalizer_name.split('_')) + '_Normaliser'
                normalizer_class = getattr(module, class_name)
            
            # Handle different normalizer constructor signatures
            if class_name in ['RefinedNeMo_English_ASR_Normaliser', 'RefinedNeMo_Singlish_ASR_Normaliser']:
                # NeMo normalizers expect different parameters
                normalizer = normalizer_class(
                    language=kwargs.get('language', 'en'),
                    cache_dir=kwargs.get('cache_dir', None),
                    use_lm_context=kwargs.get('use_lm_context', False),
                    enable_disfluency_removal=enable_disfluency_removal,
                    asr_config=kwargs.get('asr_config', None)
                )
            else:
                # Other normalizers use the standard signature
                normalizer = normalizer_class(
                    enable_disfluency_removal=enable_disfluency_removal,
                    **kwargs
                )
            
            self.normalizers[cache_key] = normalizer
            logger.info(f"Loaded normalizer: {normalizer_name}")
            
            return normalizer
            
        except Exception as e:
            logger.error(f"Failed to load normalizer '{normalizer_name}': {e}")
            return None
    
    def normalize_text(self, text: str, normalizer_name: str, enable_disfluency_removal: bool = True, **kwargs) -> str:
        """
        Normalize text using a specific normalizer.
        
        Args:
            text: Text to normalize
            normalizer_name: Name of the normalizer to use
            enable_disfluency_removal: Whether to enable disfluency removal
            **kwargs: Additional arguments for the normalizer
            
        Returns:
            Normalized text
        """
        if normalizer_name == 'none' or normalizer_name is None:
            # Fall back to existing preprocess_text_asr function
            try:
                from .text_normalizer.preprocess_text import preprocess_text_asr
                return preprocess_text_asr(text)
            except ImportError:
                logger.warning("Could not import preprocess_text_asr, returning original text")
                return text
        
        normalizer = self.get_normalizer(normalizer_name, enable_disfluency_removal, **kwargs)
        
        if normalizer is None:
            # Fall back to existing preprocess_text_asr function
            logger.warning(f"Falling back to default preprocessing for text: {text[:50]}...")
            try:
                from .text_normalizer.preprocess_text import preprocess_text_asr
                return preprocess_text_asr(text)
            except ImportError:
                logger.warning("Could not import preprocess_text_asr, returning original text")
                return text
        
        try:
            return normalizer.normalize_for_asr_eval(text)
        except Exception as e:
            logger.error(f"Normalization failed for '{text[:50]}...' with {normalizer_name}: {e}")
            # Fall back to existing preprocess_text_asr function
            try:
                from .text_normalizer.preprocess_text import preprocess_text_asr
                return preprocess_text_asr(text)
            except ImportError:
                logger.warning("Could not import preprocess_text_asr, returning original text")
                return text
    
    def list_available_normalizers(self) -> list:
        """Return list of available normalizer names."""
        return list(self.available_normalizers.keys())
    
    def get_language_normalizer(self, language: str, enable_disfluency_removal: bool = False):
        """
        Get a language-specific normalizer for ASR evaluation.
        
        Args:
            language: Language code (e.g., 'zh', 'th', 'vi', 'en')
            enable_disfluency_removal: Whether to enable disfluency removal
            
        Returns:
            LanguageNormalizer instance or None if not available
        """
        try:
            from .base_normalizer import safe_import_normalizer
            return safe_import_normalizer(language, enable_disfluency_removal=enable_disfluency_removal)
        except ImportError:
            logger.warning("Language-specific normalizer not available")
            return None
    
    def normalize_text_language_specific(self, text: str, language: str, enable_disfluency_removal: bool = False) -> str:
        """
        Normalize text using language-specific normalizer.
        
        Args:
            text: Text to normalize
            language: Language code
            enable_disfluency_removal: Whether to enable disfluency removal
            
        Returns:
            Normalized text
        """
        try:
            from .language_normalizer import normalize_text_for_language
            normalized = normalize_text_for_language([text], language, enable_disfluency_removal)
            return normalized[0] if normalized else text
        except Exception as e:
            logger.error(f"Language-specific normalization failed: {e}")
            return text
    
    def postprocess_texts_for_asr(self, dataset_instance, texts: List[str], language: str = None) -> List[str]:
        """
        Post-process texts for ASR evaluation using the integrated normalizer system.
        
        Args:
            dataset_instance: Dataset instance with language attribute
            texts: List of texts to normalize
            language: Language code (overrides dataset_instance.language if provided)
            
        Returns:
            List of normalized texts
        """
        try:
            from .language_normalizer import postprocess_text_for_asr
            # Get disfluency setting from dataset instance if available
            enable_disfluency = getattr(dataset_instance, 'normalize_disfluency', False)
            return postprocess_text_for_asr(dataset_instance, texts, language, enable_disfluency)
        except Exception as e:
            logger.error(f"ASR text postprocessing failed: {e}")
            # Fallback to basic cleaning
            return [str(text).strip() if text else "" for text in texts]
    
    def get_optimal_normalizer_for_language(self, language: str, enable_disfluency_removal: bool = False):
        """
        Get the optimal normalizer for a given language.
        
        This method delegates to the LanguageNormalizerFactory to ensure consistency.
        
        Args:
            language: Language code
            enable_disfluency_removal: Whether to enable disfluency removal
            
        Returns:
            Normalizer instance or None
        """
        try:
            from .language_normalizer import LanguageNormalizerFactory
            return LanguageNormalizerFactory.get_normalizer(language, enable_disfluency_removal)
        except Exception as e:
            logger.error(f"Failed to get normalizer for language '{language}': {e}")
            return None


# Global instance for easy access
normalizer_manager = NormalizerManager()


# =============================================================================
# Enhanced preprocessing functions
# =============================================================================

def preprocess_text_asr_with_normalizer(
    text: str, 
    normalizer_name: str = None, 
    enable_disfluency_removal: bool = True,
    **normalizer_kwargs
) -> str:
    """
    Preprocess text for ASR evaluation using a configurable normalizer.
    
    Args:
        text: Text to preprocess
        normalizer_name: Name of the normalizer to use
        enable_disfluency_removal: Whether to enable disfluency removal
        **normalizer_kwargs: Additional arguments for the normalizer
        
    Returns:
        Preprocessed text
    """
    if text is None:
        return ""
    
    if not isinstance(text, str):
        text = str(text)
    
    return normalizer_manager.normalize_text(
        text=text,
        normalizer_name=normalizer_name,
        enable_disfluency_removal=enable_disfluency_removal,
        **normalizer_kwargs
    )


def get_preprocessor_function(
    normalizer_name: str = None, 
    enable_disfluency_removal: bool = True, 
    **kwargs
) -> Callable[[str], str]:
    """
    Get a configured preprocessor function.
    
    Args:
        normalizer_name: Name of the normalizer to use
        enable_disfluency_removal: Whether to enable disfluency removal
        **kwargs: Additional arguments for the normalizer
        
    Returns:
        Configured preprocessor function
    """
    def configured_preprocessor(text: str) -> str:
        return preprocess_text_asr_with_normalizer(
            text=text,
            normalizer_name=normalizer_name,
            enable_disfluency_removal=enable_disfluency_removal,
            **kwargs
        )
    
    return configured_preprocessor


def preprocess_texts_batch(
    texts: List[str], 
    normalizer_name: str = None, 
    enable_disfluency_removal: bool = True,
    **normalizer_kwargs
) -> List[str]:
    """
    Preprocess a batch of texts for ASR evaluation.
    
    Args:
        texts: List of texts to preprocess
        normalizer_name: Name of the normalizer to use
        enable_disfluency_removal: Whether to enable disfluency removal
        **normalizer_kwargs: Additional arguments for the normalizer
        
    Returns:
        List of preprocessed texts
    """
    if not texts:
        return []
    
    if not isinstance(texts, list):
        logger.error(f"Texts must be a list, got {type(texts)}")
        return []
    
    try:
        preprocessor = get_preprocessor_function(
            normalizer_name=normalizer_name,
            enable_disfluency_removal=enable_disfluency_removal,
            **normalizer_kwargs
        )
        
        return [preprocessor(text) for text in texts]
        
    except Exception as e:
        logger.error(f"Batch preprocessing failed: {e}")
        # Fallback to basic cleaning
        return [str(text).strip() if text else "" for text in texts]


def preprocess_texts_language_specific(
    texts: List[str], 
    language: str,
    enable_disfluency_removal: bool = True
) -> List[str]:
    """
    Preprocess texts using language-specific normalization.
    
    Args:
        texts: List of texts to preprocess
        language: Language code
        enable_disfluency_removal: Whether to enable disfluency removal
        
    Returns:
        List of preprocessed texts
    """
    if not texts:
        return []
    
    if not isinstance(texts, list):
        logger.error(f"Texts must be a list, got {type(texts)}")
        return []
    
    try:
        # Create a mock dataset instance with the language and disfluency setting
        class MockDataset:
            def __init__(self, language, normalize_disfluency):
                self.language = language
                self.normalize_disfluency = normalize_disfluency
        
        mock_dataset = MockDataset(language, enable_disfluency_removal)
        
        return normalizer_manager.postprocess_texts_for_asr(
            dataset_instance=mock_dataset, 
            texts=texts, 
            language=language
        )
        
    except Exception as e:
        logger.error(f"Language-specific preprocessing failed: {e}")
        # Fallback to basic cleaning
        return [str(text).strip() if text else "" for text in texts]