"""
Normalizers package for AudioBench-SEA
Provides language-aware text normalization for ASR evaluation
"""

# Core normalizer system
from .base_normalizer import (
    BaseASRNormalizer,
    ASRConfig,
    UniversalASRNormalizer,
    create_normalizer
)

# Language-specific normalizers
from .language_normalizer import (
    LanguageNormalizer,
    LanguageNormalizerFactory,
    normalize_text_for_language,
    postprocess_text_for_asr,
    get_supported_languages,
    is_language_supported,
    get_optimal_metric_for_language,
    validate_language_code,
    get_normalizer_info,
    get_language_family,
    get_language_mappings,
    get_script_type,
    batch_normalize_texts,
    validate_normalizer_availability
)

# Normalizer manager and preprocessing functions
from .normalizer_manager import (
    normalizer_manager,
    preprocess_text_asr_with_normalizer,
    get_preprocessor_function,
    preprocess_texts_batch,
    preprocess_texts_language_specific
)

# Individual normalizers - lazy imports to avoid torch import before CUDA masking
def _lazy_import_normalizers():
    """Import individual normalizers lazily to avoid torch import before CUDA masking"""
    try:
        from .nemo_english import RefinedNeMo_English_ASR_Normaliser
    except ImportError:
        pass
    try:
        from .nemo_singlish import RefinedNeMo_Singlish_ASR_Normaliser
    except ImportError:
        pass
    try:
        from .chinese_normaliser import ChineseASRNormaliser
    except ImportError:
        pass
    try:
        from .thai_normaliser import ThaiASRNormaliser
    except ImportError:
        pass
    try:
        from .malay_normaliser import MalayASRNormaliser
    except ImportError:
        pass
    try:
        from .tamil_normaliser import TamilASRNormaliser
    except ImportError:
        pass
    try:
        from .burmese_normaliser import BurmeseASRNormaliser
    except ImportError:
        pass
    try:
        from .khmer_normaliser import KhmerASRNormaliser
    except ImportError:
        pass
    try:
        from .lao_normaliser import LaoASRNormaliser
    except ImportError:
        pass
    try:
        from .indonesian_normaliser import IndonesianASRNormaliser
    except ImportError:
        pass
    try:
        from .vietnamese_normaliser import VietnameseASRNormalizer
    except ImportError:
        pass
    try:
        from .filipino_normaliser import FilipinoASRNormalizer
    except ImportError:
        pass

__all__ = [
    # Core normalizer system
    'BaseASRNormalizer',
    'ASRConfig',
    'UniversalASRNormalizer',
    'create_normalizer',
    
    # Language-specific normalizers
    'LanguageNormalizer',
    'LanguageNormalizerFactory',
    'RefinedNeMo_English_ASR_Normaliser',
    'RefinedNeMo_Singlish_ASR_Normaliser',
    'ChineseASRNormaliser',
    'ThaiASRNormaliser',
    'MalayASRNormaliser',
    'TamilASRNormaliser',
    'BurmeseASRNormaliser',
    'KhmerASRNormaliser',
    'LaoASRNormaliser',
    'IndonesianASRNormaliser',
    'VietnameseASRNormalizer',
    'FilipinoASRNormalizer',
    
    # Normalizer manager and preprocessing functions
    'normalizer_manager',
    'preprocess_text_asr_with_normalizer',
    'get_preprocessor_function',
    'preprocess_texts_batch',
    'preprocess_texts_language_specific',
    
    # Utility functions
    'normalize_text_for_language',
    'postprocess_text_for_asr',
    'get_supported_languages',
    'is_language_supported',
    'get_optimal_metric_for_language',
    'validate_language_code',
    'get_normalizer_info',
    'get_language_family',
    'get_language_mappings',
    'get_script_type',
    'batch_normalize_texts',
    'validate_normalizer_availability',
]