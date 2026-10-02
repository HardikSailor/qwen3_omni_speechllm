#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
ASR Text Preprocessing Module for AudioBench-SEA

This module provides comprehensive text preprocessing functions for ASR evaluation,
supporting multiple languages and various preprocessing strategies.

Features:
- Unicode normalization
- Speaker tag removal
- Non-speech element removal
- Punctuation handling
- Language-specific preprocessing
- Generic preprocessing with configurable options
- Batch processing capabilities

Author: Bin Wang
Created: April 30, 2024
"""

import re
import unicodedata
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Optional dependencies
try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

try:
    from .whisper_english import EnglishTextNormalizer as _EnglishTextNormalizer
    from .whisper_malay import MalayCodeSwitchEnglishSpellingNormalizer
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class PreprocessConfig:
    """
    Configuration for generic ASR text preprocessing.
    
    Attributes:
        to_lower: Convert text to lowercase
        unicode_form: Unicode normalization form (NFC, NFKC, NFD, NFKD)
        remove_bracketed: Remove content in parentheses, brackets, braces, angle brackets
        remove_speaker_tags: Remove speaker tags like <speaker1>, <spk2>
        remove_nonspeech_meta: Remove non-speech tokens like [laughter], <noise>
        remove_punctuation: Remove punctuation marks
        keep_inword_hyphen_apostrophe: Preserve hyphens and apostrophes within words
        normalize_phones: Normalize phone numbers to digits only
        remove_urls_emails: Remove URLs and email addresses
        collapse_whitespace: Collapse multiple spaces into single spaces
        strip_edges: Remove leading and trailing whitespace
        use_jiwer: Use jiwer transforms if available
        debug: Enable debug logging
    """
    # Casing & unicode
    to_lower: bool = True
    unicode_form: str = "NFC"  # {"NFC","NFKC","NFD","NFKD"}

    # Structural removals
    remove_bracketed: bool = True  # remove (...) [...] {...} <...> and their contents
    remove_speaker_tags: bool = True  # e.g., <speaker1>:, <spk2>:
    remove_nonspeech_meta: bool = True  # e.g., [laughter], <noise>, (crosstalk)

    # Punctuation & tokens
    remove_punctuation: bool = True
    keep_inword_hyphen_apostrophe: bool = True

    # URLs, emails, numbers, phones
    normalize_phones: bool = True  # collapse  +xx xxx-xxx → digits
    remove_urls_emails: bool = True

    # Whitespace
    collapse_whitespace: bool = True
    strip_edges: bool = True

    # jiwer transforms (if installed)
    use_jiwer: bool = True  # compose a conservative chain if jiwer is available

    # Debug
    debug: bool = False

# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Speaker tags: <speaker1>, <SPEAKER2>, <spk3> (with optional trailing colon)
SPEAKER_TAG_RE = re.compile(r"<\s*(?:speaker|spk)\s*\d+\s*>:?", re.IGNORECASE)

# Bracketed content: (noise), [laughter], {cough}, <breath>, <music starts>
BRACKETED_ANY_RE = re.compile(r"[\(\[\{<]\s*[^>\}\]\)]*?\s*[\)\]\}>]")

# Punctuation patterns
PUNCT_STRICT_RE = re.compile(r"[^\w\s\u00C0-\u1FFF'-]+", re.UNICODE)
BARE_HYP_APO_RE = re.compile(r"(?<!\w)['-]|['-](?!\w)")

# URLs & emails
URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)

# Phone numbers: allow +country, spaces, dashes; collapse to digits
PHONE_RE = re.compile(r"\b(?:\+\d{1,3}[\s-]?)?(?:\d[\s-]?){8,15}\b")

# Whitespace
MULTISPACE_RE = re.compile(r"\s+")

# Non-speech tokens
NON_SPEECH_TOKENS = [
    "laughter", "laughs", "cough", "crosstalk", "noise", "music",
    "applause", "breath", "sigh", "silence", "inaudible", "unk",
    "unintelligible", "unclear", "mumbling", "whispering", "shouting"
]

# =============================================================================
# CORE PREPROCESSING FUNCTIONS
# =============================================================================

def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Normalize Unicode text to specified form with fallback to NFC."""
    try:
        return unicodedata.normalize(form, text)
    except (ValueError, TypeError):
        return unicodedata.normalize("NFC", text)

def remove_speaker_tags(text: str) -> str:
    """Remove speaker tags like <speaker1>, <spk2>."""
    return SPEAKER_TAG_RE.sub(" ", text)

def remove_bracketed_content(text: str) -> str:
    """Remove bracketed content iteratively to handle nested cases."""
    prev = None
    while prev != text:
        prev = text
        text = BRACKETED_ANY_RE.sub(" ", text)
    return text

def remove_nonspeech_words(text: str) -> str:
    """Remove non-speech tokens, optionally wrapped in brackets."""
    for tok in NON_SPEECH_TOKENS:
        text = re.sub(rf"[\[\(<{{]*\b{re.escape(tok)}\b[\]\)>}}]*", " ", text, flags=re.IGNORECASE)
    return text

def remove_urls_emails(text: str, remove_urls: bool = True, remove_emails: bool = True) -> str:
    """Remove URLs and/or email addresses from text."""
    if remove_urls:
        text = URL_RE.sub(" ", text)
    if remove_emails:
        text = EMAIL_RE.sub(" ", text)
    return text

def normalize_phone_numbers(text: str) -> str:
    """Normalize phone numbers to digits only."""
    def repl(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return digits if 8 <= len(digits) <= 15 else m.group(0)
    return PHONE_RE.sub(repl, text)

def remove_punctuation(text: str, keep_inword: bool = True) -> str:
    """Remove punctuation while optionally preserving in-word hyphens/apostrophes."""
    text = PUNCT_STRICT_RE.sub(" ", text)
    if keep_inword:
        text = BARE_HYP_APO_RE.sub(" ", text)
    else:
        text = text.replace("'", " ").replace("-", " ")
    return text

def collapse_whitespace(text: str, strip_edges: bool = True) -> str:
    """Collapse multiple spaces and optionally strip edges."""
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip() if strip_edges else text

def remove_non_speech_elements(text: str) -> str:
    """Remove common non-speech elements like 'uh', 'um', etc."""
    non_speech_patterns = r'\b(uh|umm|um|er|ah|eh|oh|hmm|hm|mm|mhm|uhh|uhm|erm|argh|ugh|tsk|tut|psst|shh|hush)\b'
    return re.sub(non_speech_patterns, '', text, flags=re.IGNORECASE)

def normalize_text_numbers(text: str) -> str:
    """Normalize text by converting digits to words and expanding contractions."""
    # Digit to word conversions
    digits_to_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
        '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
        '14': 'fourteen', '15': 'fifteen', '16': 'sixteen',
        '17': 'seventeen', '18': 'eighteen', '19': 'nineteen',
        '20': 'twenty', '30': 'thirty', '40': 'forty', '50': 'fifty',
        '60': 'sixty', '70': 'seventy', '80': 'eighty', '90': 'ninety',
    }
    for digit, word in digits_to_words.items():
        text = re.sub(r'\b' + digit + r'\b', word, text)

    # Expand common contractions
    contractions = {
        "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
        "it's": "it is", "we're": "we are", "they're": "they are",
        "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
        "doesn't": "does not", "don't": "do not", "didn't": "did not", "that's": "that is",
    }
    for contraction, expanded in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', expanded, text)

    return text

# =============================================================================
# LANGUAGE-SPECIFIC PROCESSING
# =============================================================================

def separate_and_space_chinese(text: str) -> str:
    """Separate Chinese characters and add spaces between them."""
    parts = re.split(r'([\u4e00-\u9fff]+)', text)
    processed_parts = []
    for part in parts:
        if re.match(r'[\u4e00-\u9fff]+', part):
            spaced = ' '.join(char for char in part)
            processed_parts.append(spaced)
        else:
            processed_parts.append(part)
    return ''.join(processed_parts)

def separate_and_space_thai(text: str) -> str:
    """Separate Thai characters and add spaces between them."""
    parts = re.split(r'([\u0E00-\u0E7F]+)', text)
    processed_parts = []
    for part in parts:
        if re.match(r'[\u0E00-\u0E7F]+', part):
            spaced = ' '.join(char for char in part)
            processed_parts.append(spaced)
        else:
            processed_parts.append(part)
    return ''.join(processed_parts)

# =============================================================================
# JIWER INTEGRATION
# =============================================================================

def apply_jiwer_transforms(text: str) -> str:
    """Apply conservative jiwer transforms if available."""
    if not JIWER_AVAILABLE:
        return text
    
    try:
        chain = jiwer.Compose([
            jiwer.Strip(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
        ])
        return chain(text)
    except Exception:
        return text

def apply_jiwer_standard(text: str) -> str:
    """Apply standard jiwer processing for English."""
    if not JIWER_AVAILABLE:
        return text
    
    try:
        chain = jiwer.Compose([
            jiwer.RemoveMultipleSpaces(),
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveKaldiNonWords(),
            jiwer.RemovePunctuation()
        ])
        return chain(text)
    except Exception:
        return text

# =============================================================================
# MAIN PREPROCESSING FUNCTIONS
# =============================================================================

def preprocess_text_generic(text: Optional[str], cfg: Optional[PreprocessConfig] = None) -> str:
    """
    Clean an ASR prediction robustly across languages using configurable options.

    Args:
        text: Input text to preprocess
        cfg: Configuration object (uses defaults if None)
        
    Returns:
        Cleaned text string
    """
    if text is None or not isinstance(text, str):
        return ""
    cfg = cfg or PreprocessConfig()

    if cfg.debug:
        logger.debug(f"[preprocess] raw: {text!r}")

    # 1) Unicode normalization
    text = normalize_unicode(text, cfg.unicode_form)

    # 2) Lowercase
    if cfg.to_lower:
        text = text.lower()

    # 3) Structural removals
    if cfg.remove_speaker_tags:
        text = remove_speaker_tags(text)
    if cfg.remove_bracketed:
        text = remove_bracketed_content(text)

    # 4) Non-speech/meta
    if cfg.remove_nonspeech_meta:
        text = remove_nonspeech_words(text)

    # 5) URLs/emails
    if cfg.remove_urls_emails:
        text = remove_urls_emails(text, remove_urls=True, remove_emails=True)

    # 6) Phone numbers
    if cfg.normalize_phones:
        text = normalize_phone_numbers(text)

    # 7) Punctuation
    if cfg.remove_punctuation:
        text = remove_punctuation(text, keep_inword=cfg.keep_inword_hyphen_apostrophe)

    # 8) jiwer transforms (optional)
    if cfg.use_jiwer:
        text = apply_jiwer_transforms(text)

    # 9) Whitespace cleanup
    if cfg.collapse_whitespace:
        text = collapse_whitespace(text, strip_edges=cfg.strip_edges)

    if cfg.debug:
        logger.debug(f"[preprocess] clean: {text!r}")
    return text

def preprocess_text_asr_base(text: str, language_code: str = 'en', remove_english_normalization: bool = False) -> str:
    """
    Base ASR text preprocessing function that applies standard cleaning steps.
    
    Args:
        text: Input text to preprocess
        language_code: Language code for language-specific processing
        remove_english_normalization: Skip English-specific normalization
        
    Returns:
        Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Normalize Unicode to NFC form
    text = normalize_unicode(text)
    
    # Step 2: Convert to lowercase
    text = text.lower()
    
    # Step 3: Remove speaker tags and metadata
    text = remove_speaker_tags(text)
    text = re.sub(r'\[[^\]]+\]', '', text)  # Remove bracketed metadata
    
    # Step 4: Remove parentheses content (including nested)
    text = remove_bracketed_content(text)
    
    # Step 5: Language-specific normalization (optional)
    if not remove_english_normalization and language_code in ['en', 'sgen']:
        if WHISPER_AVAILABLE:
            text = _EnglishTextNormalizer()(text)
        text = normalize_text_numbers(text)
    
    # Step 6: Standard jiwer processing
    text = apply_jiwer_standard(text)
    
    # Step 7: Remove non-speech elements
    text = remove_non_speech_elements(text)
    
    # Step 8: Remove extra whitespace and normalize
    text = collapse_whitespace(text)
    
    # Step 9: Remove any remaining control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Step 10: Final cleanup - remove any leading/trailing punctuation
    text = text.strip('.,;:!?')
    
    return text

# =============================================================================
# LANGUAGE-SPECIFIC PREPROCESSORS
# =============================================================================

def preprocess_text_asr_english(text: str) -> str:
    """Preprocess English text for ASR evaluation."""
    return preprocess_text_asr_base(text, language_code='en', remove_english_normalization=False)

def preprocess_text_asr_chinese(text: str) -> str:
    """Preprocess Chinese text for ASR evaluation."""
    text = preprocess_text_asr_base(text, language_code='zh', remove_english_normalization=False)
    text = separate_and_space_chinese(text)
    return text

def preprocess_text_asr_thai(text: str) -> str:
    """Preprocess Thai text for ASR evaluation."""
    text = preprocess_text_asr_base(text, language_code='th', remove_english_normalization=False)
    text = separate_and_space_thai(text)
    return text

def preprocess_text_asr_malay(text: str) -> str:
    """Preprocess Malay text for ASR evaluation."""
    text = preprocess_text_asr_base(text, language_code='ms', remove_english_normalization=True)
    if WHISPER_AVAILABLE:
        text = MalayCodeSwitchEnglishSpellingNormalizer().englishSpellingNormalizer(text)
    return text

def preprocess_text_asr_tamil(text: str) -> str:
    """Preprocess Tamil text for ASR evaluation."""
    return preprocess_text_asr_base(text, language_code='ta', remove_english_normalization=False)

def preprocess_text_asr_vietnamese(text: str) -> str:
    """Preprocess Vietnamese text for ASR evaluation."""
    return preprocess_text_asr_base(text, language_code='vi', remove_english_normalization=True)

def preprocess_text_asr_filipino(text: str) -> str:
    """Preprocess Filipino text for ASR evaluation."""
    return preprocess_text_asr_base(text, language_code='tl', remove_english_normalization=True)

# =============================================================================
# CER-SPECIFIC PREPROCESSORS
# =============================================================================

def preprocess_text_asr_cer_vietnamese(text: str) -> str:
    """Preprocess Vietnamese text for CER evaluation."""
    text = normalize_unicode(text)
    text = text.lower()
    text = remove_bracketed_content(text)
    text = collapse_whitespace(text)
    text = remove_non_speech_elements(text)
    return text.strip()

def preprocess_text_asr_cer_lao(text: str) -> str:
    """Preprocess Lao text for CER evaluation."""
    text = normalize_unicode(text)
    text = text.lower()
    text = remove_bracketed_content(text)
    text = collapse_whitespace(text)
    text = remove_non_speech_elements(text)
    text = re.sub(r'[^\u0E80-\u0EFF\s]', '', text)
    return text.strip()

def preprocess_text_asr_cer_thai(text: str) -> str:
    """Preprocess Thai text for CER evaluation."""
    text = normalize_unicode(text)
    text = text.lower()
    text = remove_bracketed_content(text)
    text = collapse_whitespace(text)
    text = remove_non_speech_elements(text)
    text = re.sub(r'[^\u0E00-\u0E7F\s]', '', text)
    return text.strip()

def preprocess_text_asr_cer_chinese(text: str) -> str:
    """Preprocess Chinese text for CER evaluation."""
    text = normalize_unicode(text)
    text = text.lower()
    text = remove_bracketed_content(text)
    text = collapse_whitespace(text)
    text = remove_non_speech_elements(text)
    text = re.sub(r'[^\u4E00-\u9FFF\w\s]', '', text)
    text = re.sub(r'\s+', '', text)
    return text.strip()

# =============================================================================
# UNIVERSAL AND FACTORY FUNCTIONS
# =============================================================================

def preprocess_text_asr_universal(text: str, language_code: str = 'en') -> str:
    """
    Universal ASR text preprocessing function that handles multiple languages.
    
    Args:
        text: Input text to preprocess
        language_code: Language code for language-specific processing
        
    Returns:
        Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Language-specific preprocessing functions
    language_processors = {
        'en': preprocess_text_asr_english,
        'sgen': preprocess_text_asr_english,  # Singaporean English
        'zh': preprocess_text_asr_chinese,
        'ms': preprocess_text_asr_malay,
        'ta': preprocess_text_asr_tamil,
        'th': preprocess_text_asr_thai,
        'vi': preprocess_text_asr_vietnamese,
        'lo': lambda x: preprocess_text_asr_base(x, language_code='lo', remove_english_normalization=True),
        'km': lambda x: preprocess_text_asr_base(x, language_code='km', remove_english_normalization=True),
        'my': lambda x: preprocess_text_asr_base(x, language_code='my', remove_english_normalization=True),
        'id': lambda x: preprocess_text_asr_base(x, language_code='id', remove_english_normalization=True),
        'tl': preprocess_text_asr_filipino,
    }
    
    # Get the appropriate processor or use base function
    processor = language_processors.get(language_code, preprocess_text_asr_base)
    
    if processor == preprocess_text_asr_base:
        return processor(text, language_code=language_code, remove_english_normalization=True)
    else:
        return processor(text)

def get_cer_preprocessor(language_code: str):
    """
    Get the appropriate CER preprocessor function for a language.
    
    Args:
        language_code: Language code ('vi', 'lo', 'th', 'zh')
        
    Returns:
        Appropriate preprocessing function
    """
    preprocessors = {
        'vi': preprocess_text_asr_cer_vietnamese,
        'lo': preprocess_text_asr_cer_lao, 
        'th': preprocess_text_asr_cer_thai,
        'zh': preprocess_text_asr_cer_chinese
    }
    
    if language_code not in preprocessors:
        raise ValueError(f"Unsupported language code for CER: {language_code}")
    
    return preprocessors[language_code]

def get_asr_preprocessor(language_code: str):
    """
    Get the appropriate ASR preprocessor function for a language.
    
    Args:
        language_code: Language code
        
    Returns:
        Appropriate preprocessing function
    """
    return lambda text: preprocess_text_asr_universal(text, language_code)

# =============================================================================
# MAIN INTERFACE FUNCTIONS
# =============================================================================

def clean_asr_prediction(text: str, language_code: str = 'en', method: str = 'universal') -> str:
    """
    Main function to clean ASR predictions with multiple options.
    
    Args:
        text: Input ASR prediction text
        language_code: Language code for language-specific processing
        method: Cleaning method ('universal', 'base', 'cer', 'legacy', 'generic')
        
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    if method == 'universal':
        return preprocess_text_asr_universal(text, language_code)
    elif method == 'base':
        return preprocess_text_asr_base(text, language_code)
    elif method == 'cer':
        if language_code in ['vi', 'lo', 'th', 'zh']:
            return get_cer_preprocessor(language_code)(text)
        else:
            return preprocess_text_asr_base(text, language_code)
    elif method == 'legacy':
        return preprocess_text_asr_english(text)
    elif method == 'generic':
        return preprocess_text_generic(text)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'universal', 'base', 'cer', 'legacy', or 'generic'")

def batch_preprocess_texts(texts: List[str], language_code: str = 'en', method: str = 'universal') -> List[str]:
    """
    Preprocess a batch of texts.
    
    Args:
        texts: List of texts to preprocess
        language_code: Language code for language-specific processing
        method: Cleaning method
        
    Returns:
        List of preprocessed texts
    """
    return [clean_asr_prediction(text, language_code, method) for text in texts]

def batch_preprocess_generic(texts: List[str], cfg: Optional[PreprocessConfig] = None) -> List[str]:
    """
    Preprocess a batch of texts using the generic preprocessor.
    
    Args:
        texts: List of texts to preprocess
        cfg: Configuration object (uses defaults if None)
        
    Returns:
        List of preprocessed texts
    """
    cfg = cfg or PreprocessConfig()
    return [preprocess_text_generic(text, cfg) for text in texts]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_preprocess_config(**kwargs) -> PreprocessConfig:
    """
    Create a PreprocessConfig with custom settings.
    
    Args:
        **kwargs: Configuration parameters to override defaults
        
    Returns:
        PreprocessConfig instance
    """
    return PreprocessConfig(**kwargs)

def compare_preprocessing_methods(text: str, language_code: str = 'en') -> Dict[str, str]:
    """
    Compare different preprocessing methods on the same text.
    
    Args:
        text: Input text to preprocess
        language_code: Language code for language-specific methods
        
    Returns:
        Dictionary with results from different methods
    """
    results = {}
    
    # Generic preprocessing
    cfg = PreprocessConfig()
    results['generic'] = preprocess_text_generic(text, cfg)
    
    # Base preprocessing
    results['base'] = preprocess_text_asr_base(text, language_code)
    
    # Universal preprocessing
    results['universal'] = preprocess_text_asr_universal(text, language_code)
    
    # Legacy preprocessing
    results['legacy'] = preprocess_text_asr_english(text)
    
    return results

def validate_preprocessed_text(text: str) -> Dict[str, Any]:
    """
    Validate that preprocessed text is clean and ready for evaluation.
    
    Args:
        text: Preprocessed text to validate
        
    Returns:
        Validation results with status and issues
    """
    issues = []
    
    if not text:
        issues.append("Empty text")
    
    if text and len(text.strip()) == 0:
        issues.append("Only whitespace")
    
    # Check for remaining control characters
    if re.search(r'[\x00-\x1f\x7f-\x9f]', text):
        issues.append("Contains control characters")
    
    # Check for excessive punctuation
    if len(re.findall(r'[^\w\s]', text)) > len(text) * 0.3:
        issues.append("Excessive punctuation")
    
    # Check for very long words (potential artifacts)
    words = text.split()
    if any(len(word) > 50 for word in words):
        issues.append("Contains very long words")
    
    return {
        'status': 'valid' if not issues else 'warning',
        'issues': issues,
        'text_length': len(text),
        'word_count': len(words) if text else 0
    }

def detect_primary_script(text: str) -> str:
    """
    Detect the primary script used in text.
    Returns language code based on Unicode ranges.
    """
    char_counts = {
        'zh': len(re.findall(r'[\u4E00-\u9FFF]', text)),  # Chinese
        'th': len(re.findall(r'[\u0E00-\u0E7F]', text)),  # Thai
        'lo': len(re.findall(r'[\u0E80-\u0EFF]', text)),  # Lao
        'vi': len(re.findall(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', text.lower()))  # Vietnamese diacritics
    }
    
    # Return the script with the highest character count
    if max(char_counts.values()) == 0:
        return 'en'  # Default to English if no specific script detected
    
    return max(char_counts, key=char_counts.get)

# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# Legacy function names for backward compatibility
preprocess_text_asr = preprocess_text_asr_english
preprocess_text_asr_code_switch_chinese = preprocess_text_asr_chinese
remove_parentheses = remove_bracketed_content

# Legacy jiwer process for backward compatibility
if JIWER_AVAILABLE:
    all_jiwer_process = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveKaldiNonWords(),
        jiwer.RemovePunctuation()
    ])
else:
    all_jiwer_process = None

# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    samples: List[str] = [
        "<Speaker1>: Umm… visit (see: www.example.com) [laughter] +63 912-345-6789",
        "ỜM… 1.5GB dữ liệu giá 200,000₫ <noise> email: a@b.com",
        "Ah, ano ba — tawagan mo ako sa 09 12 345 6789 (crosstalk)",
        "Music starts… then SILENCE. {breath}  That's it!",
    ]
    
    print("Testing Generic Preprocessor:")
    cfg = PreprocessConfig(debug=True)
    for s in samples:
        print("RAW   :", s)
        print("CLEAN :", preprocess_text_generic(s, cfg))
        print("-" * 60)
    
    print("\nTesting Universal Preprocessor:")
    for s in samples[:2]:  # Test first two samples
        print("RAW   :", s)
        print("CLEAN :", preprocess_text_asr_universal(s, 'en'))
        print("-" * 60)
