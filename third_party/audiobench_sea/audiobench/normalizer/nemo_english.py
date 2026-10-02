# -*- coding: utf-8 -*-
"""
NeMo English ASR Normalizer
- WFST Text Normalization (TN) and Inverse Text Normalization (ITN) via NeMo
- Deterministic vs context-aware control
- Whitelist & caching support for FAR grammars (ClassifyFst / VerbalizeFst)
- Robust fallbacks when NeMo/spaCy/num2words are unavailable
- Hard-silences NeMo's INFO spam (e.g., "Creating ClassifyFst grammars.") during init and calls
"""

import re
import logging
import unicodedata
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import string
import os, sys, io
from contextlib import contextmanager, redirect_stdout, redirect_stderr

# ---- global quiet defaults for NeMo text-processing (import-time) ----
os.environ.setdefault("NEMO_TEXT_PROCESSING_OFFLINE", "1")
os.environ.setdefault("NEMO_TEXT_PROCESSING_QUIET", "1")

try:  # omni_eval patch: only used with AB_USE_SPACY=1 (default off)
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
# Opt-in spaCy usage for speed: set AB_USE_SPACY=1 to enable advanced mode
USE_SPACY = os.environ.get("AB_USE_SPACY", "0") in {"1", "true", "True"}

# Optional: NeMo text processing (WFST TN/ITN)
try:
    from nemo_text_processing.text_normalization.normalize import Normalizer
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

try:
    from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
    NEMO_ITN_AVAILABLE = True
except ImportError:
    NEMO_ITN_AVAILABLE = False

# Optional: num2words for number/currency/time verbalization when TN is disabled
try:
    from num2words import num2words
    NUM2WORDS_AVAILABLE = True
except ImportError:
    NUM2WORDS_AVAILABLE = False

# Lightweight English text normalizer (Whisper-style)
try:
    from .text_normalizer.whisper_english import EnglishTextNormalizer
    WHISPER_NORMALIZER_AVAILABLE = True
except Exception:
    WHISPER_NORMALIZER_AVAILABLE = False

# Handle module name for logger in some import contexts
try:
    module_name = __name__
except NameError:
    module_name = "audiobench.normalizer.nemo_english"

# ---- quiet NeMo text-processing logs via logging config ----
for _name in ("NeMo-text-processing", "nemo_text_processing",
              "nemo_text_processing.text_normalization"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.ERROR)     # or WARNING for some signal
    _lg.propagate = False
    if not _lg.handlers:
        _lg.addHandler(logging.NullHandler())

logger = logging.getLogger(module_name)

# ---- hard silence context manager (logs + stdout/stderr) ----
@contextmanager
def _silence_nemo_io_and_logs():
    """
    Temporarily squelch NeMo text-processing INFO spam by:
    - disabling its loggers
    - redirecting stdout/stderr to /dev/null
    Restores everything on exit.
    """
    nemo_logger_names = [
        "NeMo-text-processing",
        "nemo_text_processing",
        "nemo_text_processing.text_normalization",
        "nemo_text_processing.inverse_text_normalization",
        "nemo_text_processing.text_normalization.normalize",
        "nemo_text_processing.inverse_text_normalization.inverse_normalize",
    ]
    saved = {}
    for name in nemo_logger_names:
        lg = logging.getLogger(name)
        saved[name] = (lg.level, lg.disabled, lg.propagate, list(lg.handlers))
        lg.setLevel(logging.CRITICAL)
        lg.disabled = True
        lg.propagate = False
        for h in lg.handlers[:]:
            lg.removeHandler(h)

    devnull = open(os.devnull, "w")
    try:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield
    finally:
        devnull.close()
        for name, (level, disabled, propagate, handlers) in saved.items():
            lg = logging.getLogger(name)
            lg.setLevel(level)
            lg.disabled = disabled
            lg.propagate = propagate
            for h in handlers:
                lg.addHandler(h)


# ----------------------------
# Configuration
# ----------------------------
@dataclass
class RefinedASRConfig:
    convert_currency_to_words: bool = True
    convert_phone_to_words: bool = True
    convert_time_to_words: bool = True
    convert_numbers_to_words: bool = True
    expand_abbreviations: bool = True
    standardize_and_in_numbers: bool = True
    remove_filler_words: bool = True
    normalize_singapore_currency: bool = True
    clean_multilingual_content: bool = True
    remove_punctuation: bool = True
    lowercase_output: bool = True
    expand_contractions: bool = True
    use_advanced_contraction_expansion: bool = True
    merge_cjk_characters: bool = True
    punctuation_removal_method: str = "unicode"  # ["unicode", "ascii", "preserve_apostrophe"]
    preserve_apostrophes: bool = False

    # NeMo WFST options
    nemo_use_tn: bool = False                 # Use WFST Text Normalization (written -> spoken)
    nemo_use_itn: bool = False               # Use WFST Inverse Text Normalization (spoken -> written)
    nemo_deterministic: bool = True          # Deterministic WFST mode (recommended for ASR eval)
    nemo_whitelist: Optional[str] = None     # Path to whitelist file (src|dst per line)
    nemo_cache_dir: Optional[str] = None     # Cache dir for compiled FAR grammars
    nemo_overwrite_cache: bool = False       # Force rebuild FARs if cache exists


# ----------------------------
# spaCy helpers (optional)
# ----------------------------
def load_spacy_model(model_name: str = "en_core_web_sm"):
    """Load spaCy model with automatic download if needed."""
    if not SPACY_AVAILABLE:
        raise ImportError("spaCy is required for advanced contraction expansion")
    
    # Import spacy locally to ensure it's available
    import spacy
    
    if not spacy.util.is_package(model_name):
        import spacy.cli
        spacy.cli.download(model_name)
    return spacy.load(model_name)


class ImprovedContractionExpander:
    """Contraction expansion with optional spaCy context for edge cases."""
    def __init__(self):
        self.contractions_map = {
            # Common
            "won't": "will not", "Won't": "Will not", "WON'T": "WILL NOT",
            "can't": "cannot", "Can't": "Cannot", "CAN'T": "CANNOT",
            "shan't": "shall not", "Shan't": "Shall not", "SHAN'T": "SHALL NOT",
            "ain't": "is not", "Ain't": "Is not", "AIN'T": "IS NOT",
            "let's": "let us", "Let's": "Let us", "LET'S": "LET US",
            "y'all": "you all", "Y'all": "You all", "Y'ALL": "YOU ALL",
            # I
            "I'm": "I am", "i'm": "i am", "I'M": "I AM",
            "I've": "I have", "i've": "i have", "I'VE": "I HAVE",
            "I'll": "I will", "i'll": "i will", "I'LL": "I WILL",
            "I'd": "I would", "i'd": "i would", "I'D": "I WOULD",
            # You
            "you're": "you are", "You're": "You are", "YOU'RE": "YOU ARE",
            "you've": "you have", "You've": "You have", "YOU'VE": "YOU HAVE",
            "you'll": "you will", "You'll": "You will", "YOU'LL": "YOU WILL",
            "you'd": "you would", "You'd": "You would", "YOU'D": "YOU WOULD",
            # He/She/It
            "he's": "he is", "He's": "He is", "HE'S": "HE IS",
            "she's": "she is", "She's": "She is", "SHE'S": "SHE IS",
            "it's": "it is", "It's": "It is", "IT'S": "IT IS",
            "he'll": "he will", "He'll": "He will", "HE'LL": "HE WILL",
            "she'll": "she will", "She'll": "She will", "SHE'LL": "SHE WILL",
            "it'll": "it will", "It'll": "It will", "IT'LL": "IT WILL",
            "he'd": "he would", "He'd": "He would", "HE'D": "HE WOULD",
            "she'd": "she would", "She'd": "She would", "SHE'D": "SHE WOULD",
            "it'd": "it would", "It'd": "It would", "IT'D": "IT WOULD",
            # We/They
            "we're": "we are", "We're": "We are", "WE'RE": "WE ARE",
            "they're": "they are", "They're": "They are", "THEY'RE": "THEY ARE",
            "we've": "we have", "We've": "We have", "WE'VE": "WE HAVE",
            "they've": "they have", "They've": "They have", "THEY'VE": "THEY HAVE",
            "we'll": "we will", "We'll": "We will", "WE'LL": "WE WILL",
            "they'll": "they will", "They'll": "They will", "THEY'LL": "THEY WILL",
            "we'd": "we would", "We'd": "We would", "WE'D": "WE WOULD",
            "they'd": "they would", "They'd": "They would", "THEY'D": "THEY WOULD",
            # Negatives
            "isn't": "is not", "Isn't": "Is not", "ISN'T": "IS NOT",
            "aren't": "are not", "Aren't": "Are not", "AREN'T": "ARE NOT",
            "wasn't": "was not", "Wasn't": "Was not", "WASN'T": "WAS NOT",
            "weren't": "were not", "Weren't": "Were not", "WEREN'T": "WERE NOT",
            "hasn't": "has not", "Hasn't": "Has not", "HASN'T": "HAS NOT",
            "haven't": "have not", "Haven't": "Have not", "HAVEN'T": "HAVE NOT",
            "hadn't": "had not", "Hadn't": "Had not", "HADN'T": "HAD NOT",
            "doesn't": "does not", "Doesn't": "Does not", "DOESN'T": "DOES NOT",
            "don't": "do not", "Don't": "Do not", "DON'T": "DO NOT",
            "didn't": "did not", "Didn't": "Did not", "DIDN'T": "DID NOT",
            "couldn't": "could not", "Couldn't": "Could not", "COULDN'T": "COULD NOT",
            "shouldn't": "should not", "Shouldn't": "Should not", "SHOULD NOT": "SHOULD NOT",
            "wouldn't": "would not", "Wouldn't": "Would not", "WOULDN'T": "WOULD NOT",
            "mightn't": "might not", "Mightn't": "Might not", "MIGHTN't": "MIGHT NOT",
            "mustn't": "must not", "Mustn't": "Must not", "MUSTN'T": "MUST NOT",
            # Other
            "that's": "that is", "That's": "That is", "THAT'S": "THAT IS",
            "what's": "what is", "What's": "What is", "WHAT'S": "WHAT IS",
            "where's": "where is", "Where's": "Where is", "WHERE'S": "WHERE IS",
            "when's": "when is", "When's": "When is", "WHEN'S": "WHEN IS",
            "why's": "why is", "Why's": "Why is", "WHY'S": "WHY IS",
            "how's": "how is", "How's": "How is", "HOW'S": "HOW IS",
            "there's": "there is", "There's": "There is", "THERE'S": "THERE IS",
            "here's": "here is", "Here's": "Here is", "HERE'S": "HERE IS",
        }
        self.use_spacy = False
        if SPACY_AVAILABLE and USE_SPACY:
            try:
                self.nlp = load_spacy_model()
                self.use_spacy = True
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                self.use_spacy = False

    def expand_contractions_simple(self, text: str) -> str:
        if not isinstance(text, str) or text is None:
            return str(text) if text is not None else ""
        result = text
        for contraction, expansion in self.contractions_map.items():
            pattern = r'\b' + re.escape(contraction) + r'\b'
            result = re.sub(pattern, expansion, result)
        return result

    def expand_contractions_advanced(self, text: str) -> str:
        if not isinstance(text, str) or text is None:
            return str(text) if text is not None else ""
        if not self.use_spacy:
            return self.expand_contractions_simple(text)

        try:
            text = self.expand_contractions_simple(text)
            doc = self.nlp(text)
            tokens = []
            for i, token in enumerate(doc):
                token_text = token.text
                if "'" in token_text and token_text not in self.contractions_map:
                    if token_text.endswith("'s"):
                        base = token_text[:-2]
                        if token.tag_ == "POS":
                            tokens.append(token_text)
                        elif i + 1 < len(doc) and doc[i + 1].tag_ == "VBN":
                            tokens.append(f"{base} has")
                        else:
                            tokens.append(f"{base} is")
                    elif token_text.endswith("'d"):
                        base = token_text[:-2]
                        if i + 1 < len(doc) and doc[i + 1].tag_ == "VBN":
                            tokens.append(f"{base} had")
                        else:
                            tokens.append(f"{base} would")
                    else:
                        tokens.append(token_text)
                else:
                    tokens.append(token_text)
                tokens.append(token.whitespace_)
            return "".join(tokens).strip()
        except Exception as e:
            logger.warning(f"Advanced contraction expansion failed: {e}")
            return self.expand_contractions_simple(text)

    def expand_contractions(self, text: str, use_advanced: bool = True) -> str:
        return (self.expand_contractions_advanced(text)
                if (use_advanced and self.use_spacy)
                else self.expand_contractions_simple(text))


# ----------------------------
# Base normalizer
# ----------------------------
class RefinedBaseNormalizer:
    def __init__(self, asr_config: Optional[RefinedASRConfig] = None):
        self.asr_config = asr_config or RefinedASRConfig()
        self._compile_refined_patterns()
        if self.asr_config.expand_contractions:
            try:
                self.contraction_expander = ImprovedContractionExpander()
                logger.info("Improved contraction expander initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize contraction expander: {e}")
                self.contraction_expander = None
        else:
            self.contraction_expander = None

        # Initialize Whisper-style English text normalizer if available
        self.whisper_text_normalizer = None
        if WHISPER_NORMALIZER_AVAILABLE:
            try:
                self.whisper_text_normalizer = EnglishTextNormalizer()
            except Exception as e:
                logger.warning(f"Failed to initialize Whisper EnglishTextNormalizer: {e}")

    def _compile_refined_patterns(self):
        # Filler words
        self.filler_words = [
            r"\blike\b", r"\bsort of\b", r"\bkind of\b", r"\byou know\b",
            r"\bi mean\b", r"\bwell\b", r"\bbasically\b", r"\bumm+\b",
            r"\buh+\b", r"\bso\b(?=\s+like)", r"\bjust\b(?=\s+like)"
        ]
        self.filler_pattern = re.compile(rf"(?:{'|'.join(self.filler_words)})", flags=re.IGNORECASE)

        # "and" in numbers
        self.number_and_patterns = {
            'hundred_and': re.compile(r'\b(one|two|three|four|five|six|seven|eight|nine)\s+hundred\s+and\s+', re.IGNORECASE),
            'thousand_and': re.compile(r'\b(\w+)\s+thousand\s+and\s+', re.IGNORECASE),
            'million_and': re.compile(r'\b(\w+)\s+million\s+and\s+', re.IGNORECASE),
        }

        # Currency
        self.currency_patterns = {
            'sgd_symbol': re.compile(r'S\$([0-9,]+(?:\.[0-9]{1,2})?)', re.IGNORECASE),
            'usd_symbol': re.compile(r'(?<!S)\$([0-9,]+(?:\.[0-9]{1,2})?)', re.IGNORECASE),
            'sgd_broken': re.compile(r'\bs\s*dollar[s]?\s*([0-9,]+)', re.IGNORECASE),
            'dollar_word_fix': re.compile(r'(\w+)\s+dollars?\s+([0-9,]+)', re.IGNORECASE),
        }

        # Phone numbers (SG + US)
        self.phone_patterns = {
            'formatted': re.compile(r'(\d{3})-(\d{3})-(\d{4})'),
            'spaced': re.compile(r'(\d{3})\s+(\d{3})\s+(\d{4})'),
            'dots': re.compile(r'(\d{3})\.(\d{3})\.(\d{4})'),
            'parentheses': re.compile(r'\((\d{3})\)\s*(\d{3})-?(\d{4})'),
            'long_number': re.compile(r'\b(\d{10,11})\b'),
            'sg_landline_dash': re.compile(r'\b([6-7]\d{3})-(\d{4})\b'),
            'sg_mobile_dash': re.compile(r'\b([89]\d{3})-(\d{4})\b'),
            'sg_with_country': re.compile(r'\+65\s*([6-9]\d{3})[\s-]?(\d{4})\b'),
            'sg_spaced': re.compile(r'\b([6-9]\d{3})\s+(\d{4})\b'),
            'sg_continuous': re.compile(r'\b([6-9]\d{7})\b'),
        }

        # Time
        self.time_patterns = {
            'colon_time': re.compile(r'\b(\d{1,2}):(\d{2})\b'),
            'simple_time': re.compile(r'\b(\d{3,4})\s*(?:am|pm|AM|PM)\b'),
            'am_pm': re.compile(r'\b(AM|PM|am|pm)\b'),
        }

        # Abbreviations
        self.abbrev_patterns = {
            'ceo': re.compile(r'\bCEO\'?s?\b', re.IGNORECASE),
            'mr': re.compile(r'\bMr\.?\b'),
            'mrs': re.compile(r'\bMrs\.?\b'),
            'dr': re.compile(r'\bDr\.?\b'),
            'st': re.compile(r'\bSt\.?\b(?=\s+[A-Z])'),
            'ave': re.compile(r'\bAve\.?\b', re.IGNORECASE),
        }

        # Multilingual
        self.multilingual_patterns = {
            'xml_tags': re.compile(r'<[^>]+>([^<]*)</[^>]+>', re.IGNORECASE),
            'bracket_tags': re.compile(r'\[([^\]]+)\]'),
            'paren_tags': re.compile(r'\(([^)]+)\)'),
            'tamil_unicode': re.compile(r'[\u0B80-\u0BFF]+'),
            'chinese_unicode': re.compile(r'[\u4e00-\u9fff]+'),
        }

        # Digit to word mapping
        self.digit_to_words = {str(d): w for d, w in enumerate(
            ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        )}

    # ----- atomic helpers -----
    def expand_contractions(self, text: str) -> str:
        if not self.asr_config.expand_contractions or text is None:
            return text if text is not None else ""
        if not isinstance(text, str):
            text = str(text)
        if self.contraction_expander is None:
            return text
        try:
            return self.contraction_expander.expand_contractions(
                text,
                use_advanced=self.asr_config.use_advanced_contraction_expansion
            )
        except Exception as e:
            logger.warning(f"Contraction expansion failed: {e}")
            return text

    def normalize_whitespace(self, text: str) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        try:
            text = re.sub(r'[\n\r]+', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        except Exception as e:
            logger.warning(f"Whitespace normalization failed: {e}")
            return text

    def remove_enhanced_filler_words(self, text: str) -> str:
        if not self.asr_config.remove_filler_words or text is None:
            return text if text is not None else ""
        if not isinstance(text, str):
            text = str(text)
        try:
            text = self.filler_pattern.sub("", text)
            return re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            logger.warning(f"Enhanced filler word removal failed: {e}")
            return text

    def standardize_number_and(self, text: str) -> str:
        if not self.asr_config.standardize_and_in_numbers or text is None:
            return text if text is not None else ""
        if not isinstance(text, str):
            text = str(text)
        try:
            text = self.number_and_patterns['hundred_and'].sub(r'\1 hundred ', text)
            text = self.number_and_patterns['thousand_and'].sub(r'\1 thousand ', text)
            text = self.number_and_patterns['million_and'].sub(r'\1 million ', text)
            return re.sub(r'\s+', ' ', text)
        except Exception as e:
            logger.warning(f"Number 'and' standardization failed: {e}")
            return text

    def fix_singapore_currency(self, text: str) -> str:
        if not self.asr_config.normalize_singapore_currency or text is None:
            return text if text is not None else ""
        if not isinstance(text, str):
            text = str(text)
        try:
            text = self.currency_patterns['sgd_broken'].sub(r'\1 singapore dollars', text)
            text = self.currency_patterns['dollar_word_fix'].sub(r'\2 \1 dollars', text)
            text = re.sub(r'\bs\s+dollars?\b', 'singapore dollars', text, flags=re.IGNORECASE)
            return text
        except Exception as e:
            logger.warning(f"Singapore currency fix failed: {e}")
            return text

    def clean_multilingual_content(self, text: str) -> str:
        if not self.asr_config.clean_multilingual_content or text is None:
            return text if text is not None else ""
        if not isinstance(text, str):
            text = str(text)
        try:
            text = self.multilingual_patterns['xml_tags'].sub(r'\1', text)
            text = self.multilingual_patterns['bracket_tags'].sub(r'\1', text)
            # Example handling; keep neutral; do not alter semantics aggressively.
            return text
        except Exception as e:
            logger.warning(f"Multilingual content cleaning failed: {e}")
            return text

    def convert_currency_to_words_enhanced(self, text: str) -> str:
        if not self.asr_config.convert_currency_to_words or text is None:
            return text if text is not None else ""
        if not isinstance(text, str):
            text = str(text)
        if not NUM2WORDS_AVAILABLE:
            return text
        # Fast-exit if no currency indicators present
        if ('$' not in text) and ('S$' not in text):
            return text

        def safe_num_to_words(num_str: str) -> str:
            try:
                clean_num = num_str.replace(',', '')
                num = float(clean_num)
                if num == int(num):
                    return num2words(int(num))
                dollars = int(num)
                cents = int(round((num - dollars) * 100))
                return f"{num2words(dollars)} point {num2words(cents)}" if cents else num2words(dollars)
            except (ValueError, OverflowError):
                return num_str

        try:
            text = self.currency_patterns['sgd_symbol'].sub(
                lambda m: f"{safe_num_to_words(m.group(1))} singapore dollars", text
            )
            text = self.currency_patterns['usd_symbol'].sub(
                lambda m: f"{safe_num_to_words(m.group(1))} dollars", text
            )
            return text
        except Exception as e:
            logger.warning(f"Enhanced currency conversion failed: {e}")
            return text

    def convert_phone_to_words(self, text: str) -> str:
        if not self.asr_config.convert_phone_to_words or text is None:
            return text if text is not None else ""
        if not isinstance(text, str):
            text = str(text)

        # Fast exit if there are no digits at all
        if not any(ch.isdigit() for ch in text):
            return text

        def digits_to_words(digits: str) -> str:
            return ' '.join(self.digit_to_words.get(d, d) for d in digits)

        try:
            # SG patterns
            for pname in ['sg_with_country', 'sg_landline_dash', 'sg_mobile_dash', 'sg_spaced', 'sg_continuous']:
                if pname in self.phone_patterns:
                    pattern = self.phone_patterns[pname]

                    def repl(m):
                        if pname == 'sg_with_country':
                            digits = '65' + m.group(1) + m.group(2)
                        elif pname == 'sg_continuous':
                            digits = m.group(1)
                            if len(digits) == 8 and digits[0] in '6789':
                                return digits_to_words(digits)
                            return m.group(0)
                        else:
                            digits = m.group(1) + m.group(2)
                        return digits_to_words(digits)

                    text = pattern.sub(repl, text)

            # US patterns
            for pname in ['formatted', 'spaced', 'dots', 'parentheses']:
                if pname in self.phone_patterns:
                    pattern = self.phone_patterns[pname]

                    def repl(m):
                        digits = m.group(1) + m.group(2) + m.group(3)
                        return digits_to_words(digits)

                    text = pattern.sub(repl, text)

            # Long unformatted
            text = self.phone_patterns['long_number'].sub(
                lambda m: digits_to_words(m.group(1)) if len(m.group(1)) in [8, 10, 11] else m.group(0), text
            )
            return text
        except Exception as e:
            logger.warning(f"Enhanced phone-to-words conversion failed: {e}")
            return text

    def convert_time_to_words(self, text: str) -> str:
        if not self.asr_config.convert_time_to_words or text is None:
            return text if text is not None else ""
        if not isinstance(text, str):
            text = str(text)
        if not NUM2WORDS_AVAILABLE:
            return text

        def time_to_words(hour: str, minute: str) -> str:
            try:
                h = int(hour.lstrip('0') or '0')
                m = int(minute)
                hour_words = num2words(h)
                if m == 0:
                    return f"{hour_words} o'clock"
                minute_words = f"oh {num2words(m)}" if m < 10 else num2words(m)
                return f"{hour_words} {minute_words}"
            except (ValueError, TypeError):
                return f"{hour} {minute}"

        # Fast exits: skip if no ':' and no am/pm tokens
        low = text.lower()
        if (':' not in low) and ('am' not in low) and ('pm' not in low):
            return text
        try:
            text = self.time_patterns['colon_time'].sub(
                lambda m: time_to_words(m.group(1), m.group(2)), text
            )
            def simple_replace(m):
                t = m.group(1)
                if len(t) == 3:
                    h, mm = t[0], t[1:]
                elif len(t) == 4:
                    h, mm = t[:2], t[2:]
                else:
                    return m.group(0)
                return time_to_words(h, mm)
            text = self.time_patterns['simple_time'].sub(simple_replace, text)

            time_am_pm_pattern = re.compile(
                r'(\b(?:\d{1,2}:\d{2}|\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirty|fifteen|forty|fifty|o\'?clock)\s*)(am|pm)\b',
                re.IGNORECASE
            )
            text = time_am_pm_pattern.sub(lambda m: f"{m.group(1)}{'a m' if m.group(2).lower()=='am' else 'p m'}", text)
            return text
        except Exception as e:
            logger.warning(f"Time-to-words conversion failed: {e}")
            return text

    def convert_numbers_to_words(self, text: str) -> str:
        if not self.asr_config.convert_numbers_to_words or text is None:
            return text if text is not None else ""
        if not isinstance(text, str):
            text = str(text)
        if not NUM2WORDS_AVAILABLE:
            return text
        # Fast-exit if no digits
        if not any(ch.isdigit() for ch in text):
            return text
        try:
            def repl(m):
                try:
                    num = int(m.group(1))
                    if 1 <= num <= 999_999:
                        return num2words(num)
                    return m.group(0)
                except (ValueError, TypeError):
                    return m.group(0)
            return re.sub(r'\b(\d{1,6})\b', repl, text)
        except Exception as e:
            logger.warning(f"Numbers-to-words conversion failed: {e}")
            return text

    def expand_abbreviations(self, text: str) -> str:
        if not self.asr_config.expand_abbreviations or text is None:
            return text if text is not None else ""
        if not isinstance(text, str):
            text = str(text)
        try:
            abbreviations = {
                self.abbrev_patterns['ceo']: 'chief executive officer',
                self.abbrev_patterns['mr']: 'mister',
                self.abbrev_patterns['mrs']: 'missus',
                self.abbrev_patterns['dr']: 'doctor',
                self.abbrev_patterns['st']: 'street',
                self.abbrev_patterns['ave']: 'avenue',
            }
            for pattern, replacement in abbreviations.items():
                text = pattern.sub(replacement, text)
            return text
        except Exception as e:
            logger.warning(f"Abbreviation expansion failed: {e}")
            return text

    def fix_compound_number_spacing(self, text: str) -> str:
        if not isinstance(text, str) or text is None:
            return str(text) if text is not None else ""
        try:
            compounds = [
                'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'
            ]
            ones = [
                'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'
            ]
            for tens in compounds:
                for one in ones:
                    text = re.sub(fr'\b{tens}{one}\b', f'{tens} {one}', text, flags=re.IGNORECASE)
            return text
        except Exception as e:
            logger.warning(f"Compound number spacing fix failed: {e}")
            return text

    def merge_cjk_characters_enhanced(self, text: str) -> str:
        if not self.asr_config.merge_cjk_characters or text is None:
            return text if text is not None else ""
        if not isinstance(text, str):
            text = str(text)
        try:
            cjk_pattern = re.compile(r'([\u4e00-\u9fff]) (?=[\u4e00-\u9fff])')
            return cjk_pattern.sub(r'\1', text)
        except Exception as e:
            logger.warning(f"Enhanced CJK merging failed: {e}")
            return text

    def remove_punctuation_enhanced(self, text: str) -> str:
        if not self.asr_config.remove_punctuation or text is None:
            return text if text is not None else ""
        if not isinstance(text, str):
            text = str(text)
        try:
            method = self.asr_config.punctuation_removal_method
            if method == "ascii":
                if self.asr_config.preserve_apostrophes:
                    punctuation_chars = string.punctuation.replace("'", "")
                    text = text.translate(str.maketrans('', '', punctuation_chars))
                else:
                    text = text.translate(str.maketrans('', '', string.punctuation))
            elif method == "unicode":
                result = []
                for ch in text:
                    if unicodedata.category(ch).startswith('P'):
                        if self.asr_config.preserve_apostrophes and ch == "'":
                            result.append(ch)
                        else:
                            result.append(' ')
                    else:
                        result.append(ch)
                text = ''.join(result)
            elif method == "preserve_apostrophe":
                punctuation_chars = string.punctuation.replace("'", "")
                translator = str.maketrans(punctuation_chars, ' ' * len(punctuation_chars))
                text = text.translate(translator)

            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        except Exception as e:
            logger.warning(f"Enhanced punctuation removal failed: {e}")
            return text

    def apply_refined_asr_standardization(self, text: str) -> str:
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        try:
            text = self.remove_enhanced_filler_words(text)
            text = self.clean_multilingual_content(text)
            text = self.merge_cjk_characters_enhanced(text)
            text = self.convert_currency_to_words_enhanced(text)
            text = self.fix_singapore_currency(text)
            text = self.convert_phone_to_words(text)
            text = self.convert_time_to_words(text)
            text = self.convert_numbers_to_words(text)
            text = self.expand_abbreviations(text)
            text = self.standardize_number_and(text)
            text = self.fix_compound_number_spacing(text)
            text = self.remove_punctuation_enhanced(text)
            text = self.normalize_whitespace(text)
            if self.asr_config.lowercase_output:
                text = text.lower()
            return text.strip()
        except Exception as e:
            logger.error(f"Enhanced ASR standardization failed: {e}")
            return text if isinstance(text, str) else ""


# ----------------------------
# NeMo-powered English normalizer
# ----------------------------
class RefinedNeMo_English_ASR_Normaliser(RefinedBaseNormalizer):
    def __init__(self,
                 language: str = 'en',
                 cache_dir: Optional[str] = None,
                 use_lm_context: bool = False,   # kept for API compatibility (unused)
                 enable_disfluency_removal: bool = True,
                 asr_config: Optional[RefinedASRConfig] = None):
        super().__init__(asr_config)
        if not language:
            language = 'en'
        self.language = language
        self.enable_disfluency_removal = enable_disfluency_removal

        # Internal TN/ITN handles
        self.text_normalizer: Optional[Normalizer] = None
        self.inverse_normalizer: Optional[InverseNormalizer] = None

        # Preprocessing patterns for disfluencies
        self._compile_preprocessing_patterns()

        # Ensure effective cache_dir precedence (constructor arg > config)
        if cache_dir is not None and isinstance(cache_dir, str) and cache_dir.strip():
            effective_cache = cache_dir.strip()
        elif hasattr(self.asr_config, 'nemo_cache_dir') and self.asr_config.nemo_cache_dir and isinstance(self.asr_config.nemo_cache_dir, str):
            effective_cache = self.asr_config.nemo_cache_dir.strip()
        else:
            effective_cache = None

        # --- NeMo TN/ITN initialization with timeout & caching ---
        if NEMO_AVAILABLE and (self.asr_config.nemo_use_tn or self.asr_config.nemo_use_itn):
            # hard mute NeMo loggers + warnings (backup to context manager)
            import warnings
            warnings.filterwarnings("ignore", module="nemo_text_processing")

            if effective_cache:
                try:
                    os.makedirs(effective_cache, exist_ok=True)
                    os.environ['NEMO_TEXT_PROCESSING_CACHE_DIR'] = str(effective_cache)
                    logger.info(f"Using NeMo cache directory: {effective_cache}")
                except Exception as e:
                    logger.warning(f"Failed to set up cache directory '{effective_cache}': {e}")
                    effective_cache = None
            else:
                logger.info("No cache directory specified, using default NeMo cache")

            init_exc = [None]
            init_success = [False]

            def _init_tn_itn():
                try:
                    # TN
                    if self.asr_config.nemo_use_tn:
                        logger.debug("Initializing NeMo Text Normalizer...")
                        cache_param = effective_cache if effective_cache and isinstance(effective_cache, str) else None
                        with _silence_nemo_io_and_logs():
                            self.text_normalizer = Normalizer(
                                input_case='cased',
                                lang=language,
                                cache_dir=cache_param,
                                whitelist=self.asr_config.nemo_whitelist,
                                overwrite_cache=self.asr_config.nemo_overwrite_cache,
                                deterministic=self.asr_config.nemo_deterministic,
                                lm=False,
                            )
                        logger.debug("NeMo Text Normalizer initialized successfully")

                    # ITN
                    if self.asr_config.nemo_use_itn and NEMO_ITN_AVAILABLE:
                        logger.debug("Initializing NeMo Inverse Text Normalizer...")
                        cache_param = effective_cache if effective_cache and isinstance(effective_cache, str) else None
                        with _silence_nemo_io_and_logs():
                            self.inverse_normalizer = InverseNormalizer(
                                lang=language,
                                cache_dir=cache_param,
                                overwrite_cache=self.asr_config.nemo_overwrite_cache,
                                deterministic=self.asr_config.nemo_deterministic,
                            )
                        logger.debug("NeMo Inverse Text Normalizer initialized successfully")

                    init_success[0] = True
                except Exception as e:
                    init_exc[0] = e
                    logger.warning(f"NeMo initialization error: {e}")
                    if "expected str, bytes or os.PathLike object" in str(e) and effective_cache:
                        logger.info("Retrying NeMo initialization without cache directory...")
                        try:
                            if self.asr_config.nemo_use_tn:
                                with _silence_nemo_io_and_logs():
                                    self.text_normalizer = Normalizer(
                                        input_case='cased',
                                        lang=language,
                                        cache_dir=None,
                                        whitelist=self.asr_config.nemo_whitelist,
                                        overwrite_cache=self.asr_config.nemo_overwrite_cache,
                                        deterministic=self.asr_config.nemo_deterministic,
                                        lm=False,
                                    )
                            logger.debug("NeMo Text Normalizer initialized successfully (no cache)")
                            init_success[0] = True
                        except Exception as e2:
                            logger.warning(f"NeMo initialization without cache also failed: {e2}")

            # Initialize directly without threading to reduce process count
            try:
                _init_tn_itn()
            except Exception as e:
                logger.warning(f"NeMo initialization failed: {e}")

            # Check if initialization was successful
            if not init_success[0]:
                logger.error("NeMo TN/ITN initialization failed; falling back to basic normalization")
                self.text_normalizer = None
                self.inverse_normalizer = None
            elif init_exc[0] is not None:
                logger.error(f"NeMo TN/ITN initialization failed: {init_exc[0]}; falling back to basic normalization")
                self.text_normalizer = None
                self.inverse_normalizer = None
            elif not init_success[0]:
                logger.debug("NeMo TN/ITN initialization completed but no normalizers were created; using basic normalization")
            else:
                logger.info("NeMo TN/ITN initialization completed successfully")
        else:
            if not (self.asr_config.nemo_use_tn or self.asr_config.nemo_use_itn):
                logger.info("NeMo TN/ITN disabled in config - skipping initialization")
            else:
                logger.warning("nemo_text_processing not available; TN/ITN disabled")

    def _compile_preprocessing_patterns(self):
        self.english_disfluencies = [
            r"\buh+\b", r"\bum+\b", r"\bumm+\b", r"\berr+\b", r"\berm+\b",
            r"\bah+\b", r"\beh+\b", r"\bmm+\b", r"\bmmhmm+\b", r"\bhmm+\b",
            r"\boh\b"
        ]
        self.disfluency_pattern = re.compile(rf"(?:{'|'.join(self.english_disfluencies)})", flags=re.IGNORECASE)

    def remove_english_disfluencies(self, text: str) -> str:
        if not self.enable_disfluency_removal or text is None:
            return text if text is not None else ""
        if not isinstance(text, str):
            text = str(text)
        try:
            cleaned = self.disfluency_pattern.sub("", text)
            return re.sub(r"\s+", " ", cleaned).strip()
        except Exception as e:
            logger.warning(f"English disfluency removal failed: {e}")
            return text

    def preprocess_english(self, text: str) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        try:
            # Minimize passes: only one whitespace normalization at end
            x = self.remove_english_disfluencies(text)
            x = self.expand_contractions(x)
            return self.normalize_whitespace(x)
        except Exception as e:
            logger.error(f"Enhanced English preprocessing failed: {e}")
            return str(text)

    # -------- TN / ITN ----------
    def text_normalize(self, text: str) -> str:
        """
        TN: written -> spoken (e.g., "123" -> "one hundred twenty three").
        Uses NeMo WFST TN if available; otherwise falls back.
        """
        if not isinstance(text, str) or text is None:
            return "" if text is None else str(text)

        if not (NEMO_AVAILABLE and self.text_normalizer is not None and self.asr_config.nemo_use_tn):
            return self._basic_english_normalize(text)

        result, exc = [None], [None]

        def _run():
            try:
                with _silence_nemo_io_and_logs():
                    result[0] = self.text_normalizer.normalize(
                        text,
                        verbose=False,
                        punct_post_process=True,
                    )
            except Exception as e:
                exc[0] = e

        # Run directly without threading to reduce process count
        try:
            _run()
        except Exception as e:
            logger.warning(f"Text normalization failed: {e}")

        if exc[0] is not None:
            logger.warning(f"NeMo TN failed for text: '{text[:50]}...'; using basic normalize")
            return self._basic_english_normalize(text)
        elif exc[0] is not None:
            logger.warning(f"NeMo TN failed: {exc[0]}; using basic normalize")
            return self._basic_english_normalize(text)
        
        return result[0] if result[0] is not None else text

    def inverse_text_normalize(self, text: str) -> str:
        """
        ITN: spoken -> written (e.g., "one hundred twenty three" -> "123").
        Useful after ASR when you want canonical written forms.
        """
        if not isinstance(text, str) or text is None:
            return "" if text is None else str(text)

        if not (NEMO_ITN_AVAILABLE and self.inverse_normalizer is not None and self.asr_config.nemo_use_itn):
            return text

        res, exc = [None], [None]

        def _run():
            try:
                with _silence_nemo_io_and_logs():
                    res[0] = self.inverse_normalizer.inverse_normalize(text, verbose=False)
            except Exception as e:
                exc[0] = e

        # Run directly without threading to reduce process count
        try:
            _run()
        except Exception as e:
            logger.warning(f"Inverse text normalization failed: {e}")

        if exc[0] is not None:
            logger.warning(f"NeMo ITN failed for text: '{text[:50]}...'; returning original")
            return text
        elif exc[0] is not None:
            logger.warning(f"NeMo ITN failed: {exc[0]}; returning original")
            return text
        
        return res[0] if res[0] is not None else text

    def _basic_english_normalize(self, text: str) -> str:
        """Basic English normalization when NeMo is not available."""
        if not text:
            return ""
        try:
            # Basic text cleaning
            text = re.sub(r'\s+', ' ', text.strip())
            # Keep basic punctuation and alphanumeric characters
            text = re.sub(r'[^\w\s\.,!?;:\-]', '', text)
            return text
        except Exception as e:
            logger.warning(f"Basic English normalization failed: {e}")
            return str(text).strip()

    # -------- Pipeline entry ----------
    def normalize_for_asr_eval(self,
                               text: str,
                               use_preprocessing: bool = True,
                               apply_asr_standardization: bool = True) -> str:
        """
        Main entry for ASR evaluation normalization.
        - Preprocess (disfluencies, contractions)
        - NeMo TN (optional)
        - Domain standardizers (some skipped if TN already applied)
        - Punct removal, lowercasing, whitespace
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        try:
            # 1) Preprocess
            x = self.preprocess_english(text) if use_preprocessing else text

            # 2) Use Whisper-style English text normalizer (fast, regex-based)
            if self.whisper_text_normalizer is not None:
                try:
                    x = self.whisper_text_normalizer(x)
                except Exception as e:
                    logger.warning(f"Whisper EnglishTextNormalizer failed: {e}")

            # 3) Domain tweaks
            if apply_asr_standardization:
                if not self.asr_config.nemo_use_tn:
                    x = self.convert_currency_to_words_enhanced(x)
                    x = self.convert_phone_to_words(x)
                    x = self.convert_time_to_words(x)
                    x = self.convert_numbers_to_words(x)

                x = self.expand_abbreviations(x)
                x = self.standardize_number_and(x)
                x = self.fix_compound_number_spacing(x)
                x = self.clean_multilingual_content(x)
                x = self.merge_cjk_characters_enhanced(x)
                x = self.remove_punctuation_enhanced(x)
                x = self.normalize_whitespace(x)

            if self.asr_config.lowercase_output:
                x = x.lower()

            return x.strip()
        except Exception as e:
            logger.error(f"Refined ASR normalization failed: {e}")
            return text.lower().strip() if isinstance(text, str) else ""
