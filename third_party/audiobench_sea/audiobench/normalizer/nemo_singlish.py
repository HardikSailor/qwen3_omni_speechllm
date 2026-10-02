"""
NeMo Singlish ASR Normalizer
"""

import re
import logging
import unicodedata
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import string

# SpaCy imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available for advanced contraction expansion")

# NeMo imports
try:
    from nemo_text_processing.text_normalization.normalize import Normalizer
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

# Num2Words imports
try:
    from num2words import num2words
    NUM2WORDS_AVAILABLE = True
except ImportError:
    NUM2WORDS_AVAILABLE = False

# Handle case where __name__ might not be available during direct file import
try:
    module_name = __name__
except NameError:
    module_name = "audiobench.normalizer.nemo_singlish"

logger = logging.getLogger(module_name)

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
    punctuation_removal_method: str = "unicode"
    preserve_apostrophes: bool = False

    # NeMo WFST options (disabled by default)
    nemo_use_tn: bool = False                 # Use WFST Text Normalization (written -> spoken)
    nemo_use_itn: bool = False               # Use WFST Inverse Text Normalization (spoken -> written)
    nemo_deterministic: bool = True          # Deterministic WFST mode (recommended for ASR eval)
    nemo_whitelist: Optional[str] = None     # Path to whitelist file (src|dst per line)
    nemo_cache_dir: Optional[str] = None     # Cache dir for compiled FAR grammars
    nemo_overwrite_cache: bool = False       # Force rebuild FARs if cache exists

def load_spacy_model(model_name="en_core_web_sm"):
    """Load spaCy model with automatic download if needed"""
    if not SPACY_AVAILABLE:
        raise ImportError("spaCy is required for advanced contraction expansion")
    
    # Import spacy locally to ensure it's available
    import spacy
    
    if not spacy.util.is_package(model_name):
        print(f"Downloading spaCy model '{model_name}'...")
        spacy.cli.download(model_name)
    return spacy.load(model_name)

class ImprovedContractionExpander:
    def __init__(self):
        self.contractions_map = {
            # Common contractions
            "won't": "will not", "Won't": "Will not", "WON'T": "WILL NOT",
            "can't": "cannot", "Can't": "Cannot", "CAN'T": "CANNOT",
            "shan't": "shall not", "Shan't": "Shall not", "SHAN'T": "SHALL NOT",
            "ain't": "is not", "Ain't": "Is not", "AIN'T": "IS NOT",
            "let's": "let us", "Let's": "Let us", "LET'S": "LET US",
            "y'all": "you all", "Y'all": "You all", "Y'ALL": "YOU ALL",
            
            # I contractions
            "I'm": "I am", "i'm": "i am", "I'M": "I AM",
            "I've": "I have", "i've": "i have", "I'VE": "I HAVE",
            "I'll": "I will", "i'll": "i will", "I'LL": "I WILL",
            "I'd": "I would", "i'd": "i would", "I'D": "I WOULD",
            
            # You contractions
            "you're": "you are", "You're": "You are", "YOU'RE": "YOU ARE",
            "you've": "you have", "You've": "You have", "YOU'VE": "YOU HAVE",
            "you'll": "you will", "You'll": "You will", "YOU'LL": "YOU WILL",
            "you'd": "you would", "You'd": "You would", "YOU'D": "YOU WOULD",
            
            # He/She/It contractions
            "he's": "he is", "He's": "He is", "HE'S": "HE IS",
            "she's": "she is", "She's": "She is", "SHE'S": "SHE IS",
            "it's": "it is", "It's": "It is", "IT'S": "IT IS",
            "he'll": "he will", "He'll": "He will", "HE'LL": "HE WILL",
            "she'll": "she will", "She'll": "She will", "SHE'LL": "SHE WILL",
            "it'll": "it will", "It'll": "It will", "IT'LL": "IT WILL",
            "he'd": "he would", "He'd": "He would", "HE'D": "HE WOULD",
            "she'd": "she would", "She'd": "She would", "SHE'D": "SHE WOULD",
            "it'd": "it would", "It'd": "It would", "IT'D": "IT WOULD",
            
            # We/They contractions
            "we're": "we are", "We're": "We are", "WE'RE": "WE ARE",
            "they're": "they are", "They're": "They are", "THEY'RE": "THEY ARE",
            "we've": "we have", "We've": "We have", "WE'VE": "WE HAVE",
            "they've": "they have", "They've": "They have", "THEY'VE": "THEY HAVE",
            "we'll": "we will", "We'll": "We will", "WE'LL": "WE WILL",
            "they'll": "they will", "They'll": "They will", "THEY'LL": "THEY WILL",
            "we'd": "we would", "We'd": "We would", "WE'D": "WE WOULD",
            "they'd": "they would", "They'd": "They would", "THEY'D": "THEY WOULD",
            
            # Negative contractions
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
            "shouldn't": "should not", "Shouldn't": "Should not", "SHOULDN'T": "SHOULD NOT",
            "wouldn't": "would not", "Wouldn't": "Would not", "WOULDN'T": "WOULD NOT",
            "mightn't": "might not", "Mightn't": "Might not", "MIGHTN't": "MIGHT NOT",
            "mustn't": "must not", "Mustn't": "Must not", "MUSTN'T": "MUST NOT",
            
            # Other common contractions
            "that's": "that is", "That's": "That is", "THAT'S": "THAT IS",
            "what's": "what is", "What's": "What is", "WHAT'S": "WHAT IS",
            "where's": "where is", "Where's": "Where is", "WHERE'S": "WHERE IS",
            "when's": "when is", "When's": "When is", "WHEN'S": "WHEN IS",
            "why's": "why is", "Why's": "Why is", "WHY'S": "WHY IS",
            "how's": "how is", "How's": "How is", "HOW'S": "HOW IS",
            "there's": "there is", "There's": "There is", "THERE'S": "THERE IS",
            "here's": "here is", "Here's": "Here is", "HERE'S": "HERE IS",
        }
        
        # Advanced contractions that need context (for spaCy-based expansion)
        self.context_contractions = {
            "'s": {"is": ["he", "she", "it", "that", "what", "who", "where", "when", "why", "how", "there", "here"],
                  "has": ["he", "she", "it", "that", "what", "who"]},
            "'d": {"had": ["past_participle_follows"], 
                  "would": ["default"]},
            "'ve": {"have": ["default"]},
            "'ll": {"will": ["default"]},
            "'re": {"are": ["default"]},
        }
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = load_spacy_model()
                self.use_spacy = True
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                self.use_spacy = False
        else:
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
                    # This is for edge cases not covered by simple mapping
                    if token_text.endswith("'s"):
                        base = token_text[:-2]
                        if token.tag_ == "POS":
                            tokens.append(token_text)
                        elif i + 1 < len(doc) and doc[i+1].tag_ == "VBN":
                            tokens.append(f"{base} has")
                        else:
                            tokens.append(f"{base} is")
                    elif token_text.endswith("'d"):
                        base = token_text[:-2]
                        if i + 1 < len(doc) and doc[i+1].tag_ == "VBN":
                            tokens.append(f"{base} had")
                        else:
                            tokens.append(f"{base} would")
                    else:
                        tokens.append(token_text)
                else:
                    tokens.append(token_text)
                
                # Add whitespace
                tokens.append(token.whitespace_)
            
            result = "".join(tokens).strip()
            return result
            
        except Exception as e:
            logger.warning(f"Advanced contraction expansion failed: {e}")
            return self.expand_contractions_simple(text)
    
    def expand_contractions(self, text: str, use_advanced: bool = True) -> str:
        if use_advanced and self.use_spacy:
            return self.expand_contractions_advanced(text)
        else:
            return self.expand_contractions_simple(text)

class RefinedBaseNormalizer:
    # Class-level cached singleton. The expander loads ~10-50 ms of regex
    # state plus optional spaCy resources; without caching, instantiating
    # one per normalize() call dominates scoring time (we saw ~30 min on a
    # 6k-sample sgen dataset purely from re-initialization).
    _shared_contraction_expander = None
    _shared_contraction_expander_logged = False

    def __init__(self, asr_config: Optional[RefinedASRConfig] = None):
        self.asr_config = asr_config or RefinedASRConfig()
        self._compile_refined_patterns()

        if self.asr_config.expand_contractions:
            try:
                if RefinedBaseNormalizer._shared_contraction_expander is None:
                    RefinedBaseNormalizer._shared_contraction_expander = (
                        ImprovedContractionExpander()
                    )
                self.contraction_expander = RefinedBaseNormalizer._shared_contraction_expander
                if not RefinedBaseNormalizer._shared_contraction_expander_logged:
                    logger.info("Improved contraction expander initialized (cached)")
                    RefinedBaseNormalizer._shared_contraction_expander_logged = True
            except Exception as e:
                logger.warning(f"Failed to initialize contraction expander: {e}")
                self.contraction_expander = None
        else:
            self.contraction_expander = None
    
    def _compile_refined_patterns(self):
        
        # Filler word patterns
        self.filler_words = [
            r"\blike\b", r"\bsort of\b", r"\bkind of\b", r"\byou know\b", 
            r"\bi mean\b", r"\bwell\b", r"\bbasically\b", r"\bumm\b", 
            r"\buh\b", r"\bso\b(?=\s+like)", r"\bjust\b(?=\s+like)"
        ]
        self.filler_pattern = re.compile(
            rf"(?:{'|'.join(self.filler_words)})", 
            flags=re.IGNORECASE
        )
        
        # Number "and" standardization patterns
        self.number_and_patterns = {
            'hundred_and': re.compile(r'\b(one|two|three|four|five|six|seven|eight|nine)\s+hundred\s+and\s+', re.IGNORECASE),
            'thousand_and': re.compile(r'\b(\w+)\s+thousand\s+and\s+', re.IGNORECASE),
            'million_and': re.compile(r'\b(\w+)\s+million\s+and\s+', re.IGNORECASE),
        }
        
        # Currency patterns
        self.currency_patterns = {
            'sgd_symbol': re.compile(r'S\$([0-9,]+(?:\.[0-9]{1,2})?)', re.IGNORECASE),
            'usd_symbol': re.compile(r'(?<!S)\$([0-9,]+(?:\.[0-9]{1,2})?)', re.IGNORECASE),
            'sgd_broken': re.compile(r'\bs\s*dollar[s]?\s*([0-9,]+)', re.IGNORECASE),
            'dollar_word_fix': re.compile(r'(\w+)\s+dollars?\s+([0-9,]+)', re.IGNORECASE),
        }
        
        # Phone number patterns
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
        
        # Time patterns
        self.time_patterns = {
            'colon_time': re.compile(r'\b(\d{1,2}):(\d{2})\b'),
            'simple_time': re.compile(r'\b(\d{3,4})\s*(?:am|pm|AM|PM)\b'),
            'am_pm': re.compile(r'\b(AM|PM|am|pm)\b'),
        }
        
        # Abbreviation patterns
        self.abbrev_patterns = {
            'ceo': re.compile(r'\bCEO\'?s?\b', re.IGNORECASE),
            'mr': re.compile(r'\bMr\.?\b'),
            'mrs': re.compile(r'\bMrs\.?\b'),
            'dr': re.compile(r'\bDr\.?\b'),
            'st': re.compile(r'\bSt\.?\b(?=\s+[A-Z])'),
            'ave': re.compile(r'\bAve\.?\b', re.IGNORECASE),
        }
        
        # Multilingual content patterns
        self.multilingual_patterns = {
            'xml_tags': re.compile(r'<[^>]+>([^<]*)</[^>]+>', re.IGNORECASE),
            'bracket_tags': re.compile(r'\[([^\]]+)\]'),
            'paren_tags': re.compile(r'\(([^)]+)\)'),
            'tamil_unicode': re.compile(r'[\u0B80-\u0BFF]+'),
            'chinese_unicode': re.compile(r'[\u4e00-\u9fff]+'),
        }
        
        # Digit to word mapping
        self.digit_to_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
    
    def expand_contractions(self, text: str) -> str:
        if not self.asr_config.expand_contractions or text is None:
            return text if text is not None else ""
        
        if not isinstance(text, str):
            text = str(text)
        
        if self.contraction_expander is None:
            return text
        
        try:
            use_advanced = self.asr_config.use_advanced_contraction_expansion
            result = self.contraction_expander.expand_contractions(text, use_advanced=use_advanced)
            return result
        except Exception as e:
            logger.warning(f"Contraction expansion failed: {e}")
            return text
    
    def normalize_whitespace(self, text: str) -> str:
        if text is None:
            return ""
        
        if not isinstance(text, str):
            text = str(text)
        
        try:
            text = re.sub(r'\n+', ' ', text)
            text = re.sub(r'\r+', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            return text
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
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            return text
        except Exception as e:
            logger.warning(f"Enhanced filler word removal failed: {e}")
            return text
    
    def standardize_number_and(self, text: str) -> str:
        """Standardize "and" in number readings for consistency."""
        if not self.asr_config.standardize_and_in_numbers or text is None:
            return text if text is not None else ""
        
        if not isinstance(text, str):
            text = str(text)
        
        try:
            for pattern_name, pattern in self.number_and_patterns.items():
                if pattern_name == 'hundred_and':
                    text = pattern.sub(r'\1 hundred ', text)
                elif pattern_name == 'thousand_and':
                    text = pattern.sub(r'\1 thousand ', text)
                elif pattern_name == 'million_and':
                    text = pattern.sub(r'\1 million ', text)
            
            text = re.sub(r'\s+', ' ', text)
            return text
        except Exception as e:
            logger.warning(f"Number and standardization failed: {e}")
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
            
            if re.search(self.multilingual_patterns['tamil_unicode'], text):
                text = self.multilingual_patterns['tamil_unicode'].sub('hello', text)
            
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
        
        def safe_num_to_words(num_str: str) -> str:
            try:
                clean_num = num_str.replace(',', '')
                num = float(clean_num)
                if num == int(num):
                    return num2words(int(num))
                else:
                    dollars = int(num)
                    cents = int(round((num - dollars) * 100))
                    if cents == 0:
                        return num2words(dollars)
                    else:
                        return f"{num2words(dollars)} point {num2words(cents)}"
            except (ValueError, OverflowError):
                return num_str
        
        try:
            def sgd_replace(match):
                words = safe_num_to_words(match.group(1))
                return f"{words} singapore dollars"
            text = self.currency_patterns['sgd_symbol'].sub(sgd_replace, text)
            
            def usd_replace(match):
                words = safe_num_to_words(match.group(1))
                return f"{words} dollars"
            text = self.currency_patterns['usd_symbol'].sub(usd_replace, text)
            
            return text
        except Exception as e:
            logger.warning(f"Enhanced currency conversion failed: {e}")
            return text
    
    def convert_phone_to_words(self, text: str) -> str:
        if not self.asr_config.convert_phone_to_words or text is None:
            return text if text is not None else ""
        
        if not isinstance(text, str):
            text = str(text)
        
        def digits_to_words(digits: str) -> str:
            return ' '.join(self.digit_to_words.get(d, d) for d in digits)
        
        try:
            singapore_patterns = [
                'sg_with_country', 'sg_landline_dash', 'sg_mobile_dash', 
                'sg_spaced', 'sg_continuous'
            ]
            
            for pattern_name in singapore_patterns:
                if pattern_name in self.phone_patterns:
                    pattern = self.phone_patterns[pattern_name]
                    
                    def sg_phone_replace(match):
                        if pattern_name == 'sg_with_country':
                            digits = '65' + match.group(1) + match.group(2)
                        elif pattern_name == 'sg_continuous':
                            digits = match.group(1)
                            if len(digits) == 8 and digits[0] in '6789':
                                return digits_to_words(digits)
                            else:
                                return match.group(0)
                        else:
                            digits = match.group(1) + match.group(2)
                        return digits_to_words(digits)
                    
                    text = pattern.sub(sg_phone_replace, text)
            
            us_patterns = ['formatted', 'spaced', 'dots', 'parentheses']
            for pattern_name in us_patterns:
                if pattern_name in self.phone_patterns:
                    pattern = self.phone_patterns[pattern_name]
                    
                    def phone_replace(match):
                        digits = match.group(1) + match.group(2) + match.group(3)
                        return digits_to_words(digits)
                    
                    text = pattern.sub(phone_replace, text)
            
            def long_number_replace(match):
                digits = match.group(1)
                if len(digits) in [8, 10, 11]:
                    return digits_to_words(digits)
                return match.group(0)
            
            text = self.phone_patterns['long_number'].sub(long_number_replace, text)
            
            return text
        except Exception as e:
            logger.warning(f"Enhanced phone to words conversion failed: {e}")
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
                elif m < 10:
                    minute_words = f"oh {num2words(m)}"
                else:
                    minute_words = num2words(m)
                
                return f"{hour_words} {minute_words}"
            except (ValueError, TypeError):
                return f"{hour} {minute}"
        
        try:
            # Convert colon format times
            def colon_replace(match):
                hour, minute = match.groups()
                return time_to_words(hour, minute)
            
            text = self.time_patterns['colon_time'].sub(colon_replace, text)
            
            # Convert simple time format
            def simple_replace(match):
                time_str = match.group(1)
                if len(time_str) == 3:
                    hour = time_str[0]
                    minute = time_str[1:]
                elif len(time_str) == 4:
                    hour = time_str[:2]
                    minute = time_str[2:]
                else:
                    return match.group(0)
                
                return time_to_words(hour, minute)
            
            text = self.time_patterns['simple_time'].sub(simple_replace, text)
            
            # Look for time patterns followed by AM/PM
            time_am_pm_pattern = re.compile(
                r'(\b(?:\d{1,2}:\d{2}|\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirty|fifteen|forty|fifty|o\'?clock)\s*)(am|pm)\b',
                re.IGNORECASE
            )
            
            def time_am_pm_replace(match):
                time_part = match.group(1)
                am_pm = match.group(2).lower()
                if am_pm == 'am':
                    return f"{time_part}a m"
                else:
                    return f"{time_part}p m"
            
            text = time_am_pm_pattern.sub(time_am_pm_replace, text)
            
            return text
        except Exception as e:
            logger.warning(f"Time to words conversion failed: {e}")
            return text
    
    def convert_numbers_to_words(self, text: str) -> str:
        if not self.asr_config.convert_numbers_to_words or text is None:
            return text if text is not None else ""
        
        if not isinstance(text, str):
            text = str(text)
        
        if not NUM2WORDS_AVAILABLE:
            return text
        
        try:
            def number_replace(match):
                try:
                    num = int(match.group(1))
                    if 1 <= num <= 999999:
                        return num2words(num)
                    return match.group(0)
                except (ValueError, TypeError):
                    return match.group(0)
            
            text = re.sub(r'\b(\d{1,6})\b', number_replace, text)
            return text
        except Exception as e:
            logger.warning(f"Numbers to words conversion failed: {e}")
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
            compound_patterns = [
                (r'\btwentyone\b', 'twenty one'), (r'\btwentytwo\b', 'twenty two'),
                (r'\btwentythree\b', 'twenty three'), (r'\btwentyfour\b', 'twenty four'),
                (r'\btwentyfive\b', 'twenty five'), (r'\btwentysix\b', 'twenty six'),
                (r'\btwentyseven\b', 'twenty seven'), (r'\btwentyeight\b', 'twenty eight'),
                (r'\btwentynine\b', 'twenty nine'), (r'\bthirtyone\b', 'thirty one'),
                (r'\bthirtytwo\b', 'thirty two'), (r'\bthirtythree\b', 'thirty three'),
                (r'\bthirtyfour\b', 'thirty four'), (r'\bthirtyfive\b', 'thirty five'),
                (r'\bthirtysix\b', 'thirty six'), (r'\bthirtyseven\b', 'thirty seven'),
                (r'\bthirtyeight\b', 'thirty eight'), (r'\bthirtynine\b', 'thirty nine'),
                (r'\bfortyone\b', 'forty one'), (r'\bfortytwo\b', 'forty two'),
                (r'\bfortythree\b', 'forty three'), (r'\bfortyfour\b', 'forty four'),
                (r'\bfortyfive\b', 'forty five'), (r'\bfortysix\b', 'forty six'),
                (r'\bfortyseven\b', 'forty seven'), (r'\bfortyeight\b', 'forty eight'),
                (r'\bfortynine\b', 'forty nine'), (r'\bfiftyone\b', 'fifty one'),
                (r'\bfiftytwo\b', 'fifty two'), (r'\bfiftythree\b', 'fifty three'),
                (r'\bfiftyfour\b', 'fifty four'), (r'\bfiftyfive\b', 'fifty five'),
                (r'\bfiftysix\b', 'fifty six'), (r'\bfiftyseven\b', 'fifty seven'),
                (r'\bfiftyeight\b', 'fifty eight'), (r'\bfiftynine\b', 'fifty nine'),
                (r'\bsixtyone\b', 'sixty one'), (r'\bsixtytwo\b', 'sixty two'),
                (r'\bsixtythree\b', 'sixty three'), (r'\bsixtyfour\b', 'sixty four'),
                (r'\bsixtyfive\b', 'sixty five'), (r'\bsixtysix\b', 'sixty six'),
                (r'\bsixtyseven\b', 'sixty seven'), (r'\bsixtyeight\b', 'sixty eight'),
                (r'\bsixtynine\b', 'sixty nine'), (r'\bseventyone\b', 'seventy one'),
                (r'\bseventytwo\b', 'seventy two'), (r'\bseventythree\b', 'seventy three'),
                (r'\bseventyfour\b', 'seventy four'), (r'\bseventyfive\b', 'seventy five'),
                (r'\bseventysix\b', 'seventy six'), (r'\bseventyseven\b', 'seventy seven'),
                (r'\bseventyeight\b', 'seventy eight'), (r'\bseventynine\b', 'seventy nine'),
                (r'\beightyone\b', 'eighty one'), (r'\beightytwo\b', 'eighty two'),
                (r'\beightythree\b', 'eighty three'), (r'\beightyfour\b', 'eighty four'),
                (r'\beightyfive\b', 'eighty five'), (r'\beightysix\b', 'eighty six'),
                (r'\beightyseven\b', 'eighty seven'), (r'\beightyeight\b', 'eighty eight'),
                (r'\beightynine\b', 'eighty nine'), (r'\bninetyone\b', 'ninety one'),
                (r'\bninetytwo\b', 'ninety two'), (r'\bninetythree\b', 'ninety three'),
                (r'\bninetyfour\b', 'ninety four'), (r'\bninetyfive\b', 'ninety five'),
                (r'\bninetysix\b', 'ninety six'), (r'\bninetyseven\b', 'ninety seven'),
                (r'\bninetyeight\b', 'ninety eight'), (r'\bninetynine\b', 'ninety nine'),
            ]
            
            for pattern, replacement in compound_patterns:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
            return text
            
        except Exception as e:
            logger.warning(f"Compound number spacing fix failed: {e}")
            return text
    
    def merge_cjk_characters_enhanced(self, text: str) -> str:
        """Enhanced CJK character merging using targeted regex."""
        if not self.asr_config.merge_cjk_characters or text is None:
            return text if text is not None else ""
        
        if not isinstance(text, str):
            text = str(text)
        
        try:
            cjk_pattern = re.compile(r'([\u4e00-\u9fff]) (?=[\u4e00-\u9fff])')
            merged = cjk_pattern.sub(r'\1', text)
            return merged
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
                if self.asr_config.preserve_apostrophes:
                    result = []
                    for ch in text:
                        if unicodedata.category(ch).startswith('P') and ch != "'":
                            result.append(' ')
                        else:
                            result.append(ch)
                    text = ''.join(result)
                else:
                    result = []
                    for ch in text:
                        if unicodedata.category(ch).startswith('P'):
                            result.append(' ')
                        else:
                            result.append(ch)
                    text = ''.join(result)
                    
            elif method == "preserve_apostrophe":
                punctuation_chars = string.punctuation.replace("'", "")
                translator = str.maketrans(punctuation_chars, ' ' * len(punctuation_chars))
                text = text.translate(translator)
            
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            return text
            
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


class RefinedNeMo_Singlish_ASR_Normaliser(RefinedBaseNormalizer):
    
    def __init__(self, 
                 language: str="en",
                 cache_dir: Optional[str] = None,
                 use_lm_context: bool = False,
                 enable_cjk: bool = True,
                 enable_disfluency_removal: bool = True,
                 asr_config: Optional[RefinedASRConfig] = None):
        
        # Initialize base normalizer first
        super().__init__(asr_config)
        
        # Handle None or empty language
        if not language or language is None:
            logger.warning("Language is None or empty, defaulting to 'en'")
            language = 'en'
        
        # Ensure cache_dir is a valid path or None
        if cache_dir is not None and not isinstance(cache_dir, str):
            logger.warning(f"Invalid cache_dir type: {type(cache_dir)}, setting to None")
            cache_dir = None
        
        # Validate boolean parameters
        self.enable_cjk = bool(enable_cjk) if enable_cjk is not None else True
        self.enable_disfluency_removal = bool(enable_disfluency_removal) if enable_disfluency_removal is not None else True
        
        # Import the English normalizer with NeMo disabled
        try:
            from .nemo_english import RefinedNeMo_English_ASR_Normaliser, RefinedASRConfig
            
            # Create config with NeMo completely disabled
            english_config = RefinedASRConfig(
                nemo_use_tn=False,      # Disable Text Normalization
                nemo_use_itn=False,     # Disable Inverse Text Normalization
                # Copy other settings from asr_config if provided
                convert_currency_to_words=getattr(asr_config, 'convert_currency_to_words', True) if asr_config else True,
                convert_phone_to_words=getattr(asr_config, 'convert_phone_to_words', True) if asr_config else True,
                convert_time_to_words=getattr(asr_config, 'convert_time_to_words', True) if asr_config else True,
                convert_numbers_to_words=getattr(asr_config, 'convert_numbers_to_words', True) if asr_config else True,
                expand_abbreviations=getattr(asr_config, 'expand_abbreviations', True) if asr_config else True,
                standardize_and_in_numbers=getattr(asr_config, 'standardize_and_in_numbers', True) if asr_config else True,
                remove_filler_words=getattr(asr_config, 'remove_filler_words', True) if asr_config else True,
                normalize_singapore_currency=getattr(asr_config, 'normalize_singapore_currency', True) if asr_config else True,
                clean_multilingual_content=getattr(asr_config, 'clean_multilingual_content', True) if asr_config else True,
                remove_punctuation=getattr(asr_config, 'remove_punctuation', True) if asr_config else True,
                lowercase_output=getattr(asr_config, 'lowercase_output', True) if asr_config else True,
                expand_contractions=getattr(asr_config, 'expand_contractions', True) if asr_config else True,
                use_advanced_contraction_expansion=getattr(asr_config, 'use_advanced_contraction_expansion', True) if asr_config else True,
                merge_cjk_characters=getattr(asr_config, 'merge_cjk_characters', True) if asr_config else True,
                punctuation_removal_method=getattr(asr_config, 'punctuation_removal_method', 'unicode') if asr_config else 'unicode',
                preserve_apostrophes=getattr(asr_config, 'preserve_apostrophes', False) if asr_config else False,
            )
            
            self.english_normalizer = RefinedNeMo_English_ASR_Normaliser(
                language=language, 
                cache_dir=cache_dir, 
                use_lm_context=use_lm_context, 
                enable_disfluency_removal=False,
                asr_config=english_config
            )
        except ImportError as e:
            logger.warning(f"Failed to import RefinedNeMo_English_ASR_Normaliser: {e}")
            # Create a fallback normalizer
            self.english_normalizer = None
        
        self._compile_singlish_patterns()
        
        logger.debug("Refined ASR-optimized Singlish normalizer initialized")
    
    def _compile_singlish_patterns(self):
        try:
            self.cjk_pattern = re.compile(r'([\u4e00-\u9fff]) (?=[\u4e00-\u9fff])')
            
            self.singlish_disfluencies = [
                r"\buh\b", r"\bum\b", r"\bumm\b", r"\berr+\b", r"\berm+\b",
                r"\bah\b", r"\beh\b", r"\bmm\b", r"\bmmhmm\b",
                r"\blah\b(?!\w)", r"\blor\b(?!\w)", r"\bmeh\b(?!\w)", 
                r"\bwah\b(?!\w)", r"\baiya\b", r"\bhaiya\b", r"\bwey\b"
            ]
            self.disfluency_pattern = re.compile(
                rf"(?:{'|'.join(self.singlish_disfluencies)})", 
                flags=re.IGNORECASE
            )
        except Exception as e:
            logger.error(f"Failed to compile Singlish patterns: {e}")
            # Create fallback patterns
            self.cjk_pattern = re.compile(r'([\u4e00-\u9fff]) (?=[\u4e00-\u9fff])')
            self.singlish_disfluencies = [r"\buh\b", r"\bum\b"]
            self.disfluency_pattern = re.compile(rf"(?:{'|'.join(self.singlish_disfluencies)})", flags=re.IGNORECASE)
    
    def merge_cjk_characters(self, text: str) -> str:
        if not self.enable_cjk or text is None:
            return text if text is not None else ""
        
        if not isinstance(text, str):
            return str(text)
        
        try:
            result = self.cjk_pattern.sub(r'\1', text)
            return result
        except Exception as e:
            logger.warning(f"CJK merging failed: {e}")
            return text
    
    def remove_singlish_disfluencies(self, text: str) -> str:
        if not self.enable_disfluency_removal or text is None:
            return text if text is not None else ""
        
        if not isinstance(text, str):
            return str(text)
        
        try:
            cleaned = self.disfluency_pattern.sub("", text)
            result = re.sub(r"\s+", " ", cleaned).strip()
            return result
        except Exception as e:
            logger.warning(f"Singlish disfluency removal failed: {e}")
            return text
    
    def fix_singapore_currency_preprocessing(self, text: str) -> str:
        if not isinstance(text, str) or text is None:
            return str(text) if text is not None else ""
        
        try:
            sgd_pattern = re.compile(r'S\$([0-9,]+(?:\.[0-9]{1,2})?)', re.IGNORECASE)
            
            def sgd_replace(match):
                amount = match.group(1)
                clean_amount = amount.replace(',', '')
                if NUM2WORDS_AVAILABLE:
                    try:
                        num = float(clean_amount)
                        if num == int(num):
                            words = num2words(int(num))
                        else:
                            words = num2words(int(num))
                        return f"{words} singapore dollars"
                    except:
                        pass
                return f"{clean_amount} singapore dollars"
            
            text = sgd_pattern.sub(sgd_replace, text)
            return text
            
        except Exception as e:
            logger.warning(f"Singapore currency preprocessing failed: {e}")
            return text

    def preprocess_singlish(self, text: str) -> str:
        if text is None:
            return ""
        
        if not isinstance(text, str):
            text = str(text)
        
        try:
            # Apply preprocessing steps in order
            text = self.normalize_whitespace(text)
            text = self.expand_contractions(text)
            text = self.convert_phone_to_words(text)
            text = self.fix_singapore_currency_preprocessing(text)
            text = self.merge_cjk_characters(text)
            text = self.remove_singlish_disfluencies(text)
            text = self.normalize_whitespace(text)
            
            return text
        except Exception as e:
            logger.error(f"Singlish preprocessing failed: {e}")
            # Return basic normalized text as fallback
            try:
                return self.normalize_whitespace(str(text))
            except:
                return str(text)
    
    def normalize_for_asr_eval(self, text: str,
                              apply_asr_standardization: bool = True) -> str:
        if text is None:
            return ""
        
        if not isinstance(text, str):
            text = str(text)
        
        try:
            preprocessed = self.preprocess_singlish(text)
            
            if self.english_normalizer is not None:
                normalized = self.english_normalizer.normalize_for_asr_eval(
                    preprocessed,
                    use_preprocessing=False,
                    apply_asr_standardization=apply_asr_standardization
                )
            else:
                # Fallback to basic normalization if English normalizer is not available
                logger.debug("English normalizer not available, using basic normalization")
                normalized = self.apply_refined_asr_standardization(preprocessed)
            
            if apply_asr_standardization:
                normalized = self.fix_compound_number_spacing(normalized)
                normalized = self.normalize_whitespace(normalized)
                if self.asr_config.lowercase_output:
                    normalized = normalized.lower()
                normalized = normalized.strip()
            
            return normalized
            
        except Exception as e:
            logger.error(f"Refined Singlish ASR normalization failed: {e}")
            return text.lower().strip() if isinstance(text, str) else ""

    def test_normalization(self, test_text: str = "Hello lah, how are you? I'm fine lor!") -> str:
        """Test method to verify the normalizer works correctly"""
        try:
            result = self.normalize_for_asr_eval(test_text)
            logger.info(f"Test input: '{test_text}'")
            logger.info(f"Test output: '{result}'")
            return result
        except Exception as e:
            logger.error(f"Test normalization failed: {e}")
            return test_text