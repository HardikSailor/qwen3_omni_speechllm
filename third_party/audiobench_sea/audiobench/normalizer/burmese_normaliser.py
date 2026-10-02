  #!/usr/bin/env python3
"""
Burmese (Myanmar) ASR Normalizer for Fair CER Calculation
Handles Zawgyi/Unicode conversion, number normalization, currency, phones, dates, etc.
"""

import re
import logging
import string
import unicodedata
from typing import List, Dict, Optional, Tuple, Union, Set
from dataclasses import dataclass
import traceback

# Myanmar-specific libraries
try:
    import myanmartools
    from myanmartools import ZawgyiDetector
    MYANMAR_TOOLS_AVAILABLE = True
except ImportError:
    MYANMAR_TOOLS_AVAILABLE = False
    print("Warning: myanmar-tools not available - using fallback Zawgyi detection")

try:
    import icu
    PYICU_AVAILABLE = True
except ImportError:
    PYICU_AVAILABLE = False
    print("Warning: PyICU not available - install with: pip install PyICU")

# Optional libraries
try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("Warning: jiwer not available - install with: pip install jiwer")

logger = logging.getLogger(__name__)

class FallbackZawgyiDetector:
    """
    Fallback Zawgyi detector when myanmar-tools is not available.
    Uses heuristics based on Unicode ranges to detect Zawgyi vs Unicode.
    """
    
    def __init__(self):
        # Unicode ranges for Myanmar script
        self.unicode_myanmar_ranges = [
            (0x1000, 0x109F),  # Myanmar
            (0xAA60, 0xAA7F),  # Myanmar Extended-A
            (0xA9E0, 0xA9FF),  # Myanmar Extended-B
        ]
        
        # Common Zawgyi-specific character patterns
        self.zawgyi_patterns = [
            # These are heuristic patterns - may need adjustment
            re.compile(r'[\u1031\u1032\u1033\u1034\u1035\u1036\u1037\u1038\u1039\u103A\u103B\u103C\u103D\u103E\u103F]'),
        ]
    
    def get_zawgyi_probability(self, text: str) -> float:
        """
        Estimate probability that text is in Zawgyi encoding.
        Returns 0.0 (likely Unicode) to 1.0 (likely Zawgyi).
        """
        if not text:
            return 0.0
        
        # Count characters in different ranges
        unicode_count = 0
        zawgyi_indicators = 0
        total_myanmar_chars = 0
        
        for char in text:
            char_code = ord(char)
            
            # Check if it's a Myanmar character
            is_myanmar = any(start <= char_code <= end for start, end in self.unicode_myanmar_ranges)
            if is_myanmar:
                total_myanmar_chars += 1
                
                # Check for Unicode Myanmar range
                if 0x1000 <= char_code <= 0x109F:
                    unicode_count += 1
                
                # Check for Zawgyi indicators (heuristic)
                for pattern in self.zawgyi_patterns:
                    if pattern.search(char):
                        zawgyi_indicators += 1
                        break
        
        if total_myanmar_chars == 0:
            return 0.0
        
        # Simple heuristic: if we see many Zawgyi indicators, it's likely Zawgyi
        zawgyi_ratio = zawgyi_indicators / total_myanmar_chars if total_myanmar_chars > 0 else 0.0
        
        # If zawgyi_ratio is high, it's likely Zawgyi
        # If unicode_count is high and zawgyi_ratio is low, it's likely Unicode
        if zawgyi_ratio > 0.3:
            return min(0.8, zawgyi_ratio * 2)  # Cap at 0.8 for fallback
        elif unicode_count > total_myanmar_chars * 0.7:
            return 0.1  # Likely Unicode
        else:
            return 0.5  # Uncertain


@dataclass
class BurmeseASRConfig:
    """Configuration for Burmese ASR normalization"""
    
    # Core text processing
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    lowercase_output: bool = True
    remove_punctuation: bool = True
    
    # Encoding
    convert_zawgyi: bool = True
    zawgyi_threshold: float = 0.95
    
    # Burmese-specific features
    convert_burmese_digits: bool = True  # ၀-၉ → 0-9
    convert_numbers_to_words: bool = False  # Keep as digits for ASR
    normalize_currency: bool = True
    normalize_phone_numbers: bool = True
    normalize_time_format: bool = True
    normalize_date_format: bool = True
    
    # Text cleanup
    remove_filler_words: bool = True
    expand_abbreviations: bool = True
    merge_repeated_chars: bool = True
    
    # Advanced options
    use_word_segmentation: bool = False  # Burmese lacks word boundaries
    enable_validation: bool = True
    enable_debug_logging: bool = False
    strict_mode: bool = False
    max_text_length: int = 500000


class ZawgyiUnicodeConverter:
    """Handle Zawgyi to Unicode conversion"""
    
    def __init__(self, config: BurmeseASRConfig):
        self.config = config
        self._initialize_converter()
    
    def _initialize_converter(self):
        """Initialize Zawgyi detector and converter"""
        self.detector = None
        self.converter = None
        
        if MYANMAR_TOOLS_AVAILABLE:
            try:
                self.detector = ZawgyiDetector()
                logger.info("Myanmar-tools Zawgyi detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Zawgyi detector: {e}")
                self.detector = FallbackZawgyiDetector()
                logger.info("Using fallback Zawgyi detector")
        else:
            self.detector = FallbackZawgyiDetector()
            logger.info("Using fallback Zawgyi detector (myanmar-tools not available)")
        
        if PYICU_AVAILABLE:
            try:
                # Create ICU transliterator for Zawgyi to Myanmar3 (Unicode)
                self.converter = icu.Transliterator.createInstance('Zawgyi-my')
                logger.info("PyICU Zawgyi converter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ICU converter: {e}")
                # Try alternative ID
                try:
                    self.converter = icu.Transliterator.createInstance('Zawgyi-Myanmar')
                    logger.info("PyICU Zawgyi converter initialized with alternative ID")
                except:
                    pass
    
    def convert_to_unicode(self, text: str) -> str:
        """Convert text from Zawgyi to Unicode if needed"""
        if not self.config.convert_zawgyi or not text:
            return text or ""
        
        # Check if conversion is needed
        if self.detector and self.converter:
            try:
                zawgyi_probability = self.detector.get_zawgyi_probability(text)
                
                if self.config.enable_debug_logging:
                    logger.debug(f"Zawgyi probability: {zawgyi_probability}")
                
                # Convert if probability exceeds threshold
                if zawgyi_probability >= self.config.zawgyi_threshold:
                    converted = self.converter.transliterate(text)
                    if self.config.enable_debug_logging:
                        logger.debug(f"Converted from Zawgyi: '{text}' → '{converted}'")
                    return converted
                
            except Exception as e:
                logger.warning(f"Zawgyi conversion failed: {e}")
        
        return text
    
    def is_zawgyi(self, text: str) -> bool:
        """Check if text is likely Zawgyi encoded"""
        if self.detector:
            try:
                return self.detector.get_zawgyi_probability(text) >= self.config.zawgyi_threshold
            except:
                pass
        return False


class BurmeseNumberConverter:
    """Convert Burmese numbers between text and digits"""
    
    def __init__(self, config: BurmeseASRConfig):
        self.config = config
        self._setup_number_mappings()
        self._compile_patterns()
    
    def _setup_number_mappings(self):
        """Setup Burmese number mappings"""
        # Burmese digits
        self.burmese_digits = {
            '၀': '0', '၁': '1', '၂': '2', '၃': '3', '၄': '4',
            '၅': '5', '၆': '6', '၇': '7', '၈': '8', '၉': '9'
        }
        
        # Reverse mapping
        self.arabic_to_burmese = {v: k for k, v in self.burmese_digits.items()}
        
        # Basic number words
        self.number_words = {
            'သုည': 0,
            'တစ်': 1, 'နှစ်': 2, 'သုံး': 3, 'လေး': 4, 'ငါး': 5,
            'ခြောက်': 6, 'ခုနစ်': 7, 'ရှစ်': 8, 'ကိုး': 9, 'ဆယ်': 10,
            
            # Special forms
            'တစ်ဆယ်': 10,
            
            # Teens
            'ဆယ့်တစ်': 11, 'ဆယ့်နှစ်': 12, 'ဆယ့်သုံး': 13, 'ဆယ့်လေး': 14, 'ဆယ့်ငါး': 15,
            'ဆယ့်ခြောက်': 16, 'ဆယ့်ခုနစ်': 17, 'ဆယ့်ရှစ်': 18, 'ဆယ့်ကိုး': 19,
            
            # Tens
            'နှစ်ဆယ်': 20, 'သုံးဆယ်': 30, 'လေးဆယ်': 40, 'ငါးဆယ်': 50,
            'ခြောက်ဆယ်': 60, 'ခုနစ်ဆယ်': 70, 'ရှစ်ဆယ်': 80, 'ကိုးဆယ်': 90,
            
            # Scale words
            'ရာ': 100, 'ထောင်': 1000, 'သောင်း': 10000, 'သိန်း': 100000,
            'သန်း': 1000000, 'ကုဋေ': 10000000,
            
            # Fractions
            'ခွဲ': 0.5,  # Half
            'ကျပ်သား': 1.6,  # Traditional weight unit
        }
        
        # Compound number patterns (e.g., နှစ်ဆယ့်ငါး = 25)
        self.compound_patterns = []
        for tens_val in range(20, 100, 10):
            tens_word = self._get_tens_word(tens_val)
            if tens_word:
                for ones_val in range(1, 10):
                    ones_word = self._get_ones_word(ones_val)
                    if ones_word:
                        compound = f"{tens_word}့{ones_word}"
                        self.number_words[compound] = tens_val + ones_val
    
    def _get_tens_word(self, value: int) -> Optional[str]:
        """Get tens word for a value"""
        tens_map = {
            20: 'နှစ်ဆယ်', 30: 'သုံးဆယ်', 40: 'လေးဆယ်', 50: 'ငါးဆယ်',
            60: 'ခြောက်ဆယ်', 70: 'ခုနစ်ဆယ်', 80: 'ရှစ်ဆယ်', 90: 'ကိုးဆယ်'
        }
        return tens_map.get(value)
    
    def _get_ones_word(self, value: int) -> Optional[str]:
        """Get ones word for a value"""
        ones_map = {
            1: 'တစ်', 2: 'နှစ်', 3: 'သုံး', 4: 'လေး', 5: 'ငါး',
            6: 'ခြောက်', 7: 'ခုနစ်', 8: 'ရှစ်', 9: 'ကိုး'
        }
        return ones_map.get(value)
    
    def _compile_patterns(self):
        """Compile number detection patterns"""
        # Pattern for Burmese digits
        self.burmese_digit_pattern = re.compile(r'[၀-၉]+')
        
        # Pattern for numbers with thousand separators
        self.thousand_sep_pattern = re.compile(r'[၀-၉]{1,3}(?:,[၀-၉]{3})+')
        
        # Pattern for decimal numbers
        self.decimal_pattern = re.compile(r'[၀-၉]+\.[၀-၉]+')
    
    def convert_burmese_digits_to_arabic(self, text: str) -> str:
        """Convert Burmese digits (၀-၉) to Arabic digits (0-9) - FIXED DECIMAL HANDLING"""
        if not self.config.convert_burmese_digits or not text:
            return text or ""
        
        result = text
        
        # First, protect decimal points between Burmese digits
        # Convert Burmese digits around decimal points but KEEP the decimal point as "."
        decimal_pattern = re.compile(r'([၀-၉]+)\.([၀-၉]+)')
        
        def protect_decimal(match):
            # Convert the digits but keep the decimal point as "."
            before = match.group(1)
            after = match.group(2)
            before_arabic = ''.join(self.burmese_digits.get(d, d) for d in before)
            after_arabic = ''.join(self.burmese_digits.get(d, d) for d in after)
            return f"{before_arabic}.{after_arabic}"
        
        result = decimal_pattern.sub(protect_decimal, result)
        
        # Handle thousand separators
        thousand_pattern = re.compile(r'[၀-၉]{1,3}(?:,[၀-၉]{3})+')
        
        for match in thousand_pattern.finditer(text):
            burmese_num = match.group(0)
            # Remove commas and convert
            clean_num = burmese_num.replace(',', '')
            arabic_num = ''.join(self.burmese_digits.get(d, d) for d in clean_num)
            result = result.replace(burmese_num, arabic_num)
        
        # CRITICAL: Convert all remaining Burmese digits to Arabic
        # This ensures that ALL Burmese digits are converted, regardless of length
        for burmese, arabic in self.burmese_digits.items():
            result = result.replace(burmese, arabic)
        
        # CRITICAL: DO NOT introduce "decimal" word - keep the decimal point as "."
        # Remove any accidental "decimal" words that might have been introduced
        result = re.sub(r'(\d+)\s*decimal\s*(\d+)', r'\1.\2', result)
        
        return result
    
    def convert_number_words_to_digits(self, text: str) -> str:
        """Convert Burmese number words to digits - FIXED VERSION WITH CURRENCY PRESERVATION"""
        if not text:
            return ""
        
        result = text
        
        # Process complex numbers with proper order and handling of ့ connector
        # AND preserve currency words
        result = self._convert_complex_numbers(result)
        
        return result
    
    def _convert_complex_numbers(self, text: str) -> str:
        """Convert complex number expressions - FIXED VERSION WITH CURRENCY AND PERCENTAGE PRESERVATION"""
        result = text

        # QUICK FIX #1: strip leading 'တစ်ဆယ့်' so that 'တစ်ဆယ့်ငါး' becomes 'ဆယ့်ငါး'
        result = result.replace('တစ်ဆယ့်', 'ဆယ့်')

        # QUICK FIX #2: after everything, collapse any leftover isolated number-words
        # We'll do this right before we return, but it's simplest to sketch it here:
        #   replace any standalone 'တစ်' → '1', 'နှစ်' → '2', etc.
        remap = {
            'တစ်': '1',  'နှစ်': '2',  'သုံး': '3',  'လေး': '4',
            'ငါး': '5',  'ခြောက်': '6', 'ခုနစ်': '7', 'ရှစ်': '8',
            'ကိုး': '9'
        }
        for w, d in remap.items():
            # match standalone w (start-of-string or whitespace before, whitespace or end-of-string after)
            pattern = rf'(?:(?<=^)|(?<=\s)){w}(?=(?:\s|$))'
            result = re.sub(pattern, d, result)
        
        # CRITICAL: Define currency words to preserve
        CURRENCY_WORDS = {
            'ကျပ်', 'ဒေါ်လာ', 'ယူရို', 'ယွမ်', 'ဘတ်', 
            'ပေါင်', 'ရူပီး', 'ဝမ်', 'ရင်းဂစ်', 'ဒိုင်နာ'
        }
        
        # CRITICAL: Also protect percentage words
        PERCENTAGE_WORDS = {'ရာခိုင်နှုန်း', 'ရာနှုန်း'}
        
        # Protect percentage words FIRST
        protected_percentage = []
        for percentage in PERCENTAGE_WORDS:
            if percentage in result:
                placeholder = f'__PERCENTAGE_{len(protected_percentage)}__'
                protected_percentage.append((placeholder, percentage))
                result = result.replace(percentage, placeholder)
        # ------------------------------------------------------------------
        # 0️⃣  Normalize spelling variants of scale words
        # ------------------------------------------------------------------
        SCALE_ALIASES = {
            'ထောင့်': 'ထောင်', 'ထောင့်': 'ထောင်',
            'သောင်း့': 'သောင်း', 'သိန်း့': 'သိန်း',
            'သန်း့': 'သန်း',   'ကုဋေ့': 'ကုဋေ',
        }
        for wrong, right in SCALE_ALIASES.items():
            result = result.replace(wrong, right)
        result = re.sub(r'(ထောင်|သောင်း|သိန်း|သန်း|ကုဋေ)့', r'\1', result)
        
        # Define proper word boundary for Burmese
        burmese_char = r'[\u1000-\u109F]'
        not_burmese = r'(?!' + burmese_char + ')'
        not_preceded_by_burmese = r'(?<!' + burmese_char + ')'

        # ------------------------------------------------------------------
        # SPECIAL HANDLING: Process "ခွဲ" (half) with scale words FIRST
        # ------------------------------------------------------------------
        # Pattern for scale word + ခွဲ (e.g., သန်းခွဲ = 1.5 million)
        scale_half_pattern = re.compile(
            not_preceded_by_burmese + 
            r'(တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး)?\s*(သန်း|ကုဋေ|သိန်း|သောင်း|ထောင်|ရာ)\s*ခွဲ' +
            not_burmese
        )
        
        def replace_scale_half(match):
            number_word = match.group(1)
            scale_word = match.group(2)
            
            # Get the multiplier
            number_map = {
                'တစ်': 1, 'နှစ်': 2, 'သုံး': 3, 'လေး': 4, 'ငါး': 5,
                'ခြောက်': 6, 'ခုနစ်': 7, 'ရှစ်': 8, 'ကိုး': 9
            }
            multiplier = number_map.get(number_word, 1)  # Default to 1 if no number
            
            # Get scale value
            scale_map = {
                'ရာ': 100, 'ထောင်': 1000, 'သောင်း': 10000,
                'သိန်း': 100000, 'သန်း': 1000000, 'ကုဋေ': 10000000
            }
            scale_value = scale_map.get(scale_word, 1)
            
            # Calculate: multiplier * scale + (scale / 2)
            # e.g., သုံးသန်းခွဲ = 3 * 1000000 + 500000 = 3500000
            total = multiplier * scale_value + (scale_value // 2)
            return str(total)
        
        result = scale_half_pattern.sub(replace_scale_half, result)

        # ------------------------------------------------------------------
        # CRITICAL: Separate currency words before processing numbers
        # ------------------------------------------------------------------
        # Find and protect currency words
        protected_currency = None
        currency_position = -1
        
        for currency in CURRENCY_WORDS:
            if currency in result:
                currency_position = result.find(currency)
                protected_currency = currency
                # Replace currency with a placeholder
                result = result.replace(currency, '__CURRENCY__')
                break
        
        # Step 1: Convert "ဆယ်" multiplier before scale words (like ဆယ်သန်း = 10 million)
        multiplier_scale_pattern = re.compile(
            not_preceded_by_burmese + 
            r'(ဆယ်|တစ်ဆယ်)\s*(သန်း|ကုဋေ|သိန်း|သောင်း)' +
            not_burmese
        )
        
        def replace_multiplier_scale(match):
            multiplier = match.group(1)
            scale = match.group(2)
            
            mult_value = 10  # Both ဆယ် and တစ်ဆယ် = 10
            scale_map = {
                'သောင်း': 10000,
                'သိန်း': 100000,
                'သန်း': 1000000,
                'ကုဋေ': 10000000
            }
            
            if scale in scale_map:
                total = mult_value * scale_map[scale]
                return str(total)
            return match.group(0)
        
        result = multiplier_scale_pattern.sub(replace_multiplier_scale, result)
        
        # Step 2: Handle "သန်း" with preceding numbers (like သန်းတစ်ရာ = 100 million)
        scale_with_following_pattern = re.compile(
            not_preceded_by_burmese + 
            r'(သန်း|ကုဋေ)\s*(တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး)?\s*(ရာ|ဆယ်|ထောင်|သောင်း|သိန်း)?' +
            not_burmese
        )
        
        def replace_scale_with_following(match):
            scale1 = match.group(1)
            number = match.group(2)
            scale2 = match.group(3)
            
            scale_map1 = {
                'သန်း': 1000000,
                'ကုဋေ': 10000000
            }
            
            if scale1 in scale_map1:
                base_value = scale_map1[scale1]
                
                if number and scale2:
                    number_map = {
                        'တစ်': 1, 'နှစ်': 2, 'သုံး': 3, 'လေး': 4, 'ငါး': 5,
                        'ခြောက်': 6, 'ခုနစ်': 7, 'ရှစ်': 8, 'ကိုး': 9
                    }
                    scale_map2 = {
                        'ဆယ်': 10, 'ရာ': 100, 'ထောင်': 1000, 
                        'သောင်း': 10000, 'သိန်း': 100000
                    }
                    
                    num_val = number_map.get(number, 1)
                    scale_val = scale_map2.get(scale2, 1)
                    total = base_value * num_val * scale_val
                    return str(total)
                elif scale2 == 'ရာ':
                    # Special case: သန်းတစ်ရာ (without number)
                    return str(base_value * 100)
                else:
                    return str(base_value)
            
            return match.group(0)
        
        result = scale_with_following_pattern.sub(replace_scale_with_following, result)
        
        # Step 3: Handle compound tens with ့ connector
        # NOTE: allow a preceding Burmese character (e.g. "ရာ့") by
        # **removing** the negative-look-behind constraint.
        compound_tens_pattern = re.compile(
            r'(နှစ်ဆယ(?:်)?|သုံးဆယ(?:်)?|လေးဆယ(?:်)?|ငါးဆယ(?:်)?|'
            r'ခြောက်ဆယ(?:်)?|ခုနစ်ဆယ(?:်)?|ရှစ်ဆယ(?:်)?|ကိုးဆယ(?:်)?)'
            r'[့်]{0,2}'
            r'(တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး)'
        )
        
        def replace_compound_tens(match):
            tens_map = {
                'နှစ်ဆယ': 20, 'သုံးဆယ': 30, 'လေးဆယ': 40, 'ငါးဆယ': 50,
                'ခြောက်ဆယ': 60, 'ခုနစ်ဆယ': 70, 'ရှစ်ဆယ': 80, 'ကိုးဆယ': 90
            }
            ones_map = {
                'တစ်': 1, 'နှစ်': 2, 'သုံး': 3, 'လေး': 4, 'ငါး': 5,
                'ခြောက်': 6, 'ခုနစ်': 7, 'ရှစ်': 8, 'ကိုး': 9
            }
            tens = tens_map.get(match.group(1), 0)
            ones = ones_map.get(match.group(2), 0)
            return str(tens + ones)
        
        result = compound_tens_pattern.sub(replace_compound_tens, result)
        
        # Step 4: Handle multi-scale expressions with proper remainder capture
        # FIXED: Remove the negative lookahead for __CURRENCY__
        teen = r'ဆယ့်(?:တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး)'
        ones = r'(?:တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး)'
        compound_tens = (
            r'(?:နှစ်ဆယ(?:်)?|သုံးဆယ(?:်)?|လေးဆယ(?:်)?|ငါးဆယ(?:်)?|'
            r'ခြောက်ဆယ(?:်)?|ခုနစ်ဆယ(?:်)?|ရှစ်ဆယ(?:်)?|ကိုးဆယ(?:်)?)'
            r'[့်]{0,2}'
            r'(?:တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး)'
        )
        number_group = rf'(?:{compound_tens}|{teen}|{ones})'

        # Modified pattern to handle multi-scale properly
        # Process all scale expressions iteratively
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            old_result = result
            
            # Pattern to match number + scale with optional remainder
            compound_with_scale_pattern = re.compile(
                rf'({number_group})\s*'
                r'(ကုဋေ|သန်း|သိန်း|သောင်း|ထောင်|ရာ)[့်]?'
            )
            
            # Find all matches
            matches = list(compound_with_scale_pattern.finditer(result))
            
            # Process matches from left to right
            offset = 0
            for match in matches:
                number_word = match.group(1)
                scale_word = match.group(2)
                
                # ones + teens
                number_map = {
                    'တစ်': 1, 'နှစ်': 2, 'သုံး': 3, 'လေး': 4, 'ငါး': 5,
                    'ခြောက်': 6, 'ခုနစ်': 7, 'ရှစ်': 8, 'ကိုး': 9,
                    'ဆယ့်တစ်': 11, 'ဆယ့်နှစ်': 12, 'ဆယ့်သုံး': 13,
                    'ဆယ့်လေး': 14, 'ဆယ့်ငါး': 15, 'ဆယ့်ခြောက်': 16,
                    'ဆယ့်ခုနစ်': 17, 'ဆယ့်ရှစ်': 18, 'ဆယ့်ကိုး': 19
                }
                scale_map = {
                    'ရာ': 100, 'ထောင်': 1000, 'သောင်း': 10000,
                    'သိန်း': 100000, 'သန်း': 1000000, 'ကုဋေ': 10000000
                }
                
                # Get main value
                multiplier = number_map.get(number_word)
                if multiplier is None:
                    # Try to convert compound tens
                    converted_word = result[match.start(1) + offset:match.end(1) + offset]
                    if converted_word.isdigit():
                        multiplier = int(converted_word)
                    else:
                        multiplier = 1
                
                scale_value = scale_map.get(scale_word, 1)
                main_value = multiplier * scale_value
                
                # Check what comes after this match
                after_match_start = match.end() + offset
                after_match_text = result[after_match_start:].strip()
                
                # If there's another scale expression immediately after, it's a multi-scale number
                next_scale_match = compound_with_scale_pattern.match(after_match_text)
                
                if next_scale_match and not after_match_text.startswith('__CURRENCY__'):
                    # This is part of a multi-scale expression, convert to partial number
                    # and let the next iteration handle the rest
                    replacement = str(main_value) + ' '
                    result = result[:match.start() + offset] + replacement + result[match.end() + offset:]
                    offset += len(replacement) - (match.end() - match.start())
                else:
                    # This is the last scale in the sequence
                    replacement = str(main_value) + ' '
                    result = result[:match.start() + offset] + replacement + result[match.end() + offset:]
                    offset += len(replacement) - (match.end() - match.start())
            
            # If nothing changed, we're done
            if result == old_result:
                break
            
            iteration += 1
        
        # Clean up: repeatedly fold chains of space-separated numbers
        # ------------------------------------------------------------------
        # (old folding pass deleted – it ran **too early**)
        # ------------------------------------------------------------------
        
        # Step 5: Handle simple number + scale patterns
        teen = r'ဆယ့်(?:တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး)'
        ones = r'(?:တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး)'
        simple_scale_pattern = re.compile(
            not_preceded_by_burmese +
            rf'({teen}|{ones})\s*(ရာ|ထောင်|သောင်း|သိန်း|သန်း|ကုဋေ)[့်]?' +
            not_burmese
        )
        
        def replace_simple_scale(match):
            number_map = {
                # teens
                'ဆယ့်တစ်': 11, 'ဆယ့်နှစ်': 12, 'ဆယ့်သုံး': 13,
                'ဆယ့်လေး': 14, 'ဆယ့်ငါး': 15, 'ဆယ့်ခြောက်': 16,
                'ဆယ့်ခုနစ်': 17, 'ဆယ့်ရှစ်': 18, 'ဆယ့်ကိုး': 19,
                # ones
                'တစ်': 1, 'နှစ်': 2, 'သုံး': 3, 'လေး': 4, 'ငါး': 5,
                'ခြောက်': 6, 'ခုနစ်': 7, 'ရှစ်': 8, 'ကိုး': 9
            }
            scale_word = match.group(2).rstrip('့')  # Remove ့ if present
            scale_map = {
                'ရာ': 100, 'ထောင်': 1000, 'သောင်း': 10000,
                'သိန်း': 100000, 'သန်း': 1000000, 'ကုဋေ': 10000000
            }
            number = number_map.get(match.group(1), 1)
            scale = scale_map.get(scale_word, 1)
            return str(number * scale)
        
        result = simple_scale_pattern.sub(replace_simple_scale, result)
        
        # Step 6: Handle teen numbers
        teen_patterns = {
            'ဆယ့်တစ်': '11', 'ဆယ့်နှစ်': '12', 'ဆယ့်သုံး': '13', 
            'ဆယ့်လေး': '14', 'ဆယ့်ငါး': '15', 'ဆယ့်ခြောက်': '16', 
            'ဆယ့်ခုနစ်': '17', 'ဆယ့်ရှစ်': '18', 'ဆယ့်ကိုး': '19'
        }
        
        for pattern, replacement in teen_patterns.items():
            result = result.replace(pattern, replacement)
        
        # Step 7: Handle simple tens
        tens_map_simple = {
            'နှစ်ဆယ်': '20', 'သုံးဆယ်': '30', 'လေးဆယ်': '40', 'ငါးဆယ်': '50',
            'ခြောက်ဆယ်': '60', 'ခုနစ်ဆယ်': '70', 'ရှစ်ဆယ်': '80', 'ကိုးဆယ်': '90'
        }
        
        for tens_word, tens_value in tens_map_simple.items():
            pattern = r'(^|[^\u1000-\u109F])' + re.escape(tens_word) + r'($|[^\u1000-\u109F])'
            result = re.sub(pattern, r'\g<1>' + tens_value + r'\g<2>', result)
        
        # Step 8: Handle "တစ်ဆယ်" and "ဆယ်"
        result = re.sub(r'(^|[^\u1000-\u109F])တစ်ဆယ်($|[^\u1000-\u109F])', r'\g<1>10\g<2>', result)
        result = re.sub(r'(^|[^\u1000-\u109F])ဆယ်($|[^\u1000-\u109F])', r'\g<1>10\g<2>', result)
        
        # Step 9: Handle remaining simple numbers
        remaining_words = {
            'သုည': '0',
            'တစ်': '1', 'နှစ်': '2', 'သုံး': '3', 'လေး': '4', 'ငါး': '5',
            'ခြောက်': '6', 'ခုနစ်': '7', 'ရှစ်': '8', 'ကိုး': '9'
        }
        
        sorted_words = sorted(remaining_words.items(), key=lambda x: len(x[0]), reverse=True)
        
        for word, digit in sorted_words:
            pattern = r'(^|[^\u1000-\u109F])' + re.escape(word) + r'($|[^\u1000-\u109F])'
            result = re.sub(pattern, r'\g<1>' + digit + r'\g<2>', result)
        
        # Step 10: Clean up
        # Keep ONE space between two numeric tokens; the final-fold pass
        # needs that space so it can add them (e.g. “2300000 45000” → 2345000)
        result = re.sub(r'(\d+)\s*့်?\s*(\d+)', r'\1 \2', result)
        result = re.sub(r'(\d+)့(\d+)', r'\1\2', result)
        
        # Remove any remaining ် marks that might cause issues
        result = re.sub(r'(\d+)်', r'\1', result)
        
        for placeholder, percentage in protected_percentage:
            result = result.replace(placeholder, percentage)

        # ------------------------------------------------------------------
        # CRITICAL: Restore currency word if it was present
        # ------------------------------------------------------------------
        if protected_currency:
            result = result.replace('__CURRENCY__', protected_currency)
            # Ensure proper spacing between number and currency
            result = re.sub(r'(\d+)(' + re.escape(protected_currency) + r')', r'\1 \2', result)

        # ------------------------------------------------------------------
        # Handle decimal numbers LAST (after all conversions)
        # ------------------------------------------------------------------
        # Pattern for "number ဒဿမ/ဒသမ number(s)"
        decimal_pattern = re.compile(
            r'(\d+)\s+(ဒဿမ|ဒသမ)\s+((?:\d+\s*)+)'
        )
        
        def replace_decimal(match):
            integer_part = match.group(1)
            decimal_marker = match.group(2)  # ဒဿမ or ဒသမ
            decimal_digits = match.group(3).strip()
            
            # Handle multiple digits after decimal (e.g., "7 5" for .75)
            # Remove spaces between digits
            decimal_part = decimal_digits.replace(' ', '')
            
            return f"{integer_part}.{decimal_part}"
        
        result = decimal_pattern.sub(replace_decimal, result)
        
        # ------------------------------------------------------------------
        # NEW • handle digit + scale patterns produced *after* earlier passes
        #      e.g.  "23 သိန်း"  →  23 × 100 000
        # Skip matching when part of 'ရာခိုင်နှုန်း'
        # ------------------------------------------------------------------
        digit_scale_pattern = re.compile(
            r'(\d+)\s*(ရာ|ထောင်|သောင်း|သိန်း|သန်း|ကုဋေ)(?!ခိုင်နှုန်း)'
        )

        def replace_digit_scale(m):
            """Convert '23သိန်း' → '2300000 ' (note trailing space)."""
            digit = int(m.group(1))
            scale_val = {
                'ရာ': 100, 'ထောင်': 1000, 'သောင်း': 10000,
                'သိန်း': 100000, 'သန်း': 1000000, 'ကုဋေ': 10000000
            }[m.group(2)]
            # ➊ multiply, ➋ **append a space** so neighbouring numerals remain
            return f"{digit * scale_val} "

        result = digit_scale_pattern.sub(replace_digit_scale, result)

        # FINAL NUMERIC FOLDING – collapse any adjacent numeric tokens
        def _fold_multi(match):
            a, b = match.group(1), match.group(2)
            # sum if at least one token is multi-digit (so '100 1'→'101', but '2 2' stays '2 2')
            if len(a) > 1 or len(b) > 1:
                return str(int(a) + int(b))
            return f"{a} {b}"

        while True:
            folded = re.sub(r'(\d+)\s+(\d+)', _fold_multi, result)
            if folded == result:
                break
            result = folded

        # remove any residual double-spaces produced in prior steps
        result = re.sub(r'\s+', ' ', result).strip()
        return result

class BurmeseCurrencyProcessor:
    """Process Burmese currency formats"""
    
    def __init__(self, config: BurmeseASRConfig):
        self.config = config
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile currency patterns"""
        self.patterns = [
            # Myanmar Kyat patterns
            (re.compile(r'(\d+)\s*ကျပ်'), r'\1 ကျပ်'),
            # match standalone "K" or "Ks" only, not as part of another word
            (re.compile(r'\bK(?:s)?\.?\s*(\d+)\b', re.IGNORECASE), r'\1 ကျပ်'),
            (re.compile(r'MMK\s*(\d+)', re.IGNORECASE), r'\1 ကျပ်'),
            (re.compile(r'(\d+)\s*MMK', re.IGNORECASE), r'\1 ကျပ်'),
            
            # Foreign currencies
            (re.compile(r'\$\s*(\d+)'), r'\1 ဒေါ်လာ'),
            (re.compile(r'USD\s*(\d+)', re.IGNORECASE), r'\1 ဒေါ်လာ'),
            (re.compile(r'€\s*(\d+)'), r'\1 ယူရို'),
            (re.compile(r'EUR\s*(\d+)', re.IGNORECASE), r'\1 ယူရို'),
            (re.compile(r'¥\s*(\d+)'), r'\1 ယွမ်'),
            (re.compile(r'CNY\s*(\d+)', re.IGNORECASE), r'\1 ယွမ်'),
            (re.compile(r'฿\s*(\d+)'), r'\1 ဘတ်'),
            (re.compile(r'THB\s*(\d+)', re.IGNORECASE), r'\1 ဘတ်'),
            
            # Price expressions
            (re.compile(r'စျေးနှုန်း\s*(\d+)'), r'စျေးနှုန်း \1'),
            (re.compile(r'အဆုံး\s*(\d+)'), r'အဆုံး \1'),
            (re.compile(r'စုစုပေါင်း\s*(\d+)'), r'စုစုပေါင်း \1'),
        ]
    
    def normalize_currency(self, text: str) -> str:
        """Normalize currency in text"""
        if not self.config.normalize_currency or not text:
            return text or ""
        
        result = text
        
        try:
            # Apply patterns in order
            for pattern, replacement in self.patterns:
                result = pattern.sub(replacement, result)
            
            # Handle special case of သိန်း/သန်း (hundred thousand/million) used colloquially
            result = re.sub(r'(\d+)\s*သိန်း(?!\s*ကျပ်)', r'\1 သိန်း', result)
            result = re.sub(r'(\d+)\s*သန်း(?!\s*ကျပ်)', r'\1 သန်း', result)
            
            if self.config.enable_debug_logging and text != result:
                logger.debug(f"Currency normalization: '{text}' → '{result}'")
            
        except Exception as e:
            logger.warning(f"Currency normalization failed: {e}")
            return text
        
        return result


class BurmesePhoneProcessor:
    """Process Burmese phone number formats - FIXED VERSION WITH PROTECTION"""
    
    def __init__(self, config: BurmeseASRConfig):
        self.config = config
        self._compile_patterns()
        self.phone_placeholders = {}  # Store phone conversions
        self.placeholder_counter = 0
    
    def _compile_patterns(self):
        """Compile phone number patterns - FIXED VERSION"""
        # Myanmar phone number patterns
        # Mobile: 09-XXX-XXXXX (11 digits) or older 09-XXXXXXX (9 digits)
        # Landline: varying lengths
        
        self.patterns = [
            # CRITICAL: Very specific patterns first for Burmese mobile numbers
            # This pattern MUST come first to capture '၀၉ ၁၂၃၄၅၆၇၈၉' format
            (re.compile(r'(?<![၀-၉])[၀][၉]\s+[၀-၉]{9}(?![၀-၉])'), self._convert_burmese_mobile_spaced_exact),
            
            # International format with Burmese digits - FIXED with better specificity
            (re.compile(r'\+[၀-၉]{2}\s+[၀-၉]\s+[၀-၉]{9}'), self._convert_burmese_international_spaced_full),
            (re.compile(r'\+[၀-၉]{2,3}\s*[၀-၉]\s*[၀-၉]+'), self._convert_burmese_international_spaced),
            (re.compile(r'\+[၀-၉]{2}\s*[၀-၉]\s*[၀-၉]{8,9}'), self._convert_burmese_international),
            (re.compile(r'\+[၀-၉]{2}[\s\-]?[၀-၉]{1,2}[\s\-]?[၀-၉]{3}[\s\-]?[၀-၉]{4}'), self._convert_burmese_international_formatted),
            
            # Burmese digits with various formatting
            (re.compile(r'[၀-၉]{2,4}[\s\-\.]+[၀-၉]{3,4}[\s\-\.]+[၀-၉]{3,4}(?:[\s\-\.]+[၀-၉]{3})?'), self._convert_burmese_formatted),
            
            # CRITICAL FIX: Only match 9-11 digit sequences that START with valid phone prefixes
            # This prevents arbitrary digit sequences from being treated as phone numbers
            (re.compile(r'(?<![၀-၉])[၀][၁-၉][၀-၉]{7,9}(?![၀-၉])'), self._convert_burmese_continuous),
            
            # More general Burmese mobile patterns
            (re.compile(r'(?<![၀-၉])[၀-၉]{2}\s+[၀-၉]{9}(?![၀-၉])'), self._convert_burmese_mobile_spaced_long),
            
            # Burmese digits with spaces (must come after more specific patterns)
            (re.compile(r'[၀-၉]{2}\s+[၀-၉]{3}\s+[၀-၉]{3}\s+[၀-၉]{3}'), self._convert_burmese_spaced),
            (re.compile(r'[၀-၉]{2}\s+[၀-၉]{7,9}'), self._convert_burmese_spaced_simple),
            (re.compile(r'[၀][၁-၉]\s+[၀-၉]+'), self._convert_burmese_spaced_generic),  # FIXED: Only match if starts with 01-09
            
            # International format with Arabic digits
            (re.compile(r'\+95\s*9\s*(\d{8,9})'), self._convert_international),
            (re.compile(r'\+95\s*(\d{1,2})\s*(\d{6,7})'), self._convert_international_landline),
            (re.compile(r'0095\s*9\s*(\d{8,9})'), self._convert_long_international),
            (re.compile(r'\+959(\d{8,9})'), self._convert_international_continuous),
            (re.compile(r'00959(\d{8,9})'), self._convert_long_international_continuous),
            
            # Mobile numbers with Arabic digits (starting with 09)
            (re.compile(r'\b09(\d{8,9})\b'), self._convert_mobile),
            (re.compile(r'\b09[\s\-](\d{3})[\s\-](\d{5,6})\b'), self._convert_mobile_formatted),
            (re.compile(r'\b09\.(\d{3})\.(\d{5,6})\b'), self._convert_mobile_dotted),
            (re.compile(r'\(09\)\s*(\d{3})[\s\-](\d{3})[\s\-](\d{3})'), self._convert_mobile_parentheses),
            
            # Landline numbers with Arabic digits
            (re.compile(r'\b0([1-8])[\s\-](\d{3})[\s\-](\d{3,4})\b'), self._convert_landline_formatted),
            (re.compile(r'\b0([1-8])(\d{6,7})\b'), self._convert_landline_simple),
            
            # Handle mixed formats
            (re.compile(r'09\s+(\d{3})\s+(\d{3})\s+(\d{3})'), self._convert_mobile_spaced),
            (re.compile(r'01\s+(\d{3})\s+(\d{4})'), self._convert_landline_spaced),
        ]
    
    def _create_placeholder(self, phone_words: str) -> str:
        """Create a unique placeholder for phone number"""
        placeholder = f"__PHONE_{self.placeholder_counter}__"
        self.phone_placeholders[placeholder] = phone_words
        self.placeholder_counter += 1
        return placeholder
    
    def _digits_to_burmese_words(self, digits: str, include_plus: bool = False) -> str:
        """Convert digits to Burmese words for phone reading"""
        digit_map = {
            '0': 'သုည', '1': 'တစ်', '2': 'နှစ်', '3': 'သုံး', '4': 'လေး',
            '5': 'ငါး', '6': 'ခြောက်', '7': 'ခုနစ်', '8': 'ရှစ်', '9': 'ကိုး',
            '+': 'အပေါင်း'
        }
        
        words = []
        
        # Add 'ပေါင်း' for plus sign if needed
        if include_plus:
            words.append('ပေါင်း')
        
        for digit in digits:
            if digit in digit_map:
                words.append(digit_map[digit])
            elif digit.isdigit():
                words.append(digit_map.get(digit, digit))
        
        return ' '.join(words)
    
    def _convert_burmese_to_arabic(self, burmese_text: str) -> str:
        """Convert Burmese digits to Arabic for processing"""
        digit_map = {
            '၀': '0', '၁': '1', '၂': '2', '၃': '3', '၄': '4',
            '၅': '5', '၆': '6', '၇': '7', '၈': '8', '၉': '9'
        }
        result = burmese_text
        for burmese, arabic in digit_map.items():
            result = result.replace(burmese, arabic)
        return result
    
    def _extract_digits_only(self, text: str) -> str:
        """Extract only digits from text"""
        return ''.join(c for c in text if c.isdigit())
    
    def _convert_burmese_mobile_spaced_exact(self, match):
        """Convert exact format '၀၉ ၁၂၃၄၅၆၇၈၉' - SPECIFIC HANDLER"""
        phone_text = match.group(0)
        # Convert to Arabic digits
        arabic_text = self._convert_burmese_to_arabic(phone_text)
        # Extract all digits (remove spaces)
        digits = self._extract_digits_only(arabic_text)
        # Convert ALL digits to words, including the leading 09
        phone_words = self._digits_to_burmese_words(digits)
        return self._create_placeholder(phone_words)
    
    def _convert_burmese_international_spaced_full(self, match):
        """Convert format like '+၉၅ ၉ ၁၂၃၄၅၆၇၈၉' - SPECIFIC METHOD FOR EXACT FORMAT"""
        phone_text = match.group(0)
        # Remove + sign first
        phone_text = phone_text.replace('+', '')
        # Convert to Arabic digits
        arabic_text = self._convert_burmese_to_arabic(phone_text)
        # Extract all digits (this will be the full number including country code)
        digits = self._extract_digits_only(arabic_text)
        # Include plus sign indicator
        phone_words = self._digits_to_burmese_words(digits, include_plus=True)
        return self._create_placeholder(phone_words)
    
    def _convert_burmese_mobile_spaced_long(self, match):
        """Convert format like '၀၉ ၁၂၃၄၅၆၇၈၉' - GENERIC METHOD"""
        phone_text = match.group(0)
        # Convert to Arabic digits
        arabic_text = self._convert_burmese_to_arabic(phone_text)
        # Extract only digits (remove spaces)
        digits = self._extract_digits_only(arabic_text)
        # Convert all digits to words
        phone_words = self._digits_to_burmese_words(digits)
        return self._create_placeholder(phone_words)
    
    def _convert_burmese_formatted(self, match):
        """Convert formatted Burmese digit phone number"""
        phone_text = match.group(0)
        # Convert to Arabic digits first
        arabic_text = self._convert_burmese_to_arabic(phone_text)
        # Extract only digits
        digits = self._extract_digits_only(arabic_text)
        phone_words = self._digits_to_burmese_words(digits)
        return self._create_placeholder(phone_words)
    
    def _convert_burmese_continuous(self, match):
        """Convert continuous Burmese digit phone number - ONLY VALID PHONE NUMBERS"""
        phone_text = match.group(0)
        # Convert to Arabic digits to validate
        digits = self._convert_burmese_to_arabic(phone_text)
        
        # Validate it's actually a phone number
        # Must start with 01-09 for landline/mobile
        if len(digits) >= 9 and len(digits) <= 11 and digits.startswith('0') and digits[1] in '123456789':
            phone_words = self._digits_to_burmese_words(digits)
            return self._create_placeholder(phone_words)
        
        # Not a valid phone number pattern, return as-is
        return phone_text
    
    def _convert_burmese_spaced(self, match):
        """Convert Burmese digits with spaces"""
        phone_text = match.group(0)
        # Convert to Arabic digits
        arabic_text = self._convert_burmese_to_arabic(phone_text)
        # Extract only digits
        digits = self._extract_digits_only(arabic_text)
        phone_words = self._digits_to_burmese_words(digits)
        return self._create_placeholder(phone_words)
    
    def _convert_burmese_spaced_simple(self, match):
        """Convert simple spaced Burmese phone number"""
        phone_text = match.group(0)
        arabic_text = self._convert_burmese_to_arabic(phone_text)
        digits = self._extract_digits_only(arabic_text)
        phone_words = self._digits_to_burmese_words(digits)
        return self._create_placeholder(phone_words)
    
    def _convert_burmese_spaced_generic(self, match):
        """Convert generic Burmese phone number with spaces - NEW METHOD"""
        phone_text = match.group(0)
        # Convert to Arabic digits
        arabic_text = self._convert_burmese_to_arabic(phone_text)
        # Extract only digits
        digits = self._extract_digits_only(arabic_text)
        # Validate it's a phone number
        if len(digits) >= 7 and len(digits) <= 13:  # Include international format
            phone_words = self._digits_to_burmese_words(digits)
            return self._create_placeholder(phone_words)
        return phone_text  # Not a valid phone number
    
    def _convert_burmese_international_spaced(self, match):
        """Convert international format with Burmese digits and spaces - FIXED"""
        phone_text = match.group(0)
        # Remove + sign first
        phone_text = phone_text.replace('+', '')
        # Convert to Arabic digits
        arabic_text = self._convert_burmese_to_arabic(phone_text)
        # Extract digits (this will be the full number including country code)
        digits = self._extract_digits_only(arabic_text)
        # Include plus sign indicator
        phone_words = self._digits_to_burmese_words(digits, include_plus=True)
        return self._create_placeholder(phone_words)
    
    def _convert_burmese_international(self, match):
        """Convert international format with Burmese digits"""
        phone_text = match.group(0)
        # Remove + sign
        phone_text = phone_text.replace('+', '')
        # Convert to Arabic digits
        arabic_text = self._convert_burmese_to_arabic(phone_text)
        # Extract digits (including the country code)
        digits = self._extract_digits_only(arabic_text)
        # Include plus sign indicator
        phone_words = self._digits_to_burmese_words(digits, include_plus=True)
        return self._create_placeholder(phone_words)
    
    def _convert_burmese_international_formatted(self, match):
        """Convert formatted international with Burmese digits"""
        phone_text = match.group(0)
        # Remove + sign
        phone_text = phone_text.replace('+', '')
        arabic_text = self._convert_burmese_to_arabic(phone_text)
        digits = self._extract_digits_only(arabic_text)
        phone_words = self._digits_to_burmese_words(digits, include_plus=True)
        return self._create_placeholder(phone_words)
    
    def _convert_international(self, match):
        """Convert international mobile format"""
        number = match.group(1)
        # Include country code 95 and the 9
        full_number = '959' + number
        phone_words = self._digits_to_burmese_words(full_number, include_plus=True)
        return self._create_placeholder(phone_words)
    
    def _convert_international_landline(self, match):
        """Convert international landline format"""
        area = match.group(1)
        number = match.group(2)
        # Include country code 95
        full_number = '95' + area + number
        phone_words = self._digits_to_burmese_words(full_number, include_plus=True)
        return self._create_placeholder(phone_words)
    
    def _convert_long_international(self, match):
        """Convert long international format (0095...)"""
        number = match.group(1)
        # Include the 0095 as separate digits
        full_number = '00959' + number
        phone_words = self._digits_to_burmese_words(full_number)
        return self._create_placeholder(phone_words)
    
    def _convert_international_continuous(self, match):
        """Convert continuous international format +959..."""
        number = match.group(1)
        full_number = '959' + number
        phone_words = self._digits_to_burmese_words(full_number, include_plus=True)
        return self._create_placeholder(phone_words)
    
    def _convert_long_international_continuous(self, match):
        """Convert continuous long international format 00959..."""
        number = match.group(1)
        full_number = '00959' + number
        phone_words = self._digits_to_burmese_words(full_number)
        return self._create_placeholder(phone_words)
    
    def _convert_mobile(self, match):
        """Convert mobile number"""
        number = '09' + match.group(1)
        phone_words = self._digits_to_burmese_words(number)
        return self._create_placeholder(phone_words)
    
    def _convert_mobile_formatted(self, match):
        """Convert formatted mobile number"""
        number = '09' + match.group(1) + match.group(2)
        phone_words = self._digits_to_burmese_words(number)
        return self._create_placeholder(phone_words)
    
    def _convert_mobile_dotted(self, match):
        """Convert dot-formatted mobile number"""
        number = '09' + match.group(1) + match.group(2)
        phone_words = self._digits_to_burmese_words(number)
        return self._create_placeholder(phone_words)
    
    def _convert_mobile_parentheses(self, match):
        """Convert parentheses-formatted mobile number"""
        number = '09' + match.group(1) + match.group(2) + match.group(3)
        phone_words = self._digits_to_burmese_words(number)
        return self._create_placeholder(phone_words)
    
    def _convert_mobile_spaced(self, match):
        """Convert spaced mobile number format"""
        number = '09' + match.group(1) + match.group(2) + match.group(3)
        phone_words = self._digits_to_burmese_words(number)
        return self._create_placeholder(phone_words)
    
    def _convert_landline_formatted(self, match):
        """Convert formatted landline number"""
        number = '0' + match.group(1) + match.group(2) + match.group(3)
        phone_words = self._digits_to_burmese_words(number)
        return self._create_placeholder(phone_words)
    
    def _convert_landline_simple(self, match):
        """Convert simple landline number"""
        number = '0' + match.group(1) + match.group(2)
        phone_words = self._digits_to_burmese_words(number)
        return self._create_placeholder(phone_words)
    
    def _convert_landline_spaced(self, match):
        """Convert spaced landline number"""
        number = '01' + match.group(1) + match.group(2)
        phone_words = self._digits_to_burmese_words(number)
        return self._create_placeholder(phone_words)
    
    def normalize_phones(self, text: str) -> str:
        """Normalize phone numbers in text - FIXED VERSION"""
        if not self.config.normalize_phone_numbers or not text:
            return text or ""
        
        result = text
        
        try:
            # Clear placeholders for this normalization
            self.phone_placeholders = {}
            self.placeholder_counter = 0
            
            # CRITICAL: Enhanced phone word pattern to protect already-converted phone numbers
            phone_word_pattern = re.compile(
                r'(?:သုည|တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး|ပေါင်း)'
                r'(?:\s+(?:သုည|တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး)){5,}'
            )
            
            # Protect existing phone words
            protected_phone_words = []
            for match in phone_word_pattern.finditer(result):
                placeholder = f"__PHONEWORDS_{len(protected_phone_words)}__"
                protected_phone_words.append((placeholder, match.group(0)))
                result = result[:match.start()] + placeholder + result[match.end():]
            
            # Process patterns in order - prioritizing the most specific patterns first
            for pattern, converter in self.patterns:
                # Find all matches first
                matches = list(pattern.finditer(result))
                
                # Process matches in reverse order to maintain positions
                for match in reversed(matches):
                    start, end = match.span()
                    
                    # Check if this region has already been processed
                    # (i.e., contains a placeholder)
                    match_text = result[start:end]
                    if '__PHONE_' in match_text:
                        continue
                    
                    replacement = converter(match)
                    result = result[:start] + replacement + result[end:]
            
            # Restore protected phone words
            for placeholder, phone_words in protected_phone_words:
                result = result.replace(placeholder, phone_words)
            
            if self.config.enable_debug_logging and text != result:
                logger.debug(f"Phone normalization: '{text}' → '{result}'")
            
        except Exception as e:
            logger.warning(f"Phone normalization failed: {e}")
            return text
        
        return result
    
    def restore_phones(self, text: str) -> str:
        """Restore phone numbers from placeholders"""
        result = text
        
        # Sort placeholders by length (longest first) to avoid partial replacements
        sorted_placeholders = sorted(self.phone_placeholders.items(), 
                                   key=lambda x: len(x[0]), reverse=True)
        
        for placeholder, phone_words in sorted_placeholders:
            result = result.replace(placeholder, phone_words)
        
        return result


class BurmeseDateTimeProcessor:
    """Process Burmese date and time formats - FIXED VERSION"""
    
    def __init__(self, config: BurmeseASRConfig):
        self.config = config
        self._setup_mappings()
        self._compile_patterns()
    
    def _setup_mappings(self):
        """Setup date/time mappings"""
        # Month names
        self.gregorian_months = {
            'ဇန်နဝါရီ': 1, 'ဖေဖော်ဝါရီ': 2, 'မတ်': 3, 'ဧပြီ': 4,
            'မေ': 5, 'ဇွန်': 6, 'ဇူလိုင်': 7, 'ဩဂုတ်': 8,
            'စက်တင်ဘာ': 9, 'အောက်တိုဘာ': 10, 'နိုဝင်ဘာ': 11, 'ဒီဇင်ဘာ': 12
        }
        
        # Myanmar calendar months
        self.myanmar_months = {
            'တန်ခူး': 'တန်ခူးလ',
            'ကဆုန်': 'ကဆုန်လ',
            'နယုန်': 'နယုန်လ',
            'ဝါဆို': 'ဝါဆိုလ',
            'ဝါခေါင်': 'ဝါခေါင်လ',
            'တော်သလင်း': 'တော်သလင်းလ',
            'သီတင်းကျွတ်': 'သီတင်းကျွတ်လ',
            'တန်ဆောင်မုန်း': 'တန်ဆောင်မုန်းလ',
            'နတ်တော်': 'နတ်တော်လ',
            'ပြာသို': 'ပြာသိုလ',
            'တပို့တွဲ': 'တပို့တွဲလ',
            'တပေါင်း': 'တပေါင်းလ'
        }
        
        # Time period words
        self.time_periods = {
            'မနက်': 'မနက်',
            'နေ့လည်': 'နေ့လည်',
            'ညနေ': 'ညနေ',
            'ည': 'ည',
            'နံနက်': 'နံနက်',
            'မွန်းလွဲ': 'မွန်းလွဲ'
        }
        
        # Number words for conversion
        self.number_words = {
            'တစ်': '1', 'နှစ်': '2', 'သုံး': '3', 'လေး': '4', 'ငါး': '5',
            'ခြောက်': '6', 'ခုနစ်': '7', 'ရှစ်': '8', 'ကိုး': '9', 'ဆယ်': '10',
            'ဆယ့်တစ်': '11', 'ဆယ့်နှစ်': '12', 'ဆယ့်သုံး': '13', 'ဆယ့်လေး': '14', 'ဆယ့်ငါး': '15',
            'ဆယ့်ခြောက်': '16', 'ဆယ့်ခုနစ်': '17', 'ဆယ့်ရှစ်': '18', 'ဆယ့်ကိုး': '19',
            'နှစ်ဆယ်': '20', 'နှစ်ဆယ့်တစ်': '21', 'နှစ်ဆယ့်နှစ်': '22', 'နှစ်ဆယ့်သုံး': '23', 
            'နှစ်ဆယ့်လေး': '24', 'နှစ်ဆယ့်ငါး': '25', 'နှစ်ဆယ့်ခြောက်': '26', 
            'နှစ်ဆယ့်ခုနစ်': '27', 'နှစ်ဆယ့်ရှစ်': '28', 'နှစ်ဆယ့်ကိုး': '29',
            'သုံးဆယ်': '30', 'သုံးဆယ့်တစ်': '31'
        }
    
    def _compile_patterns(self):
        """Compile date/time patterns - FIXED VERSION"""
        self.patterns = [
            # Time patterns with number word support
            (re.compile(r'(တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး|ဆယ်|ဆယ့်တစ်|ဆယ့်နှစ်|ဆယ့်သုံး|ဆယ့်လေး|ဆယ့်ငါး|ဆယ့်ခြောက်|ဆယ့်ခုနစ်|ဆယ့်ရှစ်|ဆယ့်ကိုး|နှစ်ဆယ်|နှစ်ဆယ့်တစ်|နှစ်ဆယ့်နှစ်|နှစ်ဆယ့်သုံး|နှစ်ဆယ့်လေး)\s*နာရီ'), self._convert_time_with_number_word),
            
            # Time with နာရီ and မိနစ် - with number words
            (re.compile(r'(တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး|ဆယ်|ဆယ့်တစ်|ဆယ့်နှစ်)\s*နာရီ\s*(ဆယ့်တစ်|ဆယ့်နှစ်|ဆယ့်သုံး|ဆယ့်လေး|ဆယ့်ငါး|ဆယ့်ခြောက်|ဆယ့်ခုနစ်|ဆယ့်ရှစ်|ဆယ့်ကိုး|နှစ်ဆယ်|နှစ်ဆယ့်တစ်|နှစ်ဆယ့်နှစ်|နှစ်ဆယ့်သုံး|နှစ်ဆယ့်လေး|နှစ်ဆယ့်ငါး|နှစ်ဆယ့်ခြောက်|နှစ်ဆယ့်ခုနစ်|နှစ်ဆယ့်ရှစ်|နှစ်ဆယ့်ကိုး|သုံးဆယ်|သုံးဆယ့်တစ်|သုံးဆယ့်နှစ်|သုံးဆယ့်သုံး|သုံးဆယ့်လေး|သုံးဆယ့်ငါး|သုံးဆယ့်ခြောက်|သုံးဆယ့်ခုနစ်|သုံးဆယ့်ရှစ်|သုံးဆယ့်ကိုး|လေးဆယ်|လေးဆယ့်တစ်|လေးဆယ့်နှစ်|လေးဆယ့်သုံး|လေးဆယ့်လေး|လေးဆယ့်ငါး|ငါးဆယ်|ငါးဆယ့်တစ်|ငါးဆယ့်နှစ်|ငါးဆယ့်သုံး|ငါးဆယ့်လေး|ငါးဆယ့်ငါး|ငါးဆယ့်ခြောက်|ငါးဆယ့်ခုနစ်|ငါးဆယ့်ရှစ်|ငါးဆယ့်ကိုး)?\s*မိနစ်'), self._convert_time_with_number_words_full),
            
            # Time patterns for "နာရီခွဲ" (half past)
            (re.compile(r'(တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး|ဆယ်|ဆယ့်တစ်|ဆယ့်နှစ်)\s*နာရီခွဲ'), self._convert_half_hour),
            
            # Time periods with hours
            (re.compile(r'(မနက်|နံနက်|နေ့လည်|မွန်းလွဲ|ညနေ|ည)\s+(တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး|ဆယ်|ဆယ့်တစ်|ဆယ့်နှစ်)\s*နာရီ'), self._convert_time_with_period),
            
            # Minutes only
            (re.compile(r'(ဆယ်|ဆယ့်တစ်|ဆယ့်နှစ်|ဆယ့်သုံး|ဆယ့်လေး|ဆယ့်ငါး|နှစ်ဆယ်|နှစ်ဆယ့်တစ်|နှစ်ဆယ့်နှစ်|နှစ်ဆယ့်သုံး|နှစ်ဆယ့်လေး|နှစ်ဆယ့်ငါး|သုံးဆယ်|သုံးဆယ့်တစ်|သုံးဆယ့်နှစ်|သုံးဆယ့်သုံး|သုံးဆယ့်လေး|သုံးဆယ့်ငါး|လေးဆယ်|လေးဆယ့်ငါး|ငါးဆယ်|ငါးဆယ့်ကိုး)\s*မိနစ်'), self._convert_minutes_only),
            
            # Time patterns with colon
            (re.compile(r'(\d{1,2}):(\d{2})'), self._convert_time),
            (re.compile(r'([၀-၉]{1,2}):([၀-၉]{2})'), self._convert_time_burmese),
            
            # Time with နာရီ (already with numbers)
            (re.compile(r'(\d+)\s*နာရီ\s*(\d*)\s*မိနစ်'), self._normalize_time_words),
            (re.compile(r'(\d+)\s*နာရီ'), r'\1 နာရီ'),
            
            # Date patterns - FIXED to preserve format
            (re.compile(r'([၀-၉]{1,2})[/]([၀-၉]{1,2})[/]([၀-၉]{4})'), self._convert_date_slash_format),
            (re.compile(r'([၀-၉]{1,2})[-]([၀-၉]{1,2})[-]([၀-၉]{4})'), self._convert_date_dash_format),
            (re.compile(r'([၀-၉]{1,2})[.]([၀-၉]{1,2})[.]([၀-၉]{4})'), self._convert_date_dot_format),
            
            # Month + day patterns with number words - ENHANCED
            (re.compile(rf"({'|'.join(self.gregorian_months.keys())})\s+(တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး|ဆယ်|ဆယ့်တစ်|ဆယ့်နှစ်|ဆယ့်သုံး|ဆယ့်လေး|ဆယ့်ငါး|ဆယ့်ခြောက်|ဆယ့်ခုနစ်|ဆယ့်ရှစ်|ဆယ့်ကိုး|နှစ်ဆယ်|နှစ်ဆယ့်တစ်|နှစ်ဆယ့်နှစ်|နှစ်ဆယ့်သုံး|နှစ်ဆယ့်လေး|နှစ်ဆယ့်ငါး|နှစ်ဆယ့်ခြောက်|နှစ်ဆယ့်ခုနစ်|နှစ်ဆယ့်ရှစ်|နှစ်ဆယ့်ကိုး|သုံးဆယ်|သုံးဆယ့်တစ်)\s*ရက်"), self._convert_month_day_with_number_word),
            
            # Month + day patterns with digits
            (re.compile(rf"({'|'.join(self.gregorian_months.keys())})\s+(\d{{1,2}})\s*ရက်"), self._convert_month_day),
            
            # Year patterns
            (re.compile(r'(\d{4})\s*ခုနှစ်'), r'\1 ခုနှစ်'),
            (re.compile(r'([၁၂][၀-၉]{3})\s*ခုနှစ်'), self._convert_year_burmese),
            
            # Myanmar Era
            (re.compile(r'မြန်မာသက္ကရာဇ်\s*(\d+)'), r'မြန်မာသက္ကရာဇ် \1'),
            
            # Relative dates
            (re.compile(r'ယနေ့'), 'ယနေ့'),
            (re.compile(r'မနက်ဖြန်'), 'မနက်ဖြန်'),
            (re.compile(r'မနေ့က'), 'မနေ့က'),
        ]
    
    def _convert_time_with_number_word(self, match):
        """Convert time with number word to digit format"""
        hour_word = match.group(1)
        hour = self.number_words.get(hour_word, hour_word)
        return f"{hour} နာရီ"
    
    def _convert_time_with_number_words_full(self, match):
        """Convert time with both hour and minute as number words"""
        hour_word = match.group(1)
        minute_word = match.group(2) if match.group(2) else None
        
        hour = self.number_words.get(hour_word, hour_word)
        
        if minute_word:
            minute = self.number_words.get(minute_word, minute_word)
            return f"{hour} နာရီ {minute} မိနစ်"
        else:
            return f"{hour} နာရီ"
    
    def _convert_half_hour(self, match):
        """Convert 'နာရီခွဲ' format to hours and 30 minutes"""
        hour_word = match.group(1)
        hour = self.number_words.get(hour_word, hour_word)
        return f"{hour} နာရီ 30 မိနစ်"
    
    def _convert_time_with_period(self, match):
        """Convert time with period (morning, evening, etc.)"""
        period = match.group(1)
        hour_word = match.group(2)
        hour = self.number_words.get(hour_word, hour_word)
        return f"{period} {hour} နာရီ"
    
    def _convert_minutes_only(self, match):
        """Convert minutes only expression"""
        minute_word = match.group(1)
        minute = self.number_words.get(minute_word, minute_word)
        return f"{minute} မိနစ်"
    
    def _convert_time(self, match):
        """Convert time format HH:MM"""
        hour = int(match.group(1))
        minute = int(match.group(2))
        
        result = f"{hour} နာရီ"
        if minute > 0:
            result += f" {minute} မိနစ်"
        
        return result
    
    def _convert_time_burmese(self, match):
        """Convert Burmese time format"""
        # Convert Burmese digits to Arabic first
        hour_burmese = match.group(1)
        minute_burmese = match.group(2)
        
        hour = self._burmese_to_arabic(hour_burmese)
        minute = self._burmese_to_arabic(minute_burmese)
        
        result = f"{hour} နာရီ"
        if int(minute) > 0:
            result += f" {minute} မိနစ်"
        
        return result
    
    def _burmese_to_arabic(self, burmese_num: str) -> str:
        """Convert Burmese digits to Arabic"""
        digit_map = {
            '၀': '0', '၁': '1', '၂': '2', '၃': '3', '၄': '4',
            '၅': '5', '၆': '6', '၇': '7', '၈': '8', '၉': '9'
        }
        return ''.join(digit_map.get(d, d) for d in burmese_num)
    
    def _normalize_time_words(self, match):
        """Normalize time with နာရီ and မိနစ်"""
        hour = match.group(1)
        minute = match.group(2) if match.group(2) else "0"
        
        if minute == "0" or minute == "":
            return f"{hour} နာရီ"
        else:
            return f"{hour} နာရီ {minute} မိနစ်"
    
    def _convert_date_slash_format(self, match):
        """Convert date format with slashes - KEEP FORMAT"""
        day = self._burmese_to_arabic(match.group(1))
        month = self._burmese_to_arabic(match.group(2))
        year = self._burmese_to_arabic(match.group(3))
        return f"{day}/{month}/{year}"
    
    def _convert_date_dash_format(self, match):
        """Convert date format with dashes - KEEP FORMAT"""
        day = self._burmese_to_arabic(match.group(1))
        month = self._burmese_to_arabic(match.group(2))
        year = self._burmese_to_arabic(match.group(3))
        return f"{day}-{month}-{year}"
    
    def _convert_date_dot_format(self, match):
        """Convert date format with dots - KEEP FORMAT"""
        day = self._burmese_to_arabic(match.group(1))
        month = self._burmese_to_arabic(match.group(2))
        year = self._burmese_to_arabic(match.group(3))
        return f"{day}.{month}.{year}"
    
    def _convert_month_day_with_number_word(self, match):
        """Convert month + day format where day is a number word"""
        month = match.group(1)
        day_word = match.group(2)
        day = self.number_words.get(day_word, day_word)
        return f"{month} {day} ရက်"
    
    def _convert_month_day(self, match):
        """Convert month + day format"""
        month = match.group(1)
        day = match.group(2)
        return f"{month} {day} ရက်"
    
    def _convert_year_burmese(self, match):
        """Convert Burmese year"""
        year_burmese = match.group(1)
        year = self._burmese_to_arabic(year_burmese)
        return f"{year} ခုနှစ်"
    
    def normalize_datetime(self, text: str) -> str:
        """Normalize date and time in text"""
        if not (self.config.normalize_date_format or self.config.normalize_time_format) or not text:
            return text or ""
        
        result = text
        
        try:
            # Apply patterns in order
            for pattern, converter in self.patterns:
                if callable(converter):
                    result = pattern.sub(converter, result)
                else:
                    result = pattern.sub(converter, result)
            
            if self.config.enable_debug_logging and text != result:
                logger.debug(f"DateTime normalization: '{text}' → '{result}'")
            
        except Exception as e:
            logger.warning(f"DateTime normalization failed: {e}")
            return text
        
        return result


class BurmeseFillerProcessor:
    """Process Burmese filler words and hesitation sounds"""
    
    def __init__(self, config: BurmeseASRConfig):
        self.config = config
        self._setup_filler_lists()
    
    def _setup_filler_lists(self):
        """Setup Burmese filler words"""
        # Common filler words that should be removed when standalone
        self.filler_words = [
            'အော်',  # Oh
            'နော်',  # You know
        ]
        
        # Words that should NOT be removed (they have meaning)
        self.preserve_words = [
            'အင်း',  # Um/yes - has meaning as acknowledgment
            'ဟုတ်လား',  # Right? - has meaning as question
        ]
        
        # Hesitation sounds (should be removed)
        self.hesitation_sounds = [
            'အာ', 'အီး', 'အမ်', 'အဲ', 'အို',
        ]
        
        # Patterns for hesitation with ellipsis
        self.hesitation_patterns = [
            r'အာ\s*\.+', r'အီး\s*\.+', r'အမ်\s*\.+', 
            r'အဲ\s*\.+', r'အို\s*\.+',
        ]
        
        # Discourse particles that might be removed in casual speech
        self.discourse_particles = [
            'ပဲ', 'လေ', 'ကွာ', 'ကွယ်', 'ပါ', 'ဗျာ', 'ဗျ',
        ]
    
    def remove_fillers(self, text: str) -> str:
        """Remove filler words from text"""
        if not self.config.remove_filler_words or not text:
            return text or ""
        
        result = text
        
        try:
            # First, remove hesitation patterns with ellipsis
            for pattern in self.hesitation_patterns:
                result = re.sub(pattern, '', result, flags=re.IGNORECASE)
            
            # Remove standalone hesitation sounds
            for sound in self.hesitation_sounds:
                # Remove at start of text
                result = re.sub(rf'^{re.escape(sound)}\s+', '', result, flags=re.IGNORECASE)
                # Remove at end of text
                result = re.sub(rf'\s+{re.escape(sound)}$', '', result, flags=re.IGNORECASE)
                # Remove when surrounded by spaces
                result = re.sub(rf'\s+{re.escape(sound)}\s+', ' ', result, flags=re.IGNORECASE)
            
            # Remove standalone filler words (careful with context)
            for filler in self.filler_words:
                # Only remove if it's truly standalone
                patterns = [
                    rf'^{re.escape(filler)}\s+',  # Start of text
                    rf'\s+{re.escape(filler)}$',  # End of text
                    rf'\s+{re.escape(filler)}\s+',  # Surrounded by spaces
                ]
                
                for pattern in patterns:
                    result = re.sub(pattern, ' ', result, flags=re.IGNORECASE)
            
            # Clean up extra spaces
            result = re.sub(r'\s+', ' ', result).strip()
            
            if self.config.enable_debug_logging and text != result:
                logger.debug(f"Filler removal: '{text}' → '{result}'")
            
        except Exception as e:
            logger.warning(f"Filler removal failed: {e}")
            return text
        
        return result


class BurmeseAbbreviationExpander:
    """Expand Burmese abbreviations"""
    
    def __init__(self, config: BurmeseASRConfig):
        self.config = config
        self._setup_abbreviations()
    
    def _setup_abbreviations(self):
        """Setup common Burmese abbreviations"""
        self.abbreviations = {
            # Titles
            'ဦး': 'ဦး',  # Mr. (keep as is)
            'ဒေါ်': 'ဒေါ်',  # Mrs. (keep as is)
            'ဒေါက်တာ': 'ဒေါက်တာ',
            'ပရော်ဖက်ဆာ': 'ပရော်ဖက်ဆာ',
            
            # Common abbreviations
            'နံ': 'နံပါတ်',
            'ဥ': 'ဥက္ကဋ္ဌ',
            'လ': 'လမ်း',
            'ရပ်': 'ရပ်ကွက်',
            
            # Organizations
            'ကုလ': 'ကုလသမဂ္ဂ',
            'အာဆီယံ': 'အရှေ့တောင်အာရှနိုင်ငံများအဖွဲ့',
        }
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand abbreviations in text - FIXED VERSION"""
        if not self.config.expand_abbreviations or not text:
            return text or ""
        
        result = text
        
        try:
            # Define Burmese character pattern for word boundaries
            burmese_char = r'[\u1000-\u109F\u104A\u104B]'  # Include punctuation marks
            not_burmese = r'(?!' + burmese_char + ')'
            not_preceded_by_burmese = r'(?<!' + burmese_char + ')'
            
            # CRITICAL: Protect Myanmar month names from abbreviation expansion
            myanmar_months = [
                'တန်ခူးလ', 'ကဆုန်လ', 'နယုန်လ', 'ဝါဆိုလ', 'ဝါခေါင်လ',
                'တော်သလင်းလ', 'သီတင်းကျွတ်လ', 'တန်ဆောင်မုန်းလ', 'နတ်တော်လ',
                'ပြာသိုလ', 'တပို့တွဲလ', 'တပေါင်းလ'
            ]
            
            # Create placeholders for Myanmar months
            month_placeholders = {}
            for i, month in enumerate(myanmar_months):
                if month in result:
                    placeholder = f"__MYANMAR_MONTH_{i}__"
                    month_placeholders[placeholder] = month
                    result = result.replace(month, placeholder)
            
            # Apply simple replacements
            for abbrev, expansion in self.abbreviations.items():
                # Use proper boundaries for multi-character abbreviations
                if len(abbrev) > 1:
                    # Pattern: not preceded by Burmese char + abbrev + not followed by Burmese char
                    pattern = not_preceded_by_burmese + re.escape(abbrev) + not_burmese
                    result = re.sub(pattern, expansion, result)
                else:
                    # For single character abbreviations, be more careful
                    # Only expand if followed by appropriate context
                    if abbrev == 'နံ':
                        # FIXED: More precise pattern to avoid double expansion
                        # Only expand နံ when NOT already followed by ပါတ် and when followed by space/digits
                        # Pattern: နံ not followed by ပါတ် but followed by whitespace and then digits
                        result = re.sub(r'နံ(?!ပါတ်)(?=\s*\d)', r'နံပါတ်', result)
                    elif abbrev == 'လ':
                        # Only expand လ if it appears after Burmese text and at a boundary
                        # AND not part of Myanmar month names (already protected)
                        # Pattern: Burmese chars + optional space + လ + not followed by Burmese char
                        pattern = r'(' + burmese_char + r'+)\s*လ' + not_burmese
                        result = re.sub(pattern, r'\1လမ်း', result)
                    elif abbrev == 'ရပ်':
                        # Expand ရပ် to ရပ်ကွက် when appropriate
                        pattern = not_preceded_by_burmese + r'ရပ်' + not_burmese
                        result = re.sub(pattern, 'ရပ်ကွက်', result)
            
            # Restore Myanmar months
            for placeholder, month in month_placeholders.items():
                result = result.replace(placeholder, month)
            
            if self.config.enable_debug_logging and text != result:
                logger.debug(f"Abbreviation expansion: '{text}' → '{result}'")
            
        except Exception as e:
            logger.warning(f"Abbreviation expansion failed: {e}")
            return text
        
        return result

class BurmeseUnitProcessor:
    """Process Burmese units and measurements - FIXED FOR ASR"""
    
    def __init__(self, config: BurmeseASRConfig):
        self.config = config
        self._setup_unit_mappings()
        self._compile_patterns()
    
    def _setup_unit_mappings(self):
        """Setup unit mappings - NO ABBREVIATIONS FOR ASR"""
        # NO abbreviations for ASR - we want full words
        self.unit_abbreviations = {}  # Empty - no abbreviations
        
        # All unit words (for pattern matching)
        self.unit_words = [
            # Length/Distance
            'မီတာ', 'ကီလိုမီတာ', 'စင်တီမီတာ', 'မီလီမီတာ',
            'တောင်', 'ထွာ', 'မိုင်', 'မိုက်', 'ပေ', 'ကိုက်', 'ယာဒ်',
            
            # Weight
            'ကီလိုဂရမ်', 'ကီလို', 'ဂရမ်', 'တန်', 'ပိဿာ', 'ကျပ်သား', 'မူးသား',
            'ပေါင်', 'အောင်စ',
            
            # Volume
            'လီတာ', 'မီလီလီတာ', 'ဂါလံ', 'တင်း', 'ဖြည်း', 'ပြည်',
            
            # Area
            'ဧက', 'စတုရန်းမီတာ', 'ဟက်တာ', 'စတုရန်းပေ', 'စတုရန်းမိုင်',
            
            # Temperature
            'ဒီဂရီ', 'ဒီဂရီစင်တီဂရိတ်', 'ဒီဂရီဖာရင်ဟိုက်', 'စင်တီဂရိတ်', 'ဖာရင်ဟိုက်',
            
            # Time (for speed)
            'နာရီ', 'မိနစ်', 'စက္ကန့်'
        ]
        
        # Number words for unit conversion
        self.number_words = {
            'တစ်': '1', 'နှစ်': '2', 'သုံး': '3', 'လေး': '4', 'ငါး': '5',
            'ခြောက်': '6', 'ခုနစ်': '7', 'ရှစ်': '8', 'ကိုး': '9', 'ဆယ်': '10',
            'ဆယ့်တစ်': '11', 'ဆယ့်နှစ်': '12', 'ဆယ့်သုံး': '13', 'ဆယ့်လေး': '14', 'ဆယ့်ငါး': '15',
            'ဆယ့်ခြောက်': '16', 'ဆယ့်ခုနစ်': '17', 'ဆယ့်ရှစ်': '18', 'ဆယ့်ကိုး': '19',
            'နှစ်ဆယ်': '20', 'နှစ်ဆယ့်ငါး': '25', 'သုံးဆယ်': '30', 'သုံးဆယ့်ခြောက်': '36',
            'လေးဆယ်': '40', 'ငါးဆယ်': '50', 'ခြောက်ဆယ်': '60', 'ခုနစ်ဆယ်': '70',
            'ရှစ်ဆယ်': '80', 'ကိုးဆယ်': '90', 'ကိုးဆယ့်ကိုး': '99',
            'တစ်ရာ': '100', 'တစ်ရာ့နှစ်ဆယ်': '120'
        }
    
    def _compile_patterns(self):
        """Compile unit-related patterns - FIXED FOR ASR"""
        # Sort unit words by length (longest first) to match correctly
        sorted_units = sorted(self.unit_words, key=len, reverse=True)
        
        # Create pattern for number word + unit (no space)
        number_pattern = '|'.join(re.escape(num) for num in sorted(self.number_words.keys(), key=len, reverse=True))
        # Exclude 'ရာခိုင်နှုန်း' and 'ရာနှုန်း' from unit pattern to prevent percentage issues
        unit_pattern = '|'.join(re.escape(unit) for unit in sorted_units if unit not in ['ရာခိုင်နှုန်း', 'ရာနှုန်း'])
        

        self.patterns = [
            # Speed: numeric per-hour expressions
            # e.g. "60 ကီလိုမီတာ တစ်နာရီ" or "60 ကီလိုမီတာ 1 နာရီ"
            (re.compile(
                rf'(\d+)\s+('
                  + '|'.join(re.escape(u) for u in ['ကီလိုမီတာ','မိုင်','မီတာ'])
                  + r')\s+(?:\d+|တစ်)\s*နာရီ'
            ), self._convert_speed_unit),

            # Pattern 1: Number word directly attached to unit
            (re.compile(rf'({number_pattern})({unit_pattern})(?![\u1000-\u109F])'),
             self._convert_number_unit),

            # Pattern 2: Already digit + unit (add space if missing)
            (re.compile(r'(\d+(?:\.\d+)?)(' + unit_pattern + r')'),
             self._add_space_to_digit_unit),

            # Pattern 3: Percentage special handling – FIXED TO PRESERVE 'ရာ'
            (re.compile(r'(\d+)\s*(ရာခိုင်နှုန်း|ရာနှုန်း)'),
             self._fix_percentage),
            (re.compile(r'(' + number_pattern + r')\s*(ရာခိုင်နှုန်း|ရာနှုန်း)'),
             self._fix_percentage_with_word),

            # Pattern 5: Decimal with unit (ensure space)
            (re.compile(r'(\d+\.\d+)(' + unit_pattern + r')'),
             self._add_space_to_decimal_unit),
        ]
    
    def _convert_number_unit(self, match):
        """Convert number word + unit to digit + space + unit"""
        number_word = match.group(1)
        unit = match.group(2)
        
        # Convert number word to digit
        digit = self.number_words.get(number_word, number_word)
        
        # For ASR, keep full unit name (no abbreviation)
        return f"{digit} {unit}"
    
    def _add_space_to_digit_unit(self, match):
        """Add space between digit and unit if missing"""
        number = match.group(1)
        unit = match.group(2)
        
        # For ASR, keep full unit name (no abbreviation)
        return f"{number} {unit}"
    
    def _add_space_to_decimal_unit(self, match):
        """Ensure space after decimal number before unit"""
        decimal = match.group(1)
        unit = match.group(2)
        
        # For ASR, keep full unit name (no abbreviation)
        return f"{decimal} {unit}"
    
    def _fix_percentage(self, match):
        """Fix percentage - preserve 'ရာ' word"""
        number = match.group(1)
        percentage_word = match.group(2)
        return f"{number} {percentage_word}"
    
    def _fix_percentage_with_word(self, match):
        """Fix percentage with number word - preserve 'ရာ'"""
        number_word = match.group(1)
        percentage_word = match.group(2)
        
        # Convert number word to digit
        digit = self.number_words.get(number_word, number_word)
        return f"{digit} {percentage_word}"
    
    def _convert_speed_unit(self, match):
        """Convert speed expressions to use slash notation"""
        speed = match.group(1)
        unit = match.group(2)
        
        return f"{speed} {unit}/နာရီ"
    
    def _convert_speed_unit_with_word(self, match):
        """Convert speed expressions with number word to use slash notation"""
        speed_word = match.group(1)
        unit = match.group(2)
        
        # Convert number word to digit
        speed = self.number_words.get(speed_word, speed_word)
        return f"{speed} {unit}/နာရီ"
    
    def normalize_units(self, text: str) -> str:
        """Normalize units and measurements in text - FIXED FOR ASR"""
        if not text:
            return ""
        
        result = text
        
        try:
            # CRITICAL: Protect percentages BEFORE any number conversion
            # This prevents 'ရာ' from being converted to 100
            percentage_placeholders = []
            
            # Find all percentage expressions
            percentage_patterns = [
                re.compile(r'(' + '|'.join(re.escape(num) for num in self.number_words.keys()) + r')\s*(ရာခိုင်နှုန်း|ရာနှုန်း)'),
                re.compile(r'(\d+(?:\.\d+)?)\s*(ရာခိုင်နှုန်း|ရာနှုန်း)'),
            ]
            
            placeholder_counter = 0
            for pattern in percentage_patterns:
                for match in pattern.finditer(result):
                    placeholder = f'__PERCENTAGE_{placeholder_counter}__'
                    if match.lastindex == 2:  # Has groups
                        if result[match.start(1):match.end(1)] in self.number_words:
                            # Convert number word to digit
                            digit = self.number_words.get(match.group(1), match.group(1))
                            replacement = f"{digit} {match.group(2)}"
                        else:
                            replacement = match.group(0)
                    else:
                        replacement = match.group(0)
                    
                    percentage_placeholders.append((placeholder, replacement))
                    result = result[:match.start()] + placeholder + result[match.end():]
                    placeholder_counter += 1
            
            # Apply all patterns
            for pattern, converter in self.patterns:
                result = pattern.sub(converter, result)
            
            # Restore percentage placeholders
            for placeholder, original in percentage_placeholders:
                result = result.replace(placeholder, original)
            
            # Final cleanup - ensure single space between number (including decimals) and unit
            number_unit_pattern = r'(\d+(?:\.\d+)?)\s+(' + '|'.join(re.escape(u) for u in self.unit_words) + r')'
            result = re.sub(number_unit_pattern, r'\1 \2', result)
            # Ensure space before percentage words as well
            result = re.sub(r'(\d+(?:\.\d+)?)\s*(ရာခိုင်နှုန်း|ရာနှုန်း)', r'\1 \2', result)
            
            if self.config.enable_debug_logging and text != result:
                logger.debug(f"Unit normalization: '{text}' → '{result}'")
            
        except Exception as e:
            logger.warning(f"Unit normalization failed: {e}")
            return text
        
        return result

class BurmeseASRNormalizer:
    """Main Burmese ASR Normalizer"""
    
    def __init__(self, config: Optional[BurmeseASRConfig] = None):
        self.config = config or BurmeseASRConfig()
        
        # Initialize components
        self.zawgyi_converter = ZawgyiUnicodeConverter(self.config)
        self.number_converter = BurmeseNumberConverter(self.config)
        self.currency_processor = BurmeseCurrencyProcessor(self.config)
        self.phone_processor = BurmesePhoneProcessor(self.config)
        self.datetime_processor = BurmeseDateTimeProcessor(self.config)
        self.filler_processor = BurmeseFillerProcessor(self.config)
        self.abbreviation_expander = BurmeseAbbreviationExpander(self.config)
        self.unit_processor = BurmeseUnitProcessor(self.config)
        
        if self.config.enable_debug_logging:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.debug("Burmese ASR Normalizer initialized")
    
    def _convert_percentage_symbols(self, text: str) -> str:
        """Convert % symbols to Burmese word form"""
        if not text:
            return text
        
        try:
            # Pattern to match number followed by %
            result = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 ရာခိုင်နှုန်း', text)
            
            if self.config.enable_debug_logging and text != result:
                logger.debug(f"Percentage conversion: '{text}' → '{result}'")
            
            return result
        except Exception as e:
            logger.warning(f"Percentage conversion failed: {e}")
            return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode representation"""
        if not self.config.normalize_unicode or not text:
            return text or ""
        
        try:
            # Apply Unicode normalization (NFC for consistency)
            text = unicodedata.normalize('NFC', text)
            
            # Fix common Unicode issues
            # Remove zero-width characters
            text = text.replace('\u200b', '')  # Zero-width space
            text = text.replace('\u200c', '')  # Zero-width non-joiner
            text = text.replace('\u200d', '')  # Zero-width joiner
            text = text.replace('\ufeff', '')  # BOM
            
            return text
            
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}")
            return text
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation while preserving Burmese text integrity, decimal points, date formats, and percentage signs"""
        if not self.config.remove_punctuation or not text:
            return text or ""
        
        try:
            # First, protect decimal points between digits
            # Replace digit + . + digit with placeholder
            text = re.sub(r'(\d)\.(\d)', r'\1__DECIMAL__\2', text)
            
            # CRITICAL: Protect percentage signs for conversion
            text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1__PERCENT__', text)
            
            # CRITICAL: Protect date formats with slashes, dashes, and dots
            # Pattern for dates like 15/3/2024, 15-3-2024, 15.3.2024
            date_slash_pattern = re.compile(r'(\d{1,2})/(\d{1,2})/(\d{2,4})')
            date_dash_pattern = re.compile(r'(\d{1,2})-(\d{1,2})-(\d{2,4})')
            date_dot_pattern = re.compile(r'(\d{1,2})\.(\d{1,2})\.(\d{2,4})')
            
            # Replace with placeholders
            date_placeholders = []
            
            # Process slash dates
            for match in date_slash_pattern.finditer(text):
                placeholder = f"__DATE_SLASH_{len(date_placeholders)}__"
                date_placeholders.append((placeholder, match.group(0)))
                text = text[:match.start()] + placeholder + text[match.end():]
            
            # Process dash dates
            for match in date_dash_pattern.finditer(text):
                placeholder = f"__DATE_DASH_{len(date_placeholders)}__"
                date_placeholders.append((placeholder, match.group(0)))
                text = text[:match.start()] + placeholder + text[match.end():]
            
            # Process dot dates (but be careful not to conflict with decimal protection)
            # Only match if it's clearly a date pattern (day.month.year)
            for match in date_dot_pattern.finditer(text):
                # Check if this looks like a date (not a decimal)
                day = int(match.group(1))
                month = int(match.group(2))
                if 1 <= day <= 31 and 1 <= month <= 12:  # Basic date validation
                    placeholder = f"__DATE_DOT_{len(date_placeholders)}__"
                    date_placeholders.append((placeholder, match.group(0)))
                    text = text[:match.start()] + placeholder + text[match.end():]
            
            # Burmese punctuation marks
            burmese_punct = '၊။၌၍၏'  # Comma, period, locative, completed action, genitive
            
            # Remove Burmese punctuation
            for punct in burmese_punct:
                text = text.replace(punct, ' ')
            
            # Remove ASCII punctuation (except underscore, slash, and %)
            skip_punct = {'_', '/', '%'}
            for punct in string.punctuation:
                if punct in skip_punct:
                    continue
                text = text.replace(punct, ' ')

            # Remove other Unicode punctuation (preserve underscore, slash, and %)
            result_chars = []
            for char in text:
                if char in ('_', '/', '%'):
                    result_chars.append(char)
                elif unicodedata.category(char).startswith('P'):
                    result_chars.append(' ')
                else:
                    result_chars.append(char)
            text = ''.join(result_chars)
            
            # Restore decimal points
            text = text.replace('__DECIMAL__', '.')
            
            # Restore percentage signs
            text = text.replace('__PERCENT__', '%')
            
            # Restore date formats
            for placeholder, original_date in date_placeholders:
                text = text.replace(placeholder, original_date)
            
            # Clean up spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.warning(f"Punctuation removal failed: {e}")
            return text
    
    def _merge_repeated_chars(self, text: str) -> str:
        """Merge repeated characters (for ASR errors)"""
        if not self.config.merge_repeated_chars or not text:
            return text or ""
        
        try:
            # Only merge repeated spaces and punctuation
            text = re.sub(r' {2,}', ' ', text)
            text = re.sub(r'([!?.])\1+', r'\1', text)
            
            # Don't merge Burmese characters as repetition might be meaningful
            
            return text
            
        except Exception as e:
            logger.warning(f"Repeated char merging failed: {e}")
            return text
    
    def _final_cleanup(self, text: str) -> str:
        """Final text cleanup"""
        try:
            # Normalize whitespace
            if self.config.normalize_whitespace:
                text = re.sub(r'\s+', ' ', text).strip()
            
            # Convert to lowercase
            if self.config.lowercase_output:
                text = text.lower()
            
            return text
            
        except Exception as e:
            logger.warning(f"Final cleanup failed: {e}")
            return str(text).lower().strip() if text else ""
    
    def _validate_output(self, original: str, normalized: str) -> bool:
        """Validate normalization output"""
        if not self.config.enable_validation:
            return True
        
        try:
            issues = []
            
            # Check for significant text loss
            if len(normalized.strip()) == 0 and len(original.strip()) > 0:
                issues.append("Complete text loss")
            
            # Check for excessive length change
            if len(normalized) > len(original) * 3:
                issues.append("Excessive length increase")
            elif len(normalized) < len(original) * 0.1:
                issues.append("Excessive length reduction")
            
            if issues:
                logger.warning(f"Validation issues: {issues}")
                if self.config.strict_mode:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return not self.config.strict_mode
    
    def normalize(self, text: str) -> str:
        """Main normalization method - FIXED WITH PHONE PROTECTION"""
        if not text:
            return ""
        
        if not isinstance(text, str):
            text = str(text)
        
        if len(text) > self.config.max_text_length:
            logger.warning(f"Text too long: {len(text)} > {self.config.max_text_length}")
            if self.config.strict_mode:
                raise ValueError("Text exceeds maximum length")
            text = text[:self.config.max_text_length]
        
        if self.config.enable_debug_logging:
            logger.debug(f"=== BURMESE ASR NORMALIZATION START ===")
            logger.debug(f"Input: '{text}'")
        
        try:
            original = text
            
            # Step 1: Convert from Zawgyi to Unicode if needed
            text = self.zawgyi_converter.convert_to_unicode(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After Zawgyi conversion: '{text}'")
            
            # Step 2: Unicode normalization
            text = self._normalize_unicode(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After Unicode normalization: '{text}'")
            
            # Step 3: CRITICAL - Normalize phone numbers and convert to placeholders
            text = self.phone_processor.normalize_phones(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After phone normalization (with placeholders): '{text}'")
            
            # CRITICAL NEW STEP: Protect phone words that are already in word form
            # This prevents them from being converted to numbers later
            phone_word_pattern = re.compile(
                r'(?:သုည|တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး|ပေါင်း)'
                r'(?:\s+(?:သုည|တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး)){5,}'
            )
            
            protected_sequences = []
            for match in phone_word_pattern.finditer(text):
                placeholder = f"__PROTECTEDPHONE_{len(protected_sequences)}__"
                protected_sequences.append((placeholder, match.group(0)))
                text = text[:match.start()] + placeholder + text[match.end():]
            
            if self.config.enable_debug_logging and protected_sequences:
                logger.debug(f"Protected phone sequences: {len(protected_sequences)}")
            
            # Step 4: Convert Burmese digits to Arabic (phone numbers are protected)
            text = self.number_converter.convert_burmese_digits_to_arabic(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After digit conversion: '{text}'")
            
            # Step 5: Convert number words to digits (phone numbers are protected)
            text = self.number_converter.convert_number_words_to_digits(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After number word conversion: '{text}'")
            
            # fix Burmese number words glued to other text (e.g. "သုံးယောက်" → "သုံး ယောက်")
            # avoid splitting when part of a compound-ten (e.g. သုံးဆယ်)
            numword_pattern = r'(?<![\u1000-\u109F])(တစ်|နှစ်|သုံး|လေး|ငါး|ခြောက်|ခုနစ်|ရှစ်|ကိုး)(?!ဆယ်)(?=[\u1000-\u109F])'
            def _split_numword(m):
                # map the Burmese word to its digit
                val = self.number_converter.number_words.get(m.group(1), m.group(1))
                return f"{val} "
            text = re.sub(numword_pattern, _split_numword, text)
            
            # Step 6: Normalize currency
            text = self.currency_processor.normalize_currency(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After currency normalization: '{text}'")
            
            # Step 6.5: Normalize units and measurements and percentage
            text = self.unit_processor.normalize_units(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After unit normalization: '{text}'")
            text = self._convert_percentage_symbols(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After percentage conversion: '{text}'")

            # Step 7: Normalize date/time
            text = self.datetime_processor.normalize_datetime(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After datetime normalization: '{text}'")
            
            # Step 8: Expand abbreviations
            text = self.abbreviation_expander.expand_abbreviations(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After abbreviation expansion: '{text}'")
            
            # CRITICAL: Restore protected phone sequences BEFORE restoring phone placeholders
            for placeholder, phone_sequence in protected_sequences:
                text = text.replace(placeholder, phone_sequence)
            
            # Step 9: Restore phone numbers from placeholders
            text = self.phone_processor.restore_phones(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After phone restoration: '{text}'")
            
            # Step 10: Remove filler words
            text = self.filler_processor.remove_fillers(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After filler removal: '{text}'")
            
            # Step 11: Remove punctuation
            text = self._remove_punctuation(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After punctuation removal: '{text}'")
            
            # Step 12: Merge repeated characters
            text = self._merge_repeated_chars(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After repeated char merging: '{text}'")
            
            # Step 13: Final cleanup
            text = self._final_cleanup(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After final cleanup: '{text}'")
            
            # Step 14: Validation
            if not self._validate_output(original, text):
                logger.error("Validation failed")
                if self.config.strict_mode:
                    raise ValueError("Normalization validation failed")
                return self._final_cleanup(original)
            
            if self.config.enable_debug_logging:
                logger.debug(f"=== FINAL RESULT: '{text}' ===")
            
            return text
            
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            if self.config.enable_debug_logging:
                logger.error(traceback.format_exc())
            
            if self.config.strict_mode:
                raise
            
            # Safe fallback
            try:
                return self._final_cleanup(original)
            except:
                return str(original).lower().strip()

class BurmeseASRNormaliser:
    """AudioBench-compatible Burmese normaliser wrapper"""
    
    def __init__(self, enable_disfluency_removal: bool = True, **kwargs):
        """Initialize the Burmese normalizer with AudioBench parameters"""
        self.enable_disfluency_removal = enable_disfluency_removal
        
        # Initialize your Burmese normalizer config
        config = BurmeseASRConfig(
            remove_filler_words=enable_disfluency_removal,
            convert_zawgyi=True,
            normalize_currency=True,
            convert_burmese_digits=True,  # Convert ၀-၉ to 0-9
            convert_numbers_to_words=False,  # Keep as digits for ASR
            lowercase_output=True,
            normalize_whitespace=True,
            expand_abbreviations=True,
            enable_validation=True
        )
        
        # Initialize the main normalizer
        self.normalizer = BurmeseASRNormalizer(config)
    
    def normalize(self, text: str) -> str:
        """Compatibility method for the new normalizer system"""
        return self.normalize_for_asr_eval(text)
    
    def normalize_for_asr_eval(self, text: str, 
                               use_preprocessing: bool = True,
                               apply_asr_standardization: bool = True) -> str:
        """Main method called by AudioBench for text normalization"""
        if not text:
            return ""
        
        # Use your Burmese normalizer's main normalize method
        return self.normalizer.normalize(text)