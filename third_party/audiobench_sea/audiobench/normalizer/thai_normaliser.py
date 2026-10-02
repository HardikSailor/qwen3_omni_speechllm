#!/usr/bin/env python3
"""
Thai ASR Normalizer for Fair CER Calculation
Based on PyThaiNLP and specialized Thai text processing
"""

import re
import logging
import string
import unicodedata
from typing import List, Dict, Optional, Tuple, Union, Set
from dataclasses import dataclass
import traceback

# Core Thai NLP libraries
try:
    import pythainlp
    from pythainlp.util import (
        normalize as thai_normalize,
        thai_digit_to_arabic_digit,
        arabic_digit_to_thai_digit,
        isthai,
        isthaichar
    )
    from pythainlp.tokenize import word_tokenize, sent_tokenize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    print("Warning: PyThaiNLP not available - install with: pip install pythainlp")

# AttaCut for fast tokenization
try:
    from attacut import tokenize as attacut_tokenize
    ATTACUT_AVAILABLE = True
except ImportError:
    ATTACUT_AVAILABLE = False
    print("Warning: AttaCut not available - install with: pip install attacut")

# DeepCut for accurate tokenization - lazy import to avoid circular dependencies
DEEPCUT_AVAILABLE = False
def _check_deepcut_availability():
    """Check if deepcut is available without importing it at module level"""
    global DEEPCUT_AVAILABLE
    if not DEEPCUT_AVAILABLE:
        try:
            import deepcut
            DEEPCUT_AVAILABLE = True
            return deepcut
        except ImportError:
            DEEPCUT_AVAILABLE = False
            print("Warning: DeepCut not available - install with: pip install deepcut")
            return None
    else:
        import deepcut
        return deepcut

# NeMo for potential future integration
try:
    from nemo_text_processing.text_normalization.normalize import Normalizer
    from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("Warning: NeMo text processing not available")

# Evaluation metric
try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("Warning: jiwer not available - install with: pip install jiwer")

# Handle case where __name__ might not be available during direct file import
try:
    module_name = __name__
except NameError:
    module_name = "audiobench.normalizer.thai_normaliser"

logger = logging.getLogger(module_name)


@dataclass
class ThaiASRConfig:
    """Configuration for Thai ASR normalization"""
    
    # Core text processing
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    lowercase_output: bool = True
    remove_punctuation: bool = True
    
    # Thai-specific features
    tokenizer: str = "attacut"  # "attacut", "deepcut", "newmm", "longest"
    convert_thai_digits: bool = True  # ๐-๙ → 0-9
    convert_numbers_to_words: bool = False  # Keep as digits for ASR
    normalize_currency: bool = True
    normalize_phone_numbers: bool = True
    normalize_time_format: bool = True
    normalize_date_format: bool = True
    convert_buddhist_era: bool = True  # BE → CE conversion
    
    # Text cleanup
    remove_filler_words: bool = True
    remove_polite_particles: bool = False
    expand_abbreviations: bool = True
    handle_code_switching: bool = True  # Thai-English mixing
    handle_repeated_chars: bool = True
    
    # Advanced options
    use_word_segmentation: bool = False
    preserve_english_words: bool = True
    enable_validation: bool = True
    enable_debug_logging: bool = False
    strict_mode: bool = False
    max_text_length: int = 500000


class ThaiTextValidator:
    """Validate and handle Thai text encoding issues"""
    
    def __init__(self, config: ThaiASRConfig):
        self.config = config
        self._setup_thai_ranges()
    
    def _setup_thai_ranges(self):
        """Setup Thai Unicode ranges"""
        # Thai Unicode block: U+0E00 to U+0E7F
        self.thai_consonants = set('กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ')
        self.thai_vowels = set('ะัาำิีึืุูเแโใไๅ')
        self.thai_digits = set('๐๑๒๓๔๕๖๗๘๙')
        self.thai_symbols = set('ฯๆ์ํ๎')
        
        # All Thai characters
        self.all_thai_chars = (self.thai_consonants | self.thai_vowels | self.thai_digits | self.thai_symbols)
    
    def fix_encoding_issues(self, text: str) -> str:
        """Fix common Thai encoding issues"""
        if not text:
            return ""
        
        # Common encoding fixes
        encoding_fixes = {
            'à¸': '',  # Common UTF-8 corruption prefix
            'à¹': '',  # Another common corruption
            '๏ฟฝ': '',  # Replacement character
            '\ufeff': '',  # BOM
            '\u200b': '',  # Zero-width space
            '\u200c': '',  # Zero-width non-joiner
            '\u200d': '',  # Zero-width joiner
        }
        
        result = text
        for wrong, correct in encoding_fixes.items():
            result = result.replace(wrong, correct)
        
        # Ensure proper Unicode normalization (NFC for Thai)
        try:
            result = unicodedata.normalize('NFC', result)
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}")
        
        return result
    
    def validate_thai_text(self, text: str) -> bool:
        """Validate that text contains proper Thai characters"""
        if not text:
            return True
        
        if PYTHAINLP_AVAILABLE:
            # Use PyThaiNLP's isthai function
            return isthai(text, ignore_chars=" \n\t0123456789")
        
        # Fallback validation
        thai_char_count = sum(1 for c in text if c in self.all_thai_chars)
        total_chars = len([c for c in text if not c.isspace()])
        
        # Text should have at least 30% Thai characters
        return total_chars == 0 or (thai_char_count / total_chars) >= 0.3


class ThaiNumberConverter:
    """Convert Thai numbers between text and digits"""
    
    def __init__(self, config: ThaiASRConfig):
        self.config = config
        self._setup_number_mappings()
        self._compile_number_patterns()
    
    def _setup_number_mappings(self):
        """Setup Thai number word mappings"""
        # Basic digits
        self.digit_words = {
            '0': 'ศูนย์',
            '1': 'หนึ่ง',
            '2': 'สอง',
            '3': 'สาม',
            '4': 'สี่',
            '5': 'ห้า',
            '6': 'หก',
            '7': 'เจ็ด',
            '8': 'แปด',
            '9': 'เก้า',
            '10': 'สิบ'
        }
        
        # Reverse mapping - COMPREHENSIVE
        self.word_to_digit = {
            # Zero variations
            'ศูนย์': 0,
            'สูญ': 0,
            
            # Basic numbers 1-10
            'หนึ่ง': 1,
            'สอง': 2,
            'สาม': 3,
            'สี่': 4,
            'ห้า': 5,
            'หก': 6,
            'เจ็ด': 7,
            'แปด': 8,
            'เก้า': 9,
            'สิบ': 10,
            
            # Special number words
            'เอ็ด': 1,  # Used in compound numbers (11, 21, 31...)
            'ยี่': 2,   # Used in ยี่สิบ (20)
            'โหล': 12,  # Dozen
            
            # Fractions
            'ครึ่ง': 0.5,
            
            # Scale words with values
            'ร้อย': 100,
            'พัน': 1000,
            'หมื่น': 10000,
            'แสน': 100000,
            'ล้าน': 1000000,
            'สิบล้าน': 10000000,
            'ร้อยล้าน': 100000000,
            'พันล้าน': 1000000000,
        }
        
        # Scale words for parsing
        self.scale_words = {
            'สิบ': 10,
            'ร้อย': 100,
            'พัน': 1000,
            'หมื่น': 10000,
            'แสน': 100000,
            'ล้าน': 1000000,
            'สิบล้าน': 10000000,
            'ร้อยล้าน': 100000000,
            'พันล้าน': 1000000000
        }
        
        # Thai digits mapping
        self.thai_digits = {
            '๐': '0', '๑': '1', '๒': '2', '๓': '3', '๔': '4',
            '๕': '5', '๖': '6', '๗': '7', '๘': '8', '๙': '9'
        }
        
        # Reverse Thai digits
        self.arabic_to_thai = {v: k for k, v in self.thai_digits.items()}
    
    def _compile_number_patterns(self):
        """Compile patterns for Thai number word detection"""
        # Create pattern for all number words
        all_number_words = list(self.word_to_digit.keys()) + list(self.scale_words.keys())
        
        # Sort by length (longest first) to match longer compounds first
        all_number_words.sort(key=len, reverse=True)
        
        # Create regex pattern
        self.number_word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in all_number_words) + r')\b'
        )
        
        # Pattern for compound numbers like "สิบเอ็ด", "ยี่สิบ", etc.
        self.compound_pattern = re.compile(
            r'\b(ยี่สิบ|[สองสามสี่ห้าหกเจ็ดแปดเก้า]สิบ)([หนึ่งสองสามสี่ห้าหกเจ็ดแปดเก้าเอ็ด]?)\b'
        )
        
        # Pattern for numbers with scale words
        self.scale_pattern = re.compile(
            r'\b([หนึ่งสองสามสี่ห้าหกเจ็ดแปดเก้าสิบยี่สิบสามสิบสี่สิบห้าสิบหกสิบเจ็ดสิบแปดสิบเก้าสิบ]+)\s*(ร้อย|พัน|หมื่น|แสน|ล้าน)\b'
        )
        
        # Pattern for decimal numbers
        self.decimal_pattern = re.compile(
            r'\b(.+?)จุด(.+?)\b'
        )
    
    def convert_thai_digits_to_arabic(self, text: str) -> str:
        """Convert Thai digits (๐-๙) to Arabic digits (0-9) - IMPROVED VERSION"""
        if not self.config.convert_thai_digits or not text:
            return text or ""
        
        # Store original text for debugging
        original = text
        
        if PYTHAINLP_AVAILABLE:
            try:
                # First, protect decimal points between Thai digits.
                # NOTE: `re` is imported at module top — a function-local
                # `import re` here was making it a local name and triggering
                # UnboundLocalError on the fallback path (line ~348) when
                # PYTHAINLP_AVAILABLE was False.
                protected_decimals = []
                decimal_pattern = re.compile(r'([๐-๙]+)\.([๐-๙]+)')
                
                def protect_decimal(match):
                    idx = len(protected_decimals)
                    thai_before = match.group(1)
                    thai_after = match.group(2)
                    # Convert to Arabic but keep the decimal
                    arabic_before = ''.join(self.thai_digits.get(c, c) for c in thai_before)
                    arabic_after = ''.join(self.thai_digits.get(c, c) for c in thai_after)
                    protected_decimals.append(f"{arabic_before}.{arabic_after}")
                    return f"__DECIMAL_{idx}__"
                
                text = decimal_pattern.sub(protect_decimal, text)
                
                # Use PyThaiNLP for conversion
                result = thai_digit_to_arabic_digit(text)
                
                # Restore protected decimals
                for idx, decimal_value in enumerate(protected_decimals):
                    result = result.replace(f"__DECIMAL_{idx}__", decimal_value)
                
                # Remove any commas that are used as thousand separators
                # Only remove commas that are between digits
                result = re.sub(r'(\d),(\d)', r'\1\2', result)
                
                # CRITICAL: Ensure no "decimal" word is inserted
                # This might happen if PyThaiNLP converts periods weirdly
                result = re.sub(r'(\d+)\s*decimal\s*(\d+)', r'\1.\2', result)
                
                return result
                
            except Exception as e:
                logger.warning(f"PyThaiNLP digit conversion failed: {e}")
                text = original  # Reset to original
        
        # Fallback conversion with proper separator and decimal handling
        result = text
        
        # First protect decimal points
        protected_decimals = []
        decimal_pattern = re.compile(r'([๐-๙]+)\.([๐-๙]+)')
        
        def protect_decimal(match):
            idx = len(protected_decimals)
            thai_before = match.group(1)
            thai_after = match.group(2)
            # Convert to Arabic but keep the decimal
            arabic_before = ''.join(self.thai_digits.get(c, c) for c in thai_before)
            arabic_after = ''.join(self.thai_digits.get(c, c) for c in thai_after)
            protected_decimals.append(f"{arabic_before}.{arabic_after}")
            return f"__DECIMAL_{idx}__"
        
        result = decimal_pattern.sub(protect_decimal, result)
        
        # Convert Thai digits to Arabic
        for thai, arabic in self.thai_digits.items():
            result = result.replace(thai, arabic)
        
        # Restore protected decimals
        for idx, decimal_value in enumerate(protected_decimals):
            result = result.replace(f"__DECIMAL_{idx}__", decimal_value)
        
        # Remove thousand separators (commas) only between digits
        result = re.sub(r'(\d),(\d)', r'\1\2', result)
        
        # Also remove any spaces that might have been introduced between digits
        # But preserve spaces between separate numbers
        result = re.sub(r'(\d)\s+(\d)', r'\1\2', result)
        
        # CRITICAL: Final check for any "decimal" word that shouldn't be there
        result = re.sub(r'(\d+)\s*decimal\s*(\d+)', r'\1.\2', result)
        
        return result
    
    def convert_thai_words_to_numbers(self, text: str) -> str:
        """Convert Thai number words to Arabic numerals"""
        if not text:
            return ""
        
        result = text
        
        # First, handle special fraction patterns
        result = result.replace('เศษหนึ่งส่วนสอง', '0.5')
        result = result.replace('เศษหนึ่งส่วนสี่', '0.25')
        result = result.replace('เศษสามส่วนสี่', '0.75')
        
        # Create a comprehensive list of all number-related words
        all_number_words = []
        
        # Add all words from word_to_digit (excluding special punctuation)
        for word, value in self.word_to_digit.items():
            if isinstance(value, (int, float)) and word not in ['.', '%']:
                all_number_words.append(word)
        
        # Add all scale words
        all_number_words.extend(self.scale_words.keys())
        
        # Add compound patterns
        compound_patterns = [
            # Tens + ones patterns
            'สิบเอ็ด', 'สิบสอง', 'สิบสาม', 'สิบสี่', 'สิบห้า', 'สิบหก', 'สิบเจ็ด', 'สิบแปด', 'สิบเก้า',
            'ยี่สิบเอ็ด', 'ยี่สิบสอง', 'ยี่สิบสาม', 'ยี่สิบสี่', 'ยี่สิบห้า', 'ยี่สิบหก', 'ยี่สิบเจ็ด', 'ยี่สิบแปด', 'ยี่สิบเก้า',
            'สามสิบเอ็ด', 'สามสิบสอง', 'สามสิบสาม', 'สามสิบสี่', 'สามสิบห้า', 'สามสิบหก', 'สามสิบเจ็ด', 'สามสิบแปด', 'สามสิบเก้า',
            'สี่สิบเอ็ด', 'สี่สิบสอง', 'สี่สิบสาม', 'สี่สิบสี่', 'สี่สิบห้า', 'สี่สิบหก', 'สี่สิบเจ็ด', 'สี่สิบแปด', 'สี่สิบเก้า',
            'ห้าสิบเอ็ด', 'ห้าสิบสอง', 'ห้าสิบสาม', 'ห้าสิบสี่', 'ห้าสิบห้า', 'ห้าสิบหก', 'ห้าสิบเจ็ด', 'ห้าสิบแปด', 'ห้าสิบเก้า',
            'หกสิบเอ็ด', 'หกสิบสอง', 'หกสิบสาม', 'หกสิบสี่', 'หกสิบห้า', 'หกสิบหก', 'หกสิบเจ็ด', 'หกสิบแปด', 'หกสิบเก้า',
            'เจ็ดสิบเอ็ด', 'เจ็ดสิบสอง', 'เจ็ดสิบสาม', 'เจ็ดสิบสี่', 'เจ็ดสิบห้า', 'เจ็ดสิบหก', 'เจ็ดสิบเจ็ด', 'เจ็ดสิบแปด', 'เจ็ดสิบเก้า',
            'แปดสิบเอ็ด', 'แปดสิบสอง', 'แปดสิบสาม', 'แปดสิบสี่', 'แปดสิบห้า', 'แปดสิบหก', 'แปดสิบเจ็ด', 'แปดสิบแปด', 'แปดสิบเก้า',
            'เก้าสิบเอ็ด', 'เก้าสิบสอง', 'เก้าสิบสาม', 'เก้าสิบสี่', 'เก้าสิบห้า', 'เก้าสิบหก', 'เก้าสิบเจ็ด', 'เก้าสิบแปด', 'เก้าสิบเก้า',
            # Just tens
            'สามสิบ', 'สี่สิบ', 'ห้าสิบ', 'หกสิบ', 'เจ็ดสิบ', 'แปดสิบ', 'เก้าสิบ'
        ]
        all_number_words.extend(compound_patterns)
        
        # Sort by length (longest first) to match longer compounds first
        all_number_words = list(set(all_number_words))  # Remove duplicates
        all_number_words.sort(key=len, reverse=True)
        
        # Create a mapping for compound numbers
        compound_number_map = {
            # 11-19
            'สิบเอ็ด': 11, 'สิบสอง': 12, 'สิบสาม': 13, 'สิบสี่': 14, 'สิบห้า': 15,
            'สิบหก': 16, 'สิบเจ็ด': 17, 'สิบแปด': 18, 'สิบเก้า': 19,
            # 20-29
            'ยี่สิบ': 20, 'ยี่สิบเอ็ด': 21, 'ยี่สิบสอง': 22, 'ยี่สิบสาม': 23, 'ยี่สิบสี่': 24, 'ยี่สิบห้า': 25,
            'ยี่สิบหก': 26, 'ยี่สิบเจ็ด': 27, 'ยี่สิบแปด': 28, 'ยี่สิบเก้า': 29,
            # 30-39
            'สามสิบ': 30, 'สามสิบเอ็ด': 31, 'สามสิบสอง': 32, 'สามสิบสาม': 33, 'สามสิบสี่': 34, 'สามสิบห้า': 35,
            'สามสิบหก': 36, 'สามสิบเจ็ด': 37, 'สามสิบแปด': 38, 'สามสิบเก้า': 39,
            # 40-49
            'สี่สิบ': 40, 'สี่สิบเอ็ด': 41, 'สี่สิบสอง': 42, 'สี่สิบสาม': 43, 'สี่สิบสี่': 44, 'สี่สิบห้า': 45,
            'สี่สิบหก': 46, 'สี่สิบเจ็ด': 47, 'สี่สิบแปด': 48, 'สี่สิบเก้า': 49,
            # 50-59
            'ห้าสิบ': 50, 'ห้าสิบเอ็ด': 51, 'ห้าสิบสอง': 52, 'ห้าสิบสาม': 53, 'ห้าสิบสี่': 54, 'ห้าสิบห้า': 55,
            'ห้าสิบหก': 56, 'ห้าสิบเจ็ด': 57, 'ห้าสิบแปด': 58, 'ห้าสิบเก้า': 59,
            # 60-69
            'หกสิบ': 60, 'หกสิบเอ็ด': 61, 'หกสิบสอง': 62, 'หกสิบสาม': 63, 'หกสิบสี่': 64, 'หกสิบห้า': 65,
            'หกสิบหก': 66, 'หกสิบเจ็ด': 67, 'หกสิบแปด': 68, 'หกสิบเก้า': 69,
            # 70-79
            'เจ็ดสิบ': 70, 'เจ็ดสิบเอ็ด': 71, 'เจ็ดสิบสอง': 72, 'เจ็ดสิบสาม': 73, 'เจ็ดสิบสี่': 74, 'เจ็ดสิบห้า': 75,
            'เจ็ดสิบหก': 76, 'เจ็ดสิบเจ็ด': 77, 'เจ็ดสิบแปด': 78, 'เจ็ดสิบเก้า': 79,
            # 80-89
            'แปดสิบ': 80, 'แปดสิบเอ็ด': 81, 'แปดสิบสอง': 82, 'แปดสิบสาม': 83, 'แปดสิบสี่': 84, 'แปดสิบห้า': 85,
            'แปดสิบหก': 86, 'แปดสิบเจ็ด': 87, 'แปดสิบแปด': 88, 'แปดสิบเก้า': 89,
            # 90-99
            'เก้าสิบ': 90, 'เก้าสิบเอ็ด': 91, 'เก้าสิบสอง': 92, 'เก้าสิบสาม': 93, 'เก้าสิบสี่': 94, 'เก้าสิบห้า': 95,
            'เก้าสิบหก': 96, 'เก้าสิบเจ็ด': 97, 'เก้าสิบแปด': 98, 'เก้าสิบเก้า': 99,
        }
        
        # Process the text multiple times to handle nested conversions
        for _ in range(5):  # Multiple passes to handle complex cases
            old_result = result
            
            # Step 1: Handle decimal patterns (e.g., "สามจุดหนึ่งสี่" → "3.14")
            # FIXED: More comprehensive decimal pattern that handles compound numbers
            decimal_patterns = [
                # Pattern for compound numbers with จุด
                (re.compile(r'(' + '|'.join(re.escape(w) for w in compound_number_map.keys()) + r')จุด(' + '|'.join(re.escape(w) for w in compound_number_map.keys()) + r')'), 
                lambda m: f"{compound_number_map.get(m.group(1), m.group(1))}.{compound_number_map.get(m.group(2), m.group(2))}"),
                
                # Pattern for simple number words with จุด
                (re.compile(r'([^\s]+)จุด([^\s]+)'), None),  # Will be handled below
            ]
            
            # Apply compound decimal patterns first
            for pattern, converter in decimal_patterns[:1]:  # Only the compound pattern
                result = pattern.sub(converter, result)
            
            # Handle general decimal pattern
            decimal_pattern = r'([^\s]+)จุด([^\s]+)'
            decimal_matches = list(re.finditer(decimal_pattern, result))
            
            for match in reversed(decimal_matches):
                integer_part = match.group(1)
                decimal_part = match.group(2)
                
                # Convert integer part
                integer_value = None
                if integer_part in compound_number_map:
                    integer_value = compound_number_map[integer_part]
                elif integer_part in self.word_to_digit:
                    value = self.word_to_digit[integer_part]
                    if isinstance(value, (int, float)):
                        integer_value = value
                else:
                    # Try to parse it as a complex number
                    parsed = self._parse_complex_thai_number(integer_part)
                    if parsed is not None:
                        integer_value = parsed
                
                if integer_value is not None:
                    # Convert decimal part digit by digit
                    decimal_digits = []
                    remaining = decimal_part
                    
                    # Try to extract individual digit words
                    while remaining:
                        found = False
                        for word in ['ศูนย์', 'หนึ่ง', 'สอง', 'สาม', 'สี่', 'ห้า', 'หก', 'เจ็ด', 'แปด', 'เก้า']:
                            if remaining.startswith(word):
                                digit = self.word_to_digit[word]
                                decimal_digits.append(str(digit))
                                remaining = remaining[len(word):]
                                found = True
                                break
                        
                        # Also check for compound numbers in decimal part
                        if not found and remaining in compound_number_map:
                            decimal_digits.append(str(compound_number_map[remaining]))
                            remaining = ""
                            found = True
                        
                        if not found:
                            break
                    
                    if decimal_digits and not remaining:
                        replacement = f"{integer_value}.{''.join(decimal_digits)}"
                        start, end = match.span()
                        result = result[:start] + replacement + result[end:]
            
            # Step 2: Convert complex number expressions
            # Find all positions of number words
            number_word_positions = []
            for word in all_number_words:
                pos = 0
                while True:
                    pos = result.find(word, pos)
                    if pos == -1:
                        break
                    number_word_positions.append((pos, pos + len(word), word))
                    pos += 1
            
            # Sort by position
            number_word_positions.sort()
            
            # Merge overlapping or adjacent number words into groups
            if number_word_positions:
                groups = []
                current_group = [number_word_positions[0]]
                
                for i in range(1, len(number_word_positions)):
                    pos_start, pos_end, word = number_word_positions[i]
                    last_end = current_group[-1][1]
                    
                    # If this word starts at or before the last word ends, they overlap or are adjacent
                    if pos_start <= last_end:
                        current_group.append(number_word_positions[i])
                    else:
                        # Start a new group
                        groups.append(current_group)
                        current_group = [number_word_positions[i]]
                
                groups.append(current_group)
                
                # Process each group from right to left
                for group in reversed(groups):
                    if not group:
                        continue
                    
                    # Get the full text span of this group
                    group_start = group[0][0]
                    group_end = max(item[1] for item in group)
                    group_text = result[group_start:group_end]
                    
                    # CRITICAL FIX: Check what comes before and after the number
                    # to determine if we need to add spaces
                    char_before = result[group_start-1] if group_start > 0 else ''
                    char_after = result[group_end] if group_end < len(result) else ''
                    
                    # Check if the character before is Thai (not a space)
                    need_space_before = char_before and self._is_thai_char(char_before)
                    # Check if the character after is Thai (not a space)
                    need_space_after = char_after and self._is_thai_char(char_after)
                    
                    # Try to parse this as a number
                    parsed_value = self._parse_complex_thai_number(group_text)
                    if parsed_value is not None:
                        # Build replacement with appropriate spacing
                        replacement = str(parsed_value)
                        
                        # Add space before if needed
                        if need_space_before:
                            replacement = ' ' + replacement
                        
                        # Add space after if needed
                        if need_space_after:
                            replacement = replacement + ' '
                        
                        # Adjust indices based on whether we're adding a space before
                        start_idx = group_start - (1 if need_space_before and char_before == ' ' else 0)
                        end_idx = group_end + (1 if need_space_after and char_after == ' ' else 0)
                        
                        result = result[:start_idx] + replacement + result[end_idx:]
            
            # Step 3: Convert remaining simple numbers
            # FIXED: Special handling for "ครึ่ง" (half) - must be 0.5 not 0
            result = re.sub(r'\bครึ่ง\b', '0.5', result)
            
            # Use a more careful approach that handles word boundaries better
            for word in all_number_words:
                if word == 'ครึ่ง':  # Already handled above
                    continue
                    
                if word in compound_number_map:
                    value = compound_number_map[word]
                elif word in self.word_to_digit:
                    value = self.word_to_digit[word]
                    if not isinstance(value, (int, float)):
                        continue
                else:
                    continue
                
                # Replace the word with its numeric value
                # Be careful about word boundaries and preserve spaces
                pos = 0
                while True:
                    pos = result.find(word, pos)
                    if pos == -1:
                        break
                    
                    # Check if this is a complete word (not part of a larger word)
                    before_ok = pos == 0 or not self._is_thai_char(result[pos-1])
                    after_ok = pos + len(word) >= len(result) or not self._is_thai_char(result[pos + len(word)])
                    
                    if before_ok and after_ok:
                        # FIXED: Check what comes before and after to determine spacing
                        char_before = result[pos-1] if pos > 0 else ''
                        char_after = result[pos + len(word)] if pos + len(word) < len(result) else ''
                        
                        # Check if we need to add spaces
                        need_space_before = char_before and self._is_thai_char(char_before) and char_before != ' '
                        need_space_after = char_after and self._is_thai_char(char_after) and char_after != ' '
                        
                        replacement = str(value)
                        
                        # Add spaces as needed
                        if need_space_before:
                            replacement = ' ' + replacement
                        if need_space_after:
                            replacement = replacement + ' '
                        
                        # Calculate actual replacement positions
                        start_idx = pos
                        end_idx = pos + len(word)
                        
                        result = result[:start_idx] + replacement + result[end_idx:]
                        pos += len(replacement)
                    else:
                        pos += 1
            
            # If no changes were made, we're done
            if result == old_result:
                break
        
        # Clean up any double spaces that might have been created
        result = re.sub(r'  +', ' ', result)
        
        return result
    
    def _is_thai_char(self, char: str) -> bool:
        """Check if a character is a Thai character"""
        return '\u0E00' <= char <= '\u0E7F'

    def _parse_thai_number_text(self, text: str) -> Optional[int]:
        """Parse complex Thai number text to integer value"""
        if not text:
            return None
        
        # Remove spaces and normalize
        text = text.strip()
        
        # Check if it's a simple number word
        if text in self.word_to_digit:
            value = self.word_to_digit[text]
            if isinstance(value, (int, float)):
                return int(value)
        
        # Parse complex numbers
        total = 0
        current_number = 0
        
        words = self._split_thai_number_words(text)
        
        i = 0
        while i < len(words):
            word = words[i]
            
            if word in self.word_to_digit:
                value = self.word_to_digit[word]
                if isinstance(value, int):
                    if value >= 100:  # It's a scale word
                        if current_number == 0:
                            current_number = 1
                        current_number *= value
                        total += current_number
                        current_number = 0
                    else:
                        current_number = current_number * 10 + value if current_number > 0 else value
            elif word in self.scale_words:
                scale = self.scale_words[word]
                if current_number == 0:
                    current_number = 1
                current_number *= scale
                total += current_number
                current_number = 0
            
            i += 1
        
        total += current_number
        return total if total > 0 else None
    
    def _parse_complex_thai_number(self, text: str) -> Optional[int]:
        """Parse complex Thai number text to integer value - COMPLETE REWRITE"""
        if not text:
            return None
        
        # Remove any spaces
        text = text.strip()
        
        # Check if it's a simple compound number first
        compound_number_map = {
            # 11-19
            'สิบเอ็ด': 11, 'สิบสอง': 12, 'สิบสาม': 13, 'สิบสี่': 14, 'สิบห้า': 15,
            'สิบหก': 16, 'สิบเจ็ด': 17, 'สิบแปด': 18, 'สิบเก้า': 19,
            # 20-29
            'ยี่สิบ': 20, 'ยี่สิบเอ็ด': 21, 'ยี่สิบสอง': 22, 'ยี่สิบสาม': 23, 'ยี่สิบสี่': 24, 'ยี่สิบห้า': 25,
            'ยี่สิบหก': 26, 'ยี่สิบเจ็ด': 27, 'ยี่สิบแปด': 28, 'ยี่สิบเก้า': 29,
            # 30-99
            'สามสิบ': 30, 'สามสิบเอ็ด': 31, 'สามสิบสอง': 32, 'สามสิบสาม': 33, 'สามสิบสี่': 34, 'สามสิบห้า': 35,
            'สามสิบหก': 36, 'สามสิบเจ็ด': 37, 'สามสิบแปด': 38, 'สามสิบเก้า': 39,
            'สี่สิบ': 40, 'สี่สิบเอ็ด': 41, 'สี่สิบสอง': 42, 'สี่สิบสาม': 43, 'สี่สิบสี่': 44, 'สี่สิบห้า': 45,
            'สี่สิบหก': 46, 'สี่สิบเจ็ด': 47, 'สี่สิบแปด': 48, 'สี่สิบเก้า': 49,
            'ห้าสิบ': 50, 'ห้าสิบเอ็ด': 51, 'ห้าสิบสอง': 52, 'ห้าสิบสาม': 53, 'ห้าสิบสี่': 54, 'ห้าสิบห้า': 55,
            'ห้าสิบหก': 56, 'ห้าสิบเจ็ด': 57, 'ห้าสิบแปด': 58, 'ห้าสิบเก้า': 59,
            'หกสิบ': 60, 'หกสิบเอ็ด': 61, 'หกสิบสอง': 62, 'หกสิบสาม': 63, 'หกสิบสี่': 64, 'หกสิบห้า': 65,
            'หกสิบหก': 66, 'หกสิบเจ็ด': 67, 'หกสิบแปด': 68, 'หกสิบเก้า': 69,
            'เจ็ดสิบ': 70, 'เจ็ดสิบเอ็ด': 71, 'เจ็ดสิบสอง': 72, 'เจ็ดสิบสาม': 73, 'เจ็ดสิบสี่': 74, 'เจ็ดสิบห้า': 75,
            'เจ็ดสิบหก': 76, 'เจ็ดสิบเจ็ด': 77, 'เจ็ดสิบแปด': 78, 'เจ็ดสิบเก้า': 79,
            'แปดสิบ': 80, 'แปดสิบเอ็ด': 81, 'แปดสิบสอง': 82, 'แปดสิบสาม': 83, 'แปดสิบสี่': 84, 'แปดสิบห้า': 85,
            'แปดสิบหก': 86, 'แปดสิบเจ็ด': 87, 'แปดสิบแปด': 88, 'แปดสิบเก้า': 89,
            'เก้าสิบ': 90, 'เก้าสิบเอ็ด': 91, 'เก้าสิบสอง': 92, 'เก้าสิบสาม': 93, 'เก้าสิบสี่': 94, 'เก้าสิบห้า': 95,
            'เก้าสิบหก': 96, 'เก้าสิบเจ็ด': 97, 'เก้าสิบแปด': 98, 'เก้าสิบเก้า': 99,
        }
        
        if text in compound_number_map:
            return compound_number_map[text]
        
        # Check if it's a simple number word
        if text in self.word_to_digit:
            value = self.word_to_digit[text]
            if isinstance(value, (int, float)):
                return int(value)
        
        # Now parse complex numbers by breaking them down
        # Thai numbers are structured as: [millions] [hundred-thousands] [ten-thousands] [thousands] [hundreds] [tens-ones]
        
        total = 0
        remaining = text
        
        # Process from largest to smallest scale
        scales = [
            ('ล้าน', 1000000),
            ('แสน', 100000),
            ('หมื่น', 10000),
            ('พัน', 1000),
            ('ร้อย', 100)
        ]
        
        for scale_word, scale_value in scales:
            if scale_word in remaining:
                # Find the position of the scale word
                pos = remaining.find(scale_word)
                
                # Everything before this scale word is the multiplier
                multiplier_text = remaining[:pos]
                
                if multiplier_text:
                    # Parse the multiplier
                    multiplier = None
                    
                    # Check compound numbers first
                    if multiplier_text in compound_number_map:
                        multiplier = compound_number_map[multiplier_text]
                    # Then check simple numbers
                    elif multiplier_text in self.word_to_digit:
                        value = self.word_to_digit[multiplier_text]
                        if isinstance(value, (int, float)):
                            multiplier = int(value)
                    # Try to parse recursively
                    else:
                        multiplier = self._parse_complex_thai_number(multiplier_text)
                    
                    if multiplier is not None:
                        total += multiplier * scale_value
                    else:
                        # If we can't parse the multiplier, assume it's 1
                        total += scale_value
                else:
                    # No multiplier means 1
                    total += scale_value
                
                # Continue with the remaining text
                remaining = remaining[pos + len(scale_word):]
        
        # Handle any remaining text (should be < 100)
        if remaining:
            if remaining in compound_number_map:
                total += compound_number_map[remaining]
            elif remaining in self.word_to_digit:
                value = self.word_to_digit[remaining]
                if isinstance(value, (int, float)):
                    total += int(value)
            else:
                # Try to parse as tens + ones
                # Check for patterns like "สิบ" followed by a digit
                if 'สิบ' in remaining:
                    pos = remaining.find('สิบ')
                    before = remaining[:pos]
                    after = remaining[pos + len('สิบ'):]
                    
                    tens = 0
                    ones = 0
                    
                    # Parse tens
                    if before == 'ยี่':
                        tens = 20
                    elif before in self.word_to_digit:
                        value = self.word_to_digit[before]
                        if isinstance(value, int) and 2 <= value <= 9:
                            tens = value * 10
                    elif not before:
                        tens = 10
                    
                    # Parse ones
                    if after == 'เอ็ด' and tens > 10:
                        ones = 1
                    elif after in self.word_to_digit:
                        value = self.word_to_digit[after]
                        if isinstance(value, int) and 1 <= value <= 9:
                            ones = value
                    
                    if tens > 0:
                        total += tens + ones
                        remaining = ""
                
                # If there's still remaining text, try to parse it as a simple number
                if remaining and remaining in self.word_to_digit:
                    value = self.word_to_digit[remaining]
                    if isinstance(value, (int, float)):
                        total += int(value)
        
        return total if total > 0 else None

    def _split_thai_number_words(self, text: str) -> List[str]:
        """Split Thai text into number words"""
        words = []
        current_word = ""
        
        # List of all valid Thai number words
        valid_words = set(self.word_to_digit.keys()) | set(self.scale_words.keys())
        
        # Try to find the longest matching words
        i = 0
        while i < len(text):
            # Try to match the longest possible word starting at position i
            longest_match = ""
            longest_match_len = 0
            
            for length in range(min(10, len(text) - i), 0, -1):
                candidate = text[i:i+length]
                if candidate in valid_words:
                    longest_match = candidate
                    longest_match_len = length
                    break
            
            if longest_match:
                words.append(longest_match)
                i += longest_match_len
            else:
                # Skip this character
                i += 1
        
        return words
    
    def convert_arabic_digits_to_thai(self, text: str) -> str:
        """Convert Arabic digits to Thai digits"""
        if PYTHAINLP_AVAILABLE:
            try:
                return arabic_digit_to_thai_digit(text)
            except Exception as e:
                logger.warning(f"PyThaiNLP digit conversion failed: {e}")
        
        # Fallback conversion
        result = text
        for arabic, thai in self.arabic_to_thai.items():
            result = result.replace(arabic, thai)
        
        return result
    
    def number_to_words(self, number: Union[int, str]) -> str:
        """Convert number to Thai words - should not be called when convert_numbers_to_words is False"""
        # If we're not supposed to convert numbers to words (ASR mode), just return the number
        if not self.config.convert_numbers_to_words:
            return str(number)
        
        try:
            if isinstance(number, str):
                # Handle decimal numbers
                if '.' in number or ',' in number:
                    return self._convert_decimal(number)
                number = int(number)
            
            if number == 0:
                return "ศูนย์"
            
            if number < 0:
                return "ลบ " + self.number_to_words(-number)
            
            # Build number words
            parts = []
            
            # Process each scale
            if number >= 1000000:
                millions = number // 1000000
                parts.append(self._convert_below_million(millions) + " ล้าน")
                number %= 1000000
            
            if number >= 100000:
                hundred_thousands = number // 100000
                parts.append(self.digit_words[str(hundred_thousands)] + " แสน")
                number %= 100000
            
            if number >= 10000:
                ten_thousands = number // 10000
                parts.append(self.digit_words[str(ten_thousands)] + " หมื่น")
                number %= 10000
            
            if number >= 1000:
                thousands = number // 1000
                parts.append(self.digit_words[str(thousands)] + " พัน")
                number %= 1000
            
            if number >= 100:
                hundreds = number // 100
                parts.append(self.digit_words[str(hundreds)] + " ร้อย")
                number %= 100
            
            if number >= 20:
                tens = number // 10
                if tens == 2:
                    parts.append("ยี่สิบ")  # Special case for 20
                else:
                    parts.append(self.digit_words[str(tens)] + "สิบ")
                number %= 10
            
            elif number >= 11:
                parts.append("สิบ")
                number %= 10
            
            elif number == 10:
                parts.append("สิบ")
                number = 0
            
            # Handle ones place
            if number > 0:
                if len(parts) > 0 and number == 1:
                    # Use เอ็ด for 1 in compound numbers
                    parts.append("เอ็ด")
                else:
                    parts.append(self.digit_words[str(number)])
            
            return "".join(parts)
            
        except Exception as e:
            logger.warning(f"Number conversion failed for {number}: {e}")
            return str(number)
    
    def _convert_below_million(self, number: int) -> str:
        """Helper to convert numbers below million"""
        if number < 10:
            return self.digit_words[str(number)]
        else:
            return self.number_to_words(number)
    
    def _convert_decimal(self, number_str: str) -> str:
        """Convert decimal numbers to Thai words"""
        # Thai uses period as decimal separator
        parts = number_str.split('.')
        if len(parts) == 2:
            integer_part = self.number_to_words(int(parts[0]))
            
            # Read decimal digits individually
            decimal_digits = []
            for digit in parts[1]:
                if digit.isdigit():
                    decimal_digits.append(self.digit_words[digit])
            
            if decimal_digits:
                return f"{integer_part}จุด{''.join(decimal_digits)}"
            else:
                return integer_part
        
        return self.number_to_words(int(parts[0]))


class ThaiWordSegmenter:
    """Handle Thai word segmentation"""
    
    def __init__(self, config: ThaiASRConfig):
        self.config = config
        self.tokenizer = config.tokenizer
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """Initialize the selected tokenizer"""
        self.tokenize_func = None
        
        if self.tokenizer == "attacut" and ATTACUT_AVAILABLE:
            self.tokenize_func = attacut_tokenize
            logger.info("Using AttaCut tokenizer")
        elif self.tokenizer == "deepcut":
            deepcut_module = _check_deepcut_availability()
            if deepcut_module:
                self.tokenize_func = deepcut_module.tokenize
                logger.info("Using DeepCut tokenizer")
            else:
                logger.warning("DeepCut not available, falling back to PyThaiNLP")
                if PYTHAINLP_AVAILABLE:
                    self.tokenize_func = self._pythainlp_tokenize
                else:
                    self.tokenize_func = self._simple_tokenize
        elif PYTHAINLP_AVAILABLE:
            # Fallback to PyThaiNLP tokenizer
            self.tokenize_func = lambda text: word_tokenize(text, engine=self.tokenizer)
            logger.info(f"Using PyThaiNLP {self.tokenizer} tokenizer")
        else:
            # PyThaiNLP not installed; logged once at module load time.
            # Per-call WARNING was emitting tens of thousands of lines during
            # large-batch normalization (e.g. 11k-sample English ASR files
            # routed through Thai for some reason), so we drop the per-call
            # warning. Module-level Warning prints handle this case already.
            logger.debug("No tokenizer available, word segmentation disabled")
    
    def segment(self, text: str) -> str:
        """Perform word segmentation on Thai text"""
        if not self.config.use_word_segmentation or not text:
            return text
        
        if not self.tokenize_func:
            return text
        
        try:
            # Tokenize the text
            tokens = self.tokenize_func(text)
            
            # Join with spaces
            # Note: This might need adjustment based on ASR requirements
            return ' '.join(tokens)
            
        except Exception as e:
            logger.warning(f"Word segmentation failed: {e}")
            return text


class ThaiCurrencyProcessor:
    """Process Thai currency formats"""
    
    def __init__(self, config: ThaiASRConfig):
        self.config = config
        self.number_converter = ThaiNumberConverter(config)
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile currency patterns"""
        # Note: Order matters! More specific patterns should come first
        self.patterns = [
            # สลึง conversion (must come before other patterns)
            (re.compile(r'(\d+)\s*สลึง', re.IGNORECASE),
            self._convert_salueng),
            
            # FIXED: Add pattern for สตางค์ without space
            (re.compile(r'(\d+)สตางค์', re.IGNORECASE),
            lambda m: f"{m.group(1)} สตางค์"),
            
            # Existing pattern for สตางค์ with space
            (re.compile(r'(\d+)\s+สตางค์', re.IGNORECASE),
            lambda m: f"{m.group(1)} สตางค์"),
            
            # Baht symbol variations - ENHANCED
            # ฿ at the beginning
            (re.compile(r'฿\s*(\d+(?:[,.]?\d{3})*(?:\.\d{1,2})?)', re.IGNORECASE),
            self._convert_baht_symbol),
            
            # ฿ at the end (like "1000฿")
            (re.compile(r'(\d+(?:[,.]?\d{3})*(?:\.\d{1,2})?)\s*฿', re.IGNORECASE),
            self._convert_baht_symbol_suffix),
            
            # บาท/baht/THB variations
            (re.compile(r'(\d+(?:[,.]?\d{3})*(?:\.\d{1,2})?)\s*(?:บาท|baht|THB)', re.IGNORECASE),
            self._convert_baht_suffix),
            
            # Abbreviated บ. (with period)
            (re.compile(r'(\d+(?:[,.]?\d{3})*(?:\.\d{1,2})?)\s*บ\.', re.IGNORECASE),
            self._convert_baht_abbreviated),
            
            # USD - Multiple formats
            (re.compile(r'\$\s*(\d+(?:[,.]?\d+)?)', re.IGNORECASE),
            self._convert_usd),
            (re.compile(r'USD\s*(\d+(?:[,.]?\d+)?)', re.IGNORECASE),
            self._convert_usd_prefix),
            
            # EUR - Multiple formats
            (re.compile(r'€\s*(\d+(?:[,.]?\d+)?)', re.IGNORECASE),
            self._convert_eur),
            (re.compile(r'EUR\s*(\d+(?:[,.]?\d+)?)', re.IGNORECASE),
            self._convert_eur_prefix),
            
            # JPY - Multiple formats
            (re.compile(r'¥\s*(\d+(?:[,.]?\d+)?)', re.IGNORECASE),
            self._convert_jpy),
            (re.compile(r'JPY\s*(\d+(?:[,.]?\d+)?)', re.IGNORECASE),
            self._convert_jpy_prefix),
            
            # GBP - Multiple formats
            (re.compile(r'£\s*(\d+(?:[,.]?\d+)?)', re.IGNORECASE),
            self._convert_gbp),
            (re.compile(r'GBP\s*(\d+(?:[,.]?\d+)?)', re.IGNORECASE),
            self._convert_gbp_prefix),
            
            # Common price patterns
            (re.compile(r'ราคา\s*(\d+)\s*บาท', re.IGNORECASE),
            lambda m: f"ราคา {m.group(1)} บาท"),
        ]
    
    def _clean_amount(self, amount_str: str) -> str:
        """Clean amount string by removing thousand separators"""
        return amount_str.replace(',', '')
    
    def _convert_salueng(self, match):
        """Convert สลึง to สตางค์ (1 สลึง = 25 สตางค์)"""
        amount_str = match.group(1)
        try:
            amount = int(amount_str)
            satang = amount * 25
            
            # If it equals 100 สตางค์ or more, we could convert to บาท
            # but based on the test cases, we keep it as สตางค์
            return f"{satang} สตางค์"
        except (ValueError, TypeError):
            # If conversion fails, return original
            return match.group(0)
    
    def _convert_baht_symbol(self, match):
        """Convert ฿ symbol amounts (symbol at beginning)"""
        amount_str = match.group(1)
        clean_amount = self._clean_amount(amount_str)
        return f"{clean_amount} บาท"
    
    def _convert_baht_symbol_suffix(self, match):
        """Convert amounts with ฿ symbol at end"""
        amount_str = match.group(1)
        clean_amount = self._clean_amount(amount_str)
        return f"{clean_amount} บาท"
    
    def _convert_baht_suffix(self, match):
        """Convert amounts with baht suffix"""
        amount_str = match.group(1)
        clean_amount = self._clean_amount(amount_str)
        return f"{clean_amount} บาท"
    
    def _convert_baht_abbreviated(self, match):
        """Convert amounts with abbreviated บ."""
        amount_str = match.group(1)
        clean_amount = self._clean_amount(amount_str)
        return f"{clean_amount} บาท"
    
    def _convert_usd(self, match):
        """Convert USD amounts with $ symbol"""
        amount = self._clean_amount(match.group(1))
        return f"{amount} ดอลลาร์"
    
    def _convert_usd_prefix(self, match):
        """Convert USD amounts with USD prefix"""
        amount = self._clean_amount(match.group(1))
        return f"{amount} ดอลลาร์"
    
    def _convert_eur(self, match):
        """Convert EUR amounts with € symbol"""
        amount = self._clean_amount(match.group(1))
        return f"{amount} ยูโร"
    
    def _convert_eur_prefix(self, match):
        """Convert EUR amounts with EUR prefix"""
        amount = self._clean_amount(match.group(1))
        return f"{amount} ยูโร"
    
    def _convert_jpy(self, match):
        """Convert JPY amounts with ¥ symbol"""
        amount = self._clean_amount(match.group(1))
        return f"{amount} เยน"
    
    def _convert_jpy_prefix(self, match):
        """Convert JPY amounts with JPY prefix"""
        amount = self._clean_amount(match.group(1))
        return f"{amount} เยน"
    
    def _convert_gbp(self, match):
        """Convert GBP amounts with £ symbol"""
        amount = self._clean_amount(match.group(1))
        return f"{amount} ปอนด์"
    
    def _convert_gbp_prefix(self, match):
        """Convert GBP amounts with GBP prefix"""
        amount = self._clean_amount(match.group(1))
        return f"{amount} ปอนด์"
    
    def normalize_currency(self, text: str) -> str:
        """Normalize currency in text"""
        if not self.config.normalize_currency or not text:
            return text or ""
        
        result = text
        
        try:
            # Apply patterns in order
            for pattern, converter in self.patterns:
                result = pattern.sub(converter, result)
            
            if self.config.enable_debug_logging and text != result:
                logger.debug(f"Currency normalization: '{text}' → '{result}'")
            
        except Exception as e:
            logger.warning(f"Currency normalization failed: {e}")
            return text
        
        return result


class ThaiPhoneProcessor:
    """Process Thai phone number formats"""
    def __init__(self, config: ThaiASRConfig):
        self.config = config
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile phone number patterns"""
        # Thai phone prefixes
        self.mobile_prefixes = ['06', '08', '09']
        self.landline_prefixes = ['02', '03', '04', '05', '07']
        
        # Compile patterns - MORE SPECIFIC TO AVOID FALSE MATCHES
        self.patterns = [
            # International format FIRST (highest priority)
            # FIXED: Handle +662-xxx-xxxx where 2 is the area code
            (re.compile(r'\+66([2-57])-(\d{3})-(\d{4})\b'), self._convert_international_landline_with_dash),
            (re.compile(r'\+66([2-57])\s*(\d{3})\s*(\d{4})\b'), self._convert_international_landline_with_space),
            (re.compile(r'\+66([2-57])(\d{3})(\d{4})\b'), self._convert_international_landline_continuous),
            
            # Mobile international formats
            (re.compile(r'\+66([689])-(\d{3})-(\d{4})\b'), self._convert_international_mobile_with_dash),
            (re.compile(r'\+66([689])\s*(\d{3})\s*(\d{4})\b'), self._convert_international_mobile_with_space),
            (re.compile(r'\+66([689])(\d{3})(\d{4})\b'), self._convert_international_mobile_continuous),
            
            # Standard international formats with separator
            (re.compile(r'\+66\s*(\d{9})\b'), self._convert_international),
            (re.compile(r'\+66[\s-]([689])[\s-](\d{4})[\s-](\d{4})\b'), self._convert_international_formatted_mobile),
            (re.compile(r'\+66[\s-]([2-57])[\s-](\d{3})[\s-](\d{4})\b'), self._convert_international_formatted_landline),
            (re.compile(r'\+66[\s-](\d{1,2})[\s-](\d{3})[\s-](\d{4})\b'), self._convert_international_formatted),
            
            # Very long international numbers starting with 0066
            (re.compile(r'\b(0066)([689]\d{8})\b'), self._convert_long_international_mobile),
            (re.compile(r'\b(0066)([2-57]\d{7})\b'), self._convert_long_international_landline),
            
            # Mobile numbers (10 digits) - must start with 06, 08, or 09
            (re.compile(r'\b(0[689]\d{8})\b'), self._convert_mobile),
            (re.compile(r'\b(0[689]\d)[\s-](\d{3})[\s-](\d{4})\b'), self._convert_mobile_formatted),
            (re.compile(r'\b(0[689]\d)[\s-](\d{4})[\s-](\d{3})\b'), self._convert_mobile_formatted_alt),
            (re.compile(r'\b(0[689]\d)\.(\d{3})\.(\d{4})\b'), self._convert_mobile_dotted),
            (re.compile(r'\(\s*(0[689]\d)\s*\)\s*(\d{3})[\s-](\d{4})\b'), self._convert_mobile_parentheses),
            
            # Landline (9 digits) - must start with 02-05 or 07
            (re.compile(r'\b(0[2-57]\d{7})\b'), self._convert_landline),
            (re.compile(r'\b(0[2-57])[\s-](\d{3})[\s-](\d{4})\b'), self._convert_landline_formatted),
            (re.compile(r'\b(0[2-57]\d)[\s-](\d{6})\b'), self._convert_landline_formatted_alt),
        ]
    
    def _is_phone_number(self, number_str: str) -> bool:
        """Check if a number string is likely a phone number"""
        # Remove any formatting
        digits_only = re.sub(r'[^\d]', '', number_str)
        
        # Check length
        if len(digits_only) not in [9, 10, 11, 13]:
            return False
        
        # Check prefixes
        if len(digits_only) == 10:
            return digits_only.startswith(('06', '08', '09'))
        elif len(digits_only) == 9:
            return digits_only.startswith(('02', '03', '04', '05', '07'))
        elif len(digits_only) == 11:
            return digits_only.startswith('668') or digits_only.startswith('662')
        elif len(digits_only) == 13:
            return digits_only.startswith('00668') or digits_only.startswith('00662')
        
        return False
    
    def _digits_to_thai_phone(self, digits: str, include_plus: bool = False) -> str:
        """Convert phone digits to Thai words for phone reading"""
        # In phone numbers, digits are read individually
        # IMPORTANT: Keep tone marks for proper Thai pronunciation
        digit_map = {
            '0': 'ศูนย์', '1': 'หนึ่ง', '2': 'สอง', '3': 'สาม', '4': 'สี่',
            '5': 'ห้า', '6': 'หก', '7': 'เจ็ด', '8': 'แปด', '9': 'เก้า'
        }
        
        words = []
        
        # Add 'บวก' for plus sign if needed
        if include_plus:
            words.append('บวก')
        
        for digit in digits:
            if digit.isdigit():
                words.append(digit_map[digit])
        
        return ' '.join(words)
    
    def _convert_mobile(self, match):
        """Convert mobile number"""
        return self._digits_to_thai_phone(match.group(1))
    
    def _convert_mobile_formatted(self, match):
        """Convert formatted mobile number"""
        all_digits = match.group(1) + match.group(2) + match.group(3)
        return self._digits_to_thai_phone(all_digits)
    
    def _convert_mobile_formatted_alt(self, match):
        """Convert alternatively formatted mobile number"""
        all_digits = match.group(1) + match.group(2) + match.group(3)
        return self._digits_to_thai_phone(all_digits)
    
    def _convert_mobile_dotted(self, match):
        """Convert dot-formatted mobile number"""
        all_digits = match.group(1) + match.group(2) + match.group(3)
        return self._digits_to_thai_phone(all_digits)
    
    def _convert_mobile_parentheses(self, match):
        """Convert parentheses-formatted mobile number"""
        all_digits = match.group(1) + match.group(2) + match.group(3)
        return self._digits_to_thai_phone(all_digits)
    
    def _convert_landline(self, match):
        """Convert landline number"""
        return self._digits_to_thai_phone(match.group(1))
    
    def _convert_landline_formatted(self, match):
        """Convert formatted landline number"""
        all_digits = match.group(1) + match.group(2) + match.group(3)
        return self._digits_to_thai_phone(all_digits)
    
    def _convert_landline_formatted_alt(self, match):
        """Convert alternatively formatted landline number"""
        all_digits = match.group(1) + match.group(2)
        return self._digits_to_thai_phone(all_digits)
    
    def _convert_international(self, match):
        """Convert international format - FIXED to include country code"""
        digits = '66' + match.group(1)  # Include country code
        return self._digits_to_thai_phone(digits, include_plus=True)
    
    def _convert_international_formatted(self, match):
        """Convert formatted international number - FIXED"""
        # For landline format: +66 2 123 4567
        digits = '66' + match.group(1) + match.group(2) + match.group(3)
        return self._digits_to_thai_phone(digits, include_plus=True)
    
    def _convert_international_formatted_mobile(self, match):
        """Convert formatted international mobile number - FIXED"""
        # For mobile format: +66 8 1234 5678
        digits = '66' + match.group(1) + match.group(2) + match.group(3)
        return self._digits_to_thai_phone(digits, include_plus=True)
    
    def _convert_international_formatted_landline(self, match):
        """Convert formatted international landline number - FIXED"""
        # For landline format: +66 2 123 4567 or +66-2-123-4567
        digits = '66' + match.group(1) + match.group(2) + match.group(3)
        return self._digits_to_thai_phone(digits, include_plus=True)
    
    # NEW METHODS for better handling of specific formats
    def _convert_international_landline_with_dash(self, match):
        """Convert international landline format with dashes: +662-123-4567 - FIXED"""
        area_code = match.group(1)  # '2'
        part1 = match.group(2)      # '123'
        part2 = match.group(3)      # '4567'
        # Include country code 66
        digits = '66' + area_code + part1 + part2  # '6621234567'
        return self._digits_to_thai_phone(digits, include_plus=True)
    
    def _convert_international_landline_with_space(self, match):
        """Convert international landline format with spaces: +662 123 4567 - FIXED"""
        area_code = match.group(1)
        part1 = match.group(2)
        part2 = match.group(3)
        digits = '66' + area_code + part1 + part2
        return self._digits_to_thai_phone(digits, include_plus=True)
    
    def _convert_international_landline_continuous(self, match):
        """Convert international landline format continuous: +6621234567 - FIXED"""
        area_code = match.group(1)
        part1 = match.group(2)
        part2 = match.group(3)
        digits = '66' + area_code + part1 + part2
        return self._digits_to_thai_phone(digits, include_plus=True)
    
    def _convert_international_mobile_with_dash(self, match):
        """Convert international mobile format with dashes: +668-123-4567 - FIXED"""
        first_digit = match.group(1)  # '8'
        part1 = match.group(2)        # '123'
        part2 = match.group(3)        # '4567'
        digits = '66' + first_digit + part1 + part2  # '6681234567'
        return self._digits_to_thai_phone(digits, include_plus=True)
    
    def _convert_international_mobile_with_space(self, match):
        """Convert international mobile format with spaces: +668 123 4567 - FIXED"""
        first_digit = match.group(1)
        part1 = match.group(2)
        part2 = match.group(3)
        digits = '66' + first_digit + part1 + part2
        return self._digits_to_thai_phone(digits, include_plus=True)
    
    def _convert_international_mobile_continuous(self, match):
        """Convert international mobile format continuous: +6681234567 - FIXED"""
        first_digit = match.group(1)
        part1 = match.group(2)
        part2 = match.group(3)
        digits = '66' + first_digit + part1 + part2
        return self._digits_to_thai_phone(digits, include_plus=True)
    
    def _convert_long_international_mobile(self, match):
        """Convert long international format for mobile: 0066812345678 - NEW"""
        country_prefix = match.group(1)  # '0066'
        local_number = match.group(2)    # '812345678'
        # Convert all digits including the 0066
        all_digits = country_prefix + local_number
        return self._digits_to_thai_phone(all_digits)
    
    def _convert_long_international_landline(self, match):
        """Convert long international format for landline: 006621234567 - NEW"""
        country_prefix = match.group(1)  # '0066'
        local_number = match.group(2)    # '21234567'
        # Convert all digits including the 0066
        all_digits = country_prefix + local_number
        return self._digits_to_thai_phone(all_digits)
    
    def normalize_phones(self, text: str) -> str:
        """Normalize phone numbers in text"""
        if not self.config.normalize_phone_numbers or not text:
            return text or ""
        
        result = text
        
        try:
            # Apply patterns in order (international format has priority)
            for pattern, converter in self.patterns:
                result = pattern.sub(converter, result)
            
            if self.config.enable_debug_logging and text != result:
                logger.debug(f"Phone normalization: '{text}' → '{result}'")
            
        except Exception as e:
            logger.warning(f"Phone normalization failed: {e}")
            return text
        
        return result


class ThaiDateTimeProcessor:
    """Process Thai date and time formats - GENERALIZED VERSION"""
    
    def __init__(self, config: ThaiASRConfig):
        self.config = config
        self.number_converter = ThaiNumberConverter(config)
        self._setup_mappings()
        self._compile_patterns()
    
    def _setup_mappings(self):
        """Setup comprehensive mappings for Thai time expressions"""
        # Thai number words for time
        self.thai_numbers = {
            'ศูนย์': 0, 'หนึ่ง': 1, 'สอง': 2, 'สาม': 3, 'สี่': 4,
            'ห้า': 5, 'หก': 6, 'เจ็ด': 7, 'แปด': 8, 'เก้า': 9,
            'สิบ': 10, 'สิบเอ็ด': 11, 'สิบสอง': 12, 'สิบสาม': 13,
            'สิบสี่': 14, 'สิบห้า': 15, 'สิบหก': 16, 'สิบเจ็ด': 17,
            'สิบแปด': 18, 'สิบเก้า': 19, 'ยี่สิบ': 20, 'ยี่สิบเอ็ด': 21,
            'ยี่สิบสอง': 22, 'ยี่สิบสาม': 23, 'ยี่สิบสี่': 24,
            # Add more as needed
        }
        
        # Month names (comprehensive) - FIXED: Use full names consistently
        self.month_names = {
            'มกราคม': 1, 'กุมภาพันธ์': 2, 'มีนาคม': 3, 'เมษายน': 4,
            'พฤษภาคม': 5, 'มิถุนายน': 6, 'กรกฎาคม': 7, 'สิงหาคม': 8,
            'กันยายน': 9, 'ตุลาคม': 10, 'พฤศจิกายน': 11, 'ธันวาคม': 12,
            # Short forms
            'ม.ค.': 1, 'ก.พ.': 2, 'มี.ค.': 3, 'เม.ย.': 4,
            'พ.ค.': 5, 'มิ.ย.': 6, 'ก.ค.': 7, 'ส.ค.': 8,
            'ก.ย.': 9, 'ต.ค.': 10, 'พ.ย.': 11, 'ธ.ค.': 12,
            # Colloquial forms
            'มกรา': 1, 'กุมภา': 2, 'มีนา': 3, 'เมษา': 4,
            'พฤษภา': 5, 'มิถุนา': 6, 'กรกฎา': 7, 'สิงหา': 8,
            'กันยา': 9, 'ตุลา': 10, 'พฤศจิกา': 11, 'ธันวา': 12
        }
        
        # Reverse mapping - FIXED: Use full month names only
        self.month_numbers_to_names = {
            1: 'มกราคม', 2: 'กุมภาพันธ์', 3: 'มีนาคม', 4: 'เมษายน',
            5: 'พฤษภาคม', 6: 'มิถุนายน', 7: 'กรกฎาคม', 8: 'สิงหาคม',
            9: 'กันยายน', 10: 'ตุลาคม', 11: 'พฤศจิกายน', 12: 'ธันวาคม'
        }
        
        # Day names (full and abbreviated)
        self.day_names = {
            'จันทร์': 'จันทร์', 'อังคาร': 'อังคาร', 'พุธ': 'พุธ',
            'พฤหัสบดี': 'พฤหัสบดี', 'พฤหัส': 'พฤหัสบดี',
            'ศุกร์': 'ศุกร์', 'เสาร์': 'เสาร์', 'อาทิตย์': 'อาทิตย์',
            # Abbreviations
            'จ.': 'จันทร์', 'อ.': 'อังคาร', 'พ.': 'พุธ',
            'พฤ.': 'พฤหัสบดี', 'ศ.': 'ศุกร์', 'ส.': 'เสาร์', 'อา.': 'อาทิตย์'
        }
    
    def _convert_date_written_simple_with_year_handling(self, match):
        """Convert simple written date format with 2 or 4 digit year support"""
        day = int(match.group(1))
        month_str = match.group(2)
        year_str = match.group(3)
        year = int(year_str)
        
        # Handle 2-digit year (assume 2500s for Buddhist Era)
        if year < 100:
            year = 2500 + year
        
        # Check if วันที่ already exists before this match
        full_text = match.string
        match_start = match.start()
        
        # Look back up to 10 characters before the match
        lookback_start = max(0, match_start - 10)
        prefix_text = full_text[lookback_start:match_start].strip()
        
        # Normalize month name
        month_name = month_str
        if month_str in self.month_names:
            month_num = self.month_names[month_str]
            month_name = self.month_numbers_to_names.get(month_num, month_str)
        
        # If วันที่ already exists in the prefix, don't add it again
        if 'วันที่' in prefix_text:
            return f"{day} {month_name} {year}"
        else:
            return f"วันที่ {day} {month_name} {year}"

    def _compile_patterns(self):
        """Compile date/time patterns - FIXED VERSION WITH BETTER DATE HANDLING"""
        # Build dynamic patterns for Thai numbers
        thai_num_pattern = '|'.join(self.thai_numbers.keys())
        
        self.patterns = [
            # Generic ตี pattern (1-12 AM)
            (re.compile(rf'ตี({thai_num_pattern})'), self._convert_tee_time),
            
            # Special time words
            (re.compile(r'เที่ยงคืน'), lambda m: '0 นาฬิกา'),
            (re.compile(r'เที่ยงวัน'), lambda m: '12 นาฬิกา'),
            (re.compile(r'เที่ยง(?!คืน|วัน)'), lambda m: '12 นาฬิกา'),  # Just เที่ยง
            
            # Generic ทุ่ม pattern with Thai words
            (re.compile(rf'({thai_num_pattern})\s*ทุ่ม(?:ครึ่ง)?'), self._convert_tum_time_word),
            (re.compile(r'(\d{1,2})\s*ทุ่ม(?:ครึ่ง)?'), self._convert_tum_time),
            (re.compile(r'([๐-๙]{1,2})\s*ทุ่ม(?:ครึ่ง)?'), self._convert_tum_time_thai),
            
            # Generic โมงเช้า pattern
            (re.compile(rf'({thai_num_pattern})\s*โมง\s*เช้า'), self._convert_morning_time_word),
            (re.compile(r'(\d{1,2})\s*โมง\s*เช้า'), self._convert_morning_time),
            (re.compile(r'([๐-๙]{1,2})\s*โมง\s*เช้า'), self._convert_morning_time_thai),
            
            # Generic บ่าย pattern
            (re.compile(r'บ่าย\s*โมง(?:ตรง)?'), lambda m: '13 นาฬิกา'),
            (re.compile(rf'บ่าย\s*({thai_num_pattern})\s*(?:โมง)?'), self._convert_afternoon_time_word),
            (re.compile(r'บ่าย\s*(\d{1,2})\s*(?:โมง)?'), self._convert_afternoon_time),
            (re.compile(r'บ่าย\s*([๐-๙]{1,2})\s*(?:โมง)?'), self._convert_afternoon_time_thai),
            
            # Generic โมงเย็น pattern
            (re.compile(rf'({thai_num_pattern})\s*โมง\s*เย็น'), self._convert_evening_time_word),
            (re.compile(r'(\d{1,2})\s*โมง\s*เย็น'), self._convert_evening_time),
            (re.compile(r'([๐-๙]{1,2})\s*โมง\s*เย็น'), self._convert_evening_time_thai),
            
            # Time with colon format
            (re.compile(r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*น\.?'), self._convert_time_with_suffix),
            (re.compile(r'([๐-๙]{1,2}):([๐-๙]{2})(?::([๐-๙]{2}))?\s*น\.?'), self._convert_time_with_suffix_thai),
            (re.compile(r'(\d{1,2}):(\d{2})(?::(\d{2}))?'), self._convert_time),
            (re.compile(r'([๐-๙]{1,2}):([๐-๙]{2})(?::([๐-๙]{2}))?'), self._convert_time_thai),
            
            # Time with Thai format
            (re.compile(rf'({thai_num_pattern})\s*(?:นาฬิกา|โมง)\s*({thai_num_pattern})?\s*(?:นาที)?(?!.*วินาที)', re.IGNORECASE),
            self._convert_time_written_word),
            (re.compile(r'(\d{1,2})\s*(?:นาฬิกา|โมง)\s*(\d{1,2})?\s*(?:นาที)?(?!.*วินาที)', re.IGNORECASE),
            self._convert_time_written),
            (re.compile(r'([๐-๙]{1,2})\s*(?:นาฬิกา|โมง)\s*([๐-๙]{1,2})?\s*(?:นาที)?(?!.*วินาที)', re.IGNORECASE),
            self._convert_time_written_thai),
            
            # FIXED: Add patterns for simple date formats WITHOUT year BEFORE other date patterns
            # Pattern for Thai digits + month name/abbreviation (no year)
            (re.compile(rf'([๐-๙]{{1,2}})\s+({"|".join(self.month_names.keys())})(?!\s+\d{{2,4}})', re.IGNORECASE),
            self._convert_date_simple_thai_no_year),
            
            # Pattern for Arabic digits + month name/abbreviation (no year)
            (re.compile(rf'(\d{{1,2}})\s+({"|".join(self.month_names.keys())})(?!\s+\d{{2,4}})', re.IGNORECASE),
            self._convert_date_simple_no_year),
            
            # Date patterns
            (re.compile(r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})'), self._convert_date),
            (re.compile(r'([๐-๙]{1,2})[/\-\.]([๐-๙]{1,2})[/\-\.]([๐-๙]{4})'), self._convert_date_thai),
            
            # Written date with month names (generalized)
            (re.compile(rf'วันที่\s*(\d{{1,2}})\s*(?:เดือน)?\s*({"|".join(self.month_names.keys())})\s*(?:พ\.ศ\.|ปี)?\s*(\d{{4}})?', re.IGNORECASE),
            self._convert_date_written),
            
            # Date with day and month (with year)
            (re.compile(rf'(\d{{1,2}})\s+({"|".join(self.month_names.keys())})\s+(\d{{2,4}})', re.IGNORECASE),
            self._convert_date_written_simple_with_year_handling),
            
            # Day of week (generalized)
            (re.compile(rf'วัน({"|".join(self.day_names.keys())})', re.IGNORECASE),
            self._normalize_weekday),
            
            # Relative dates
            (re.compile(r'วันนี้'), lambda m: 'วันนี้'),
            (re.compile(r'พรุ่งนี้'), lambda m: 'พรุ่งนี้'),
            (re.compile(r'มะรืน(?:นี้)?'), lambda m: 'มะรืนนี้'),
            (re.compile(r'เมื่อวาน(?:นี้)?'), lambda m: 'เมื่อวานนี้'),
            
            # Time periods
            (re.compile(r'ตอนเช้า'), lambda m: 'ตอนเช้า'),
            (re.compile(r'ตอนเย็น'), lambda m: 'ตอนเย็น'),
            (re.compile(r'ตอนกลางวัน'), lambda m: 'ตอนกลางวัน'),
            (re.compile(r'ตอนกลางคืน'), lambda m: 'ตอนกลางคืน'),
            (re.compile(r'ช่วงเช้า'), lambda m: 'ช่วงเช้า'),
            (re.compile(r'ช่วงบ่าย'), lambda m: 'ช่วงบ่าย'),
            (re.compile(r'ช่วงเย็น'), lambda m: 'ช่วงเย็น'),
            (re.compile(r'ช่วงดึก'), lambda m: 'ช่วงดึก'),
        ]
    
    def _convert_buddhist_to_gregorian(self, year: int) -> int:
        """Convert Buddhist Era year to Gregorian"""
        if self.config.convert_buddhist_era and year > 2400:
            return year - 543
        return year
    
    def _convert_thai_digits_to_arabic(self, text: str) -> str:
        """Convert Thai digits to Arabic digits"""
        thai_digits = {
            '๐': '0', '๑': '1', '๒': '2', '๓': '3', '๔': '4',
            '๕': '5', '๖': '6', '๗': '7', '๘': '8', '๙': '9'
        }
        result = text
        for thai, arabic in thai_digits.items():
            result = result.replace(thai, arabic)
        return result
    
    def _convert_thai_word_to_number(self, word: str) -> int:
        """Convert Thai number word to integer"""
        return self.thai_numbers.get(word, -1)
    
    def _convert_tee_time(self, match):
        """Convert ตี time expressions (1-12 AM)"""
        time_word = match.group(1)
        hour = self._convert_thai_word_to_number(time_word)
        if hour > 0:
            return f"{hour} นาฬิกา"
        return match.group(0)
    
    def _convert_tum_time_word(self, match):
        """Convert ทุ่ม time with Thai words"""
        text = match.group(0)
        time_word = match.group(1)
        hour = self._convert_thai_word_to_number(time_word)
        
        if hour > 0:
            # ทุ่ม counts from 6pm: 1ทุ่ม = 19:00, 2ทุ่ม = 20:00, etc.
            actual_hour = 18 + hour
            if actual_hour >= 24:
                actual_hour -= 24
            
            # Check for ครึ่ง (half)
            if 'ครึ่ง' in text or 'ครึง' in text:
                return f"{actual_hour} นาฬิกา 30 นาที"
            else:
                return f"{actual_hour} นาฬิกา"
        
        return match.group(0)
    
    def _convert_tum_time(self, match):
        """Convert ทุ่ม time (Thai evening hours)"""
        text = match.group(0)
        hour_str = match.group(1)
        hour = int(hour_str)
        
        # ทุ่ม counts from 6pm: 1ทุ่ม = 19:00, 2ทุ่ม = 20:00, etc.
        actual_hour = 18 + hour
        if actual_hour >= 24:
            actual_hour -= 24
        
        # Check for ครึ่ง (half)
        if 'ครึ่ง' in text or 'ครึง' in text:
            return f"{actual_hour} นาฬิกา 30 นาที"
        else:
            return f"{actual_hour} นาฬิกา"
    
    def _convert_tum_time_thai(self, match):
        """Convert ทุ่ม time with Thai digits"""
        text = match.group(0)
        hour_str = self._convert_thai_digits_to_arabic(match.group(1))
        hour = int(hour_str)
        
        actual_hour = 18 + hour
        if actual_hour >= 24:
            actual_hour -= 24
        
        if 'ครึ่ง' in text:
            return f"{actual_hour} นาฬิกา 30 นาที"
        else:
            return f"{actual_hour} นาฬิกา"
    
    def _convert_morning_time_word(self, match):
        """Convert morning time with Thai words"""
        time_word = match.group(1)
        hour = self._convert_thai_word_to_number(time_word)
        if hour > 0:
            return f"{hour} นาฬิกา"
        return match.group(0)
    
    def _convert_morning_time(self, match):
        """Convert morning time (Xโมงเช้า)"""
        hour = int(match.group(1))
        return f"{hour} นาฬิกา"
    
    def _convert_morning_time_thai(self, match):
        """Convert morning time with Thai digits"""
        hour = int(self._convert_thai_digits_to_arabic(match.group(1)))
        return f"{hour} นาฬิกา"
    
    def _convert_afternoon_time_word(self, match):
        """Convert afternoon time with Thai words"""
        time_word = match.group(1)
        hour = self._convert_thai_word_to_number(time_word)
        if hour > 0:
            # บ่าย 1 = 13:00, บ่าย 2 = 14:00, etc.
            if hour <= 6:
                hour += 12
            return f"{hour} นาฬิกา"
        return match.group(0)
    
    def _convert_afternoon_time(self, match):
        """Convert afternoon time (บ่ายX)"""
        hour = int(match.group(1))
        # บ่าย 1 = 13:00, บ่าย 2 = 14:00, etc.
        if hour <= 6:
            hour += 12
        return f"{hour} นาฬิกา"
    
    def _convert_afternoon_time_thai(self, match):
        """Convert afternoon time with Thai digits"""
        hour = int(self._convert_thai_digits_to_arabic(match.group(1)))
        if hour <= 6:
            hour += 12
        return f"{hour} นาฬิกา"
    
    def _convert_evening_time_word(self, match):
        """Convert evening time with Thai words"""
        time_word = match.group(1)
        hour = self._convert_thai_word_to_number(time_word)
        if hour > 0:
            # Evening time typically refers to 4pm-6pm
            if hour <= 6:
                hour += 12
            return f"{hour} นาฬิกา"
        return match.group(0)
    
    def _convert_evening_time(self, match):
        """Convert evening time (Xโมงเย็น)"""
        hour = int(match.group(1))
        # Evening time typically refers to 4pm-6pm
        if hour <= 6:
            hour += 12
        return f"{hour} นาฬิกา"
    
    def _convert_evening_time_thai(self, match):
        """Convert evening time with Thai digits"""
        hour = int(self._convert_thai_digits_to_arabic(match.group(1)))
        if hour <= 6:
            hour += 12
        return f"{hour} นาฬิกา"
    
    def _convert_date_simple_no_year(self, match):
        """Convert simple date format without year (15 มกราคม)"""
        day = int(match.group(1))
        month_str = match.group(2)
        
        # Normalize month name
        month_name = month_str
        if month_str in self.month_names:
            month_num = self.month_names[month_str]
            month_name = self.month_numbers_to_names.get(month_num, month_str)
        
        return f"วันที่ {day} {month_name}"

    def _convert_date_simple_thai_no_year(self, match):
        """Convert simple date format with Thai digits without year (๑ มกรา)"""
        day_thai = match.group(1)
        month_str = match.group(2)
        
        # Convert Thai digits to Arabic
        day = int(self._convert_thai_digits_to_arabic(day_thai))
        
        # Normalize month name
        month_name = month_str
        if month_str in self.month_names:
            month_num = self.month_names[month_str]
            month_name = self.month_numbers_to_names.get(month_num, month_str)
        
        return f"วันที่ {day} {month_name}"

    def _convert_time(self, match):
        """Convert digital time format HH:MM[:SS]"""
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = int(match.group(3)) if match.group(3) else None
        
        result = f"{hour} นาฬิกา"
        
        # Always include minutes when seconds are present
        if second is not None:
            # When there are seconds, always show minutes (even if 0)
            result += f" {minute} นาที"
            if second > 0:
                result += f" {second} วินาที"
        else:
            # When no seconds, only show minutes if > 0
            if minute > 0:
                result += f" {minute} นาที"
        
        return result
    
    def _convert_time_thai(self, match):
        """Convert Thai digital time format"""
        hour = int(self._convert_thai_digits_to_arabic(match.group(1)))
        minute = int(self._convert_thai_digits_to_arabic(match.group(2)))
        second = int(self._convert_thai_digits_to_arabic(match.group(3))) if match.group(3) else None
        
        result = f"{hour} นาฬิกา"
        
        # Always include minutes when seconds are present
        if second is not None:
            # When there are seconds, always show minutes (even if 0)
            result += f" {minute} นาที"
            if second > 0:
                result += f" {second} วินาที"
        else:
            # When no seconds, only show minutes if > 0
            if minute > 0:
                result += f" {minute} นาที"
        
        return result
    
    def _convert_time_with_suffix(self, match):
        """Convert time with น. suffix"""
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = int(match.group(3)) if match.group(3) else None
        
        result = f"{hour} นาฬิกา"
        
        # Always include minutes when seconds are present
        if second is not None:
            # When there are seconds, always show minutes (even if 0)
            result += f" {minute} นาที"
            if second > 0:
                result += f" {second} วินาที"
        else:
            # When no seconds, only show minutes if > 0
            if minute > 0:
                result += f" {minute} นาที"
        
        return result
    
    def _convert_time_with_suffix_thai(self, match):
        """Convert Thai time with น. suffix"""
        hour = int(self._convert_thai_digits_to_arabic(match.group(1)))
        minute = int(self._convert_thai_digits_to_arabic(match.group(2)))
        second = int(self._convert_thai_digits_to_arabic(match.group(3))) if match.group(3) else None
        
        result = f"{hour} นาฬิกา"
        
        # Always include minutes when seconds are present
        if second is not None:
            # When there are seconds, always show minutes (even if 0)
            result += f" {minute} นาที"
            if second > 0:
                result += f" {second} วินาที"
        else:
            # When no seconds, only show minutes if > 0
            if minute > 0:
                result += f" {minute} นาที"
        
        return result
    
    def _convert_time_written_word(self, match):
        """Convert Thai written time format with Thai words"""
        hour_word = match.group(1)
        minute_word = match.group(2) if match.group(2) else None
        
        hour = self._convert_thai_word_to_number(hour_word)
        if hour < 0:
            return match.group(0)
        
        result = f"{hour} นาฬิกา"
        
        if minute_word:
            minute = self._convert_thai_word_to_number(minute_word)
            if minute >= 0:
                result += f" {minute} นาที"
        
        return result
        
    def _convert_time_written(self, match):
        """Convert Thai written time format"""
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        
        result = f"{hour} นาฬิกา"
        if minute > 0:
            result += f" {minute} นาที"
        
        return result
    
    def _convert_time_written_thai(self, match):
        """Convert Thai written time format with Thai digits"""
        hour = int(self._convert_thai_digits_to_arabic(match.group(1)))
        minute = int(self._convert_thai_digits_to_arabic(match.group(2))) if match.group(2) else 0
        
        result = f"{hour} นาฬิกา"
        if minute > 0:
            result += f" {minute} นาที"
        
        return result
    
    def _convert_date(self, match):
        """Convert date with slashes/dots/dashes"""
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))

        # Get month name
        month_name = self.month_numbers_to_names.get(month, str(month))
        
        # Return without "เดือน" and "ปี" to match expected format
        return f"วันที่ {day} {month_name} {year}"
    
    def _convert_date_thai(self, match):
        """Convert date with Thai digits"""
        day = int(self._convert_thai_digits_to_arabic(match.group(1)))
        month = int(self._convert_thai_digits_to_arabic(match.group(2)))
        year = int(self._convert_thai_digits_to_arabic(match.group(3)))
        
        # Get month name
        month_name = self.month_numbers_to_names.get(month, str(month))
        
        return f"วันที่ {day} {month_name} {year}"
    
    def _convert_date_written(self, match):
        """Convert written date format"""
        day = int(match.group(1))
        month_str = match.group(2)
        year = int(match.group(3)) if match.group(3) else None
        
        # Normalize month name
        month_name = month_str
        if month_str in self.month_names:
            month_num = self.month_names[month_str]
            month_name = self.month_numbers_to_names.get(month_num, month_str)
        
        # Build result without "เดือน" and "ปี"
        if year:
            # DON'T convert Buddhist Era - keep the original year
            # year = self._convert_buddhist_to_gregorian(year)
            result = f"วันที่ {day} {month_name} {year}"
        else:
            result = f"วันที่ {day} {month_name}"
        
        return result
    
    def _convert_date_written_simple(self, match):
        """Convert simple written date format (15 มกราคม 2567) - FIXED to avoid double วันที่"""
        day = int(match.group(1))
        month_str = match.group(2)
        year = int(match.group(3))
        
        # Check if วันที่ already exists before this match
        full_text = match.string
        match_start = match.start()
        
        # Look back up to 10 characters before the match
        lookback_start = max(0, match_start - 10)
        prefix_text = full_text[lookback_start:match_start].strip()
        
        # Normalize month name
        month_name = month_str
        if month_str in self.month_names:
            month_num = self.month_names[month_str]
            month_name = self.month_numbers_to_names.get(month_num, month_str)
        
        # If วันที่ already exists in the prefix, don't add it again
        if 'วันที่' in prefix_text:
            return f"{day} {month_name} {year}"
        else:
            return f"วันที่ {day} {month_name} {year}"
    
    def _normalize_weekday(self, match):
        """Normalize weekday spelling"""
        day = match.group(1)
        normalized = self.day_names.get(day, day)
        return f"วัน{normalized}"
    
    def normalize_datetime(self, text: str) -> str:
        """Normalize date and time in text"""
        if not (self.config.normalize_date_format or self.config.normalize_time_format) or not text:
            return text or ""
        
        result = text
        
        try:
            # Keep track of what's been processed to avoid overlapping replacements
            processed_spans = []
            
            # First pass: collect all matches with their patterns
            all_matches = []
            for pattern_idx, (pattern, converter) in enumerate(self.patterns):
                for match in pattern.finditer(result):
                    all_matches.append((match.start(), match.end(), match, pattern_idx, converter))
            
            # Sort matches by start position and length (longer matches first)
            all_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
            
            # Process matches, skipping overlapping ones
            offset = 0
            for start, end, match, pattern_idx, converter in all_matches:
                # Check if this match overlaps with any already processed span
                overlaps = False
                for proc_start, proc_end in processed_spans:
                    if not (end <= proc_start or start >= proc_end):
                        overlaps = True
                        break
                
                if overlaps:
                    continue
                
                # Apply the converter
                old_text = match.group(0)
                new_text = converter(match) if callable(converter) else converter
                
                # Special handling for date patterns to avoid duplication
                if 'วันที่' in new_text and pattern_idx in [26, 30]:  # Patterns that add วันที่
                    # Check if วันที่ already exists before this match in the original text
                    text_before = result[:start]
                    # Look for วันที่ within 10 characters before
                    lookback_start = max(0, start - 10)
                    recent_text = result[lookback_start:start].strip()
                    if 'วันที่' in recent_text:
                        # Don't add วันที่ again, just use the conversion without it
                        new_text = new_text.replace('วันที่ ', '')
                
                # Calculate the actual position with offset
                actual_start = start + offset
                actual_end = end + offset
                
                # Replace in result
                result = result[:actual_start] + new_text + result[actual_end:]
                
                # Update offset for future replacements
                offset += len(new_text) - len(old_text)
                
                # Mark this span as processed (using original positions)
                processed_spans.append((start, end))
                
                if self.config.enable_debug_logging:
                    logger.debug(f"DateTime pattern {pattern_idx} applied: '{old_text}' → '{new_text}'")
            
            if self.config.enable_debug_logging and text != result:
                logger.debug(f"DateTime normalization: '{text}' → '{result}'")
            
        except Exception as e:
            logger.warning(f"DateTime normalization failed: {e}")
            return text
        
        return result

# Keep all meaningful words and only remove repetitions and hesitation sounds
class ThaiFillerProcessor:
    """Process Thai filler words and polite particles"""
    
    def __init__(self, config: ThaiASRConfig):
        self.config = config
        self._setup_filler_lists()
    
    def _setup_filler_lists(self):
        """Setup Thai filler words and particles"""
        # Common filler words
        self.filler_words = [
        ]
        
        # Polite particles (gender/formality markers)
        self.polite_particles = []
        
        # Hesitation sounds
        self.hesitation_sounds = [
            'เอ่อ', 'อ่า', 'อืม', 'อือ', 'เอิ่ม', 'อ้อ', 'อ๋อ'
        ]
        
        # Repeated patterns
        self.repeated_patterns = [
            (r'\b(แบบ)\s+\1+\b', r'\1'),  # แบบ แบบ → แบบ
            (r'\b(คือ)\s+\1+\b', r'\1'),   # คือ คือ → คือ
            (r'\b(ว่า)\s+\1+\b', r'\1'),   # ว่า ว่า → ว่า
        ]
    
    def remove_fillers(self, text: str) -> str:
        """Remove filler words from text"""
        if not self.config.remove_filler_words or not text:
            return text or ""
        
        result = text
        
        try:
            # Remove repeated patterns
            for pattern, replacement in self.repeated_patterns:
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            
            # Remove hesitation sounds
            for sound in self.hesitation_sounds:
                pattern = r'\b' + re.escape(sound) + r'\b'
                result = re.sub(pattern, ' ', result, flags=re.IGNORECASE)
            
            # Remove polite particles if configured
            if self.config.remove_polite_particles:
                for particle in self.polite_particles:
                    # Remove at end of sentences or utterances
                    patterns = [
                        rf'{re.escape(particle)}$',  # End of text
                        rf'{re.escape(particle)}\s*[,.]',  # Before punctuation
                        rf'{re.escape(particle)}\s+',  # Before space
                    ]
                    
                    for pattern in patterns:
                        result = re.sub(pattern, ' ', result, flags=re.IGNORECASE)
            
            # Remove filler words (careful with context)
            for filler in self.filler_words:
                # Only remove if it's truly a filler (standalone)
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


class ThaiAbbreviationExpander:
    """Expand Thai abbreviations"""
    
    def __init__(self, config: ThaiASRConfig):
        self.config = config
        self._setup_abbreviations()
    
    def _setup_abbreviations(self):
        """Setup common Thai abbreviations - SAFE VERSION"""
        self.abbreviations = {
            
            # Locations (safe)
            'กทม.': 'กรุงเทพมหานคร',     # Bangkok
            
            # Titles (safe multi-character)
            'น.ส.': 'นางสาว',           # Miss
            'ดร.': 'ดอกเตอร์',          # Doctor
            'ศ.ดร.': 'ศาสตราจารย์ดอกเตอร์',  # Professor Dr.
            'รศ.': 'รองศาสตราจารย์',     # Associate Professor
            'ผศ.': 'ผู้ช่วยศาสตราจารย์',   # Assistant Professor
            'พ.ต.': 'พันตำรวจ',         # Police Colonel
            'พ.ต.อ.': 'พันตำรวจเอก',    # Police Lt. Colonel (FIXED)
            'พ.อ.': 'พันเอก',           # Colonel
            'พล.อ.': 'พลเอก',           # General
            'พล.ต.': 'พลตำรวจ',         # Police General
            'ร.ต.': 'ร้อยตำรวจ',         # Police Captain
            'ร.อ.': 'ร้อยเอก',          # Captain
            
            # Units (only safe multi-character ones)
            'กม.': 'กิโลเมตร',          # Kilometer
            'ซม.': 'เซนติเมตร',         # Centimeter
            'มม.': 'มิลลิเมตร',         # Millimeter
            'กก.': 'กิโลกรัม',          # Kilogram
            'มก.': 'มิลลิกรัม',         # Milligram
            'มล.': 'มิลลิลิตร',         # Milliliter
            'ตร.ม.': 'ตารางเมตร',       # Square meter
            'ตร.กม.': 'ตารางกิโลเมตร',   # Square kilometer
            'ลบ.ม.': 'ลูกบาศก์เมตร',     # Cubic meter
            
            # Time and dates
            'พ.ศ.': 'พุทธศักราช',        # Buddhist Era
            'ค.ศ.': 'คริสต์ศักราช',       # Christian Era
            
            # ADDED: Month abbreviations
            'ม.ค.': 'มกราคม',
            'ก.พ.': 'กุมภาพันธ์',
            'มี.ค.': 'มีนาคม',
            'เม.ย.': 'เมษายน',
            'พ.ค.': 'พฤษภาคม',
            'มิ.ย.': 'มิถุนายน',
            'ก.ค.': 'กรกฎาคม',
            'ส.ค.': 'สิงหาคม',
            'ก.ย.': 'กันยายน',
            'ต.ค.': 'ตุลาคม',
            'พ.ย.': 'พฤศจิกายน',
            'ธ.ค.': 'ธันวาคม',
            
            # Common multi-character abbreviations
            'ฯลฯ': 'และอื่น ๆ',          # Etc.
            'อาทิ': 'อาทิเช่น',          # For example
            'เช่น': 'เช่น',             # Such as (keep as is)
            
            # Government/Official
            'สนง.': 'สำนักงาน',          # Office
            'กระทรวง': 'กระทรวง',        # Ministry (keep as is)
            'ส.ส.': 'สมาชิกสภาผู้แทนราษฎร',  # Member of Parliament
            'ส.ว.': 'สมาชิกวุฒิสภา',      # Senator
            'อบต.': 'องค์การบริหารส่วนตำบล', # Subdistrict Administrative Organization
            'อบจ.': 'องค์การบริหารส่วนจังหวัด', # Provincial Administrative Organization
            
            # Single character units (ADDED)
            'ม.': 'เมตร',              # Meter
            'ก.': 'กรัม',              # Gram
            'ล.': 'ลิตร',              # Liter
            
            # Road/Location
            'ถ.': 'ถนน',               # Road
            'ซ.': 'ซอย',               # Soi/Lane
            'หมู่': 'หมู่ที่',           # Village number
        }
        
        # Regex patterns for complex abbreviations - FIXED VERSION
        self.complex_patterns = [
            # Special pattern for Bangkok
            (re.compile(r'กทม\.', re.IGNORECASE), 'กรุงเทพมหานคร'),
            
            # University pattern - more flexible
            (re.compile(r'ม\.\s*(\d+)'), r'มหาวิทยาลัย \1'),
            
            # School pattern - more flexible
            (re.compile(r'รร\.(?=\s*[ก-ฮ])'), r'โรงเรียน'),
            
            # Location patterns - FIXED to add space after expansion
            (re.compile(r'จ\.(?=\s*[ก-ฮ])'), r'จังหวัด '),
            (re.compile(r'อ\.(?=\s*[ก-ฮ])'), r'อำเภอ '),
            (re.compile(r'ต\.(?=\s*[ก-ฮ])'), r'ตำบล '),
        ]
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand abbreviations in text - FIXED VERSION"""
        if not self.config.expand_abbreviations or not text:
            return text or ""
        
        result = text
        
        try:
            # CRITICAL: First, protect full words that might be mistaken for abbreviations
            protected_words = {
                'วันอาทิตย์': '__PROTECTED_SUNDAY__',
                'อาทิตย์': '__PROTECTED_SUNDAY_SHORT__',
                'อาทิเช่น': '__PROTECTED_FOR_EXAMPLE__',
            }
            
            for word, placeholder in protected_words.items():
                result = result.replace(word, placeholder)
            
            # CRITICAL: Handle title abbreviations FIRST before any location patterns
            # This prevents พ.ต.อ. from being corrupted by location patterns
            title_patterns = [
                # Multi-part titles first (longest match first)
                (re.compile(r'\bพ\.ต\.อ\.'), 'พันตำรวจเอก'),
                (re.compile(r'\bพล\.อ\.'), 'พลเอก'),
                (re.compile(r'\bพล\.ต\.'), 'พลตำรวจ'),
                (re.compile(r'\bศ\.ดร\.'), 'ศาสตราจารย์ดอกเตอร์'),
                (re.compile(r'\bร\.ต\.'), 'ร้อยตำรวจ'),
                (re.compile(r'\bร\.อ\.'), 'ร้อยเอก'),
                (re.compile(r'\bพ\.อ\.'), 'พันเอก'),
                (re.compile(r'\bพ\.ต\.'), 'พันตำรวจ'),
                
                # Single-part titles
                (re.compile(r'\bน\.ส\.'), 'นางสาว'),
                (re.compile(r'\bดร\.'), 'ดอกเตอร์'),
                (re.compile(r'\bรศ\.'), 'รองศาสตราจารย์'),
                (re.compile(r'\bผศ\.'), 'ผู้ช่วยศาสตราจารย์'),
            ]
            
            for pattern, replacement in title_patterns:
                result = pattern.sub(replacement, result)
            
            # CRITICAL: Handle month abbreviations BEFORE unit patterns
            # This prevents conflicts with meter pattern
            month_patterns = [
                (re.compile(r'\bม\.ค\.'), 'มกราคม'),
                (re.compile(r'\bก\.พ\.'), 'กุมภาพันธ์'),
                (re.compile(r'\bมี\.ค\.'), 'มีนาคม'),
                (re.compile(r'\bเม\.ย\.'), 'เมษายน'),
                (re.compile(r'\bพ\.ค\.'), 'พฤษภาคม'),
                (re.compile(r'\bมิ\.ย\.'), 'มิถุนายน'),
                (re.compile(r'\bก\.ค\.'), 'กรกฎาคม'),
                (re.compile(r'\bส\.ค\.'), 'สิงหาคม'),
                (re.compile(r'\bก\.ย\.'), 'กันยายน'),
                (re.compile(r'\bต\.ค\.'), 'ตุลาคม'),
                (re.compile(r'\bพ\.ย\.'), 'พฤศจิกายน'),
                (re.compile(r'\bธ\.ค\.'), 'ธันวาคม'),
            ]
            
            for pattern, replacement in month_patterns:
                result = pattern.sub(replacement, result)
            
            # Now apply other complex patterns
            # FIXED: Location patterns WITHOUT trailing spaces for road/soi
            # but WITH space for province/district/subdistrict
            location_patterns = [
                # Special case for Bangkok
                (re.compile(r'\bกทม\.', re.IGNORECASE), 'กรุงเทพมหานคร'),
                
                # Location patterns - FIXED spacing based on expected output
                (re.compile(r'\bจ\.(?=\s*[\u0E00-\u0E7F])'), 'จังหวัด'),     # NO trailing space
                (re.compile(r'\bอ\.(?=\s*[\u0E00-\u0E7F])'), 'อำเภอ'),      # NO trailing space
                (re.compile(r'\bต\.(?=\s*[\u0E00-\u0E7F])'), 'ตำบล'),       # NO trailing space
                (re.compile(r'\bถ\.(?=\s*[\u0E00-\u0E7F])'), 'ถนน'),         # NO space
                (re.compile(r'\bซ\.(?=\s*[\u0E00-\u0E7F])'), 'ซอย'),         # NO space
                
                # Other patterns from complex_patterns
                (re.compile(r'\bม\.\s*(\d+)'), r'มหาวิทยาลัย \1'),
                (re.compile(r'\bรร\.(?=\s*[\u0E00-\u0E7F])'), 'โรงเรียน'),
            ]
            
            for pattern, replacement in location_patterns:
                result = pattern.sub(replacement, result)
            
            # Handle day abbreviations - ONLY when in day context
            day_context_pattern = re.compile(r'วัน\s*(จ\.|อ\.|พ\.|พฤ\.|ศ\.|ส\.|อา\.)')
            
            def replace_day_in_context(match):
                day_abbrev = match.group(1)
                day_map = {
                    'จ.': 'จันทร์',
                    'อ.': 'อังคาร',
                    'พ.': 'พุธ',
                    'พฤ.': 'พฤหัสบดี',
                    'ศ.': 'ศุกร์',
                    'ส.': 'เสาร์',
                    'อา.': 'อาทิตย์',
                }
                return 'วัน' + day_map.get(day_abbrev, day_abbrev)
            
            result = day_context_pattern.sub(replace_day_in_context, result)
            
            # Handle standalone day abbreviations at sentence boundaries
            standalone_day_patterns = [
                (re.compile(r'^(จ\.|พ\.|พฤ\.|ศ\.|ส\.|อา\.)(?=\s|$)'), None),
                (re.compile(r'(?<=\s)(จ\.|พ\.|พฤ\.|ศ\.|ส\.|อา\.)(?=\s|$)'), None),
            ]
            
            day_full_names = {
                'จ.': 'จันทร์',
                'พ.': 'พุธ',
                'พฤ.': 'พฤหัสบดี',
                'ศ.': 'ศุกร์',
                'ส.': 'เสาร์',
                'อา.': 'อาทิตย์',
            }
            
            for pattern, _ in standalone_day_patterns:
                def replace_standalone_day(match):
                    abbrev = match.group(1) if match.lastindex else match.group(0)
                    return day_full_names.get(abbrev, abbrev)
                result = pattern.sub(replace_standalone_day, result)
            
            # FIXED: Add pattern for องศา without degree symbol
            # This should come BEFORE the general unit patterns
            result = re.sub(r'(\d+)องศา\b', r'\1 องศา', result)
            
            # Handle unit abbreviations with numbers - MOVED AFTER MONTH PATTERNS
            unit_patterns = [
                # Distance/Length
                (re.compile(r'(\d+)\s*กม\.'), r'\1 กิโลเมตร'),
                (re.compile(r'(\d+)\s*ม\.'), r'\1 เมตร'),
                (re.compile(r'(\d+)\s*ซม\.'), r'\1 เซนติเมตร'),
                (re.compile(r'(\d+)\s*มม\.'), r'\1 มิลลิเมตร'),
                
                # Weight
                (re.compile(r'(\d+)\s*กก\.'), r'\1 กิโลกรัม'),
                (re.compile(r'(\d+)\s*ก\.'), r'\1 กรัม'),
                (re.compile(r'(\d+)\s*มก\.'), r'\1 มิลลิกรัม'),
                
                # Volume
                (re.compile(r'(\d+)\s*ล\.'), r'\1 ลิตร'),
                (re.compile(r'(\d+)\s*มล\.'), r'\1 มิลลิลิตร'),
                
                # Area
                (re.compile(r'(\d+)\s*ตร\.ม\.'), r'\1 ตารางเมตร'),
                (re.compile(r'(\d+)\s*ตร\.กม\.'), r'\1 ตารางกิโลเมตร'),
                (re.compile(r'(\d+)\s*ลบ\.ม\.'), r'\1 ลูกบาศก์เมตร'),
                
                # Temperature - NO SPACE after องศา (based on expected outputs)
                (re.compile(r'(\d+(?:\.\d+)?)\s*°C', re.IGNORECASE), r'\1 องศาเซลเซียส'),
                (re.compile(r'(\d+(?:\.\d+)?)\s*°F', re.IGNORECASE), r'\1 องศาฟาเรนไฮต์'),
            ]
            
            for pattern, replacement in unit_patterns:
                result = pattern.sub(replacement, result)
            
            # Apply simple abbreviations that don't conflict
            safe_abbreviations = {
                # Special cases
                'ฯลฯ': 'และอื่น ๆ',
                
                # Era markers
                'พ.ศ.': 'พุทธศักราช',
                'ค.ศ.': 'คริสต์ศักราช',
                
                # Government/Organization
                'สนง.': 'สำนักงาน',
                'ส.ส.': 'สมาชิกสภาผู้แทนราษฎร',
                'ส.ว.': 'สมาชิกวุฒิสภา',
                'อบต.': 'องค์การบริหารส่วนตำบล',
                'อบจ.': 'องค์การบริหารส่วนจังหวัด',
                
                # Village number - NO space based on expected output
                'หมู่': 'หมู่ที่',  # This produces "หมู่ที่" without space
            }
            
            for abbrev, expansion in safe_abbreviations.items():
                if abbrev.endswith('.'):
                    pattern = re.compile(r'\b' + re.escape(abbrev) + r'(?![\u0E00-\u0E7F])')
                    result = pattern.sub(expansion, result)
                else:
                    result = result.replace(abbrev, expansion)
            
            # Restore protected words
            for word, placeholder in protected_words.items():
                result = result.replace(placeholder, word)
            
            # CRITICAL FIX: Ensure proper spacing between expanded abbreviations and following words
            # This is especially important for location abbreviations
            # Fix patterns like "จังหวัดกรุงเทพ" to ensure there's a space
            spacing_fixes = [
                (re.compile(r'(จังหวัด|อำเภอ|ตำบล)(?=[\u0E00-\u0E7F])'), r'\1 '),
            ]
            
            for pattern, replacement in spacing_fixes:
                result = pattern.sub(replacement, result)
            
            # Clean up any double spaces
            result = re.sub(r'\s+', ' ', result).strip()
            
            if self.config.enable_debug_logging and text != result:
                logger.debug(f"Abbreviation expansion: '{text}' → '{result}'")
            
        except Exception as e:
            logger.warning(f"Abbreviation expansion failed: {e}")
            return text
        
        return result

class ThaiASRNormalizer:
    """Main Thai ASR Normalizer"""
    
    def __init__(self, config: Optional[ThaiASRConfig] = None):
        self.config = config or ThaiASRConfig()
        
        # Initialize components
        self.text_validator = ThaiTextValidator(self.config)
        self.word_segmenter = ThaiWordSegmenter(self.config)
        self.number_converter = ThaiNumberConverter(self.config)
        self.currency_processor = ThaiCurrencyProcessor(self.config)
        self.phone_processor = ThaiPhoneProcessor(self.config)
        self.datetime_processor = ThaiDateTimeProcessor(self.config)
        self.filler_processor = ThaiFillerProcessor(self.config)
        self.abbreviation_expander = ThaiAbbreviationExpander(self.config)
        
        if self.config.enable_debug_logging:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.debug("Thai ASR Normalizer initialized")
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode and fix encoding issues"""
        if not self.config.normalize_unicode or not text:
            return text or ""
        
        try:
            # Fix encoding issues first
            text = self.text_validator.fix_encoding_issues(text)
            
            # Apply Thai-specific normalization if PyThaiNLP is available
            if PYTHAINLP_AVAILABLE:
                try:
                    text = thai_normalize(text)
                except Exception as e:
                    logger.warning(f"PyThaiNLP normalization failed: {e}")
            
            # Apply Unicode normalization (NFC for Thai)
            text = unicodedata.normalize('NFC', text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}")
            return text
    
    def _handle_code_switching(self, text: str) -> str:
        """Handle Thai-English code switching"""
        if not self.config.handle_code_switching or not text:
            return text
        
        # This is a simple implementation
        # More sophisticated handling could use language detection
        
        # Preserve English words if configured
        if self.config.preserve_english_words:
            # Convert English to lowercase
            def lowercase_english(match):
                return match.group(0).lower()
            
            # Pattern for English words
            english_pattern = re.compile(r'[a-zA-Z]+')
            text = english_pattern.sub(lowercase_english, text)
        
        return text
    
    def _protect_phone_patterns(self, text: str) -> str:
        """Protect phone number patterns before word segmentation"""
        if not text:
            return text
        
        # Store the actual phone matches for later restoration
        self.protected_phones = []
        
        # Phone patterns that might be broken by word segmentation
        phone_patterns = [
            # Mobile numbers (10 digits) - must start with 06, 08, or 09
            (r'\b(0[689]\d{8})\b', 'simple_mobile'),
            (r'\b(0[689]\d)[\s-](\d{3})[\s-](\d{4})\b', 'formatted_mobile'),
            (r'\b(0[689]\d)[\s-](\d{4})[\s-](\d{3})\b', 'formatted_mobile_alt'),
            (r'\b(0[689]\d)\.(\d{3})\.(\d{4})\b', 'dotted_mobile'),
            (r'\(\s*(0[689]\d)\s*\)\s*(\d{3})[\s-](\d{4})\b', 'parentheses_mobile'),
            
            # Landline (9 digits) - must start with 02-05 or 07
            (r'\b(0[2-57]\d{7})\b', 'simple_landline'),
            (r'\b(0[2-57])[\s-](\d{3})[\s-](\d{4})\b', 'formatted_landline'),
            (r'\b(0[2-57]\d)[\s-](\d{6})\b', 'formatted_landline_alt'),
            
            # International format
            (r'\+66\s*(\d{9})\b', 'international_simple'),
            (r'\+66[\s-](\d{1,2})[\s-](\d{3})[\s-](\d{4})\b', 'international_formatted'),
            (r'\+66[\s-](\d)[\s-](\d{4})[\s-](\d{4})\b', 'international_formatted_mobile'),
            
            # Very long numbers that might be international without +
            (r'\b(00668\d{8})\b', 'long_international'),
        ]
        
        protected_text = text
        
        for pattern_str, pattern_type in phone_patterns:
            pattern = re.compile(pattern_str)
            matches = list(pattern.finditer(protected_text))
            
            # Process matches in reverse order to maintain positions
            for match in reversed(matches):
                # Store the match information
                phone_info = {
                    'type': pattern_type,
                    'original': match.group(0),
                    'groups': match.groups(),
                    'start': match.start(),
                    'end': match.end()
                }
                
                # Create a unique placeholder
                placeholder = f'__PHONE_{len(self.protected_phones)}__'
                self.protected_phones.append((placeholder, phone_info))
                
                # Replace with placeholder
                protected_text = protected_text[:match.start()] + placeholder + protected_text[match.end():]
        
        return protected_text

    def _unprotect_phone_patterns(self, text: str) -> str:
        """Restore protected phone patterns and convert them"""
        if not text or not hasattr(self, 'protected_phones'):
            return text
        
        # Process each protected phone
        for placeholder, phone_info in self.protected_phones:
            if placeholder in text:
                # Convert the phone number based on its type
                converted_phone = self._convert_protected_phone(phone_info)
                # Replace placeholder with converted phone
                text = text.replace(placeholder, converted_phone)
        
        # Clean up any remaining artifacts
        text = re.sub(r'__PHONE_\d+__', '', text)
        
        # Clear the protected phones list
        self.protected_phones = []
        
        return text

    def _convert_protected_phone(self, phone_info: dict) -> str:
        """Convert a protected phone number to Thai words"""
        phone_type = phone_info['type']
        groups = phone_info['groups']
        
        # Extract all digits based on phone type
        if phone_type == 'simple_mobile' or phone_type == 'simple_landline':
            all_digits = groups[0]
        
        elif phone_type in ['formatted_mobile', 'formatted_mobile_alt', 'dotted_mobile', 
                            'parentheses_mobile', 'formatted_landline', 'international_formatted_mobile']:
            # Concatenate all digit groups
            all_digits = ''.join(groups)
        
        elif phone_type == 'formatted_landline_alt':
            all_digits = groups[0] + groups[1]
        
        elif phone_type == 'international_simple':
            all_digits = '0' + groups[0]  # Add back the 0
        
        elif phone_type == 'international_formatted':
            # For landline format: +66 2 123 4567
            all_digits = '0' + ''.join(groups)
        
        elif phone_type == 'long_international':
            # 00668xxxxxxxx -> 08xxxxxxxx
            full_number = groups[0]
            if full_number.startswith('00668'):
                all_digits = '08' + full_number[5:]
            elif full_number.startswith('00662'):
                all_digits = '02' + full_number[5:]
            else:
                all_digits = full_number
        
        else:
            # Fallback - just use the original
            all_digits = phone_info['original']
        
        # Remove any non-digit characters that might have been included
        all_digits = re.sub(r'[^\d]', '', all_digits)
        
        # Convert to Thai phone words
        if self.phone_processor and hasattr(self.phone_processor, '_digits_to_thai_phone'):
            return self.phone_processor._digits_to_thai_phone(all_digits)
        else:
            # Fallback conversion
            digit_map = {
                '0': 'ศูนย์', '1': 'หนึ่ง', '2': 'สอง', '3': 'สาม', '4': 'สี่',
                '5': 'ห้า', '6': 'หก', '7': 'เจ็ด', '8': 'แปด', '9': 'เก้า'
            }
            words = []
            for digit in all_digits:
                if digit.isdigit():
                    words.append(digit_map.get(digit, digit))
            return ' '.join(words)

    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation while preserving Thai integrity - FIXED FOR DECIMALS AND THAI MARKS"""
        if not self.config.remove_punctuation or not text:
            return text or ""
        
        try:
            # re module is already imported at the top of the file
            
            # CRITICAL: First protect any existing placeholders from being damaged
            # Store all placeholder patterns
            placeholder_patterns = []
            placeholder_regex = re.compile(r'(__[A-Z_0-9]+__)')
            for match in placeholder_regex.finditer(text):
                placeholder_patterns.append((match.start(), match.end(), match.group(0)))
            
            # Replace placeholders with safe markers
            for i, (start, end, placeholder) in enumerate(placeholder_patterns):
                safe_marker = f'PLACEHOLDER{i}MARKER'
                text = text[:start] + safe_marker + text[end:]
                # Adjust positions for subsequent replacements
                diff = len(safe_marker) - len(placeholder)
                for j in range(i + 1, len(placeholder_patterns)):
                    if placeholder_patterns[j][0] >= end:
                        placeholder_patterns[j] = (
                            placeholder_patterns[j][0] + diff,
                            placeholder_patterns[j][1] + diff,
                            placeholder_patterns[j][2]
                        )
            
            # Now proceed with punctuation removal
            # First, protect decimal points between digits (both Arabic and Thai)
            
            # Protect decimal points between Arabic digits (with or without spaces)
            text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1__DECIMAL__\2', text)
            
            # Protect decimal points between Thai digits (with or without spaces)
            text = re.sub(r'([๐-๙])\s*\.\s*([๐-๙])', r'\1__DECIMAL__\2', text)
            
            # Also protect thousand separators in numbers
            text = re.sub(r'(\d)\s*,\s*(\d)', r'\1__COMMA__\2', text)
            text = re.sub(r'([๐-๙])\s*,\s*([๐-๙])', r'\1__COMMA__\2', text)
            
            # CRITICAL: Protect ์ in ALL words, not just specific ones
            # Replace all occurrences of ์ with a protected version
            text = text.replace('์', '__THAI_MAI_TAI_KHU__')
            
            # CRITICAL FIX: Thai punctuation - REMOVED ๆ from the list!
            # ๆ (MAIYAMOK) is NOT punctuation - it's a repetition marker that's part of words
            thai_punct = 'ฯํ๎'  # Removed both ์ and ๆ from this list
            
            # Remove Thai punctuation
            for punct in thai_punct:
                text = text.replace(punct, ' ')
            
            # Remove ASCII punctuation (except those we're protecting)
            for punct in string.punctuation:
                # Skip the underscore which is part of our protection markers
                if punct == '_':
                    continue
                text = text.replace(punct, ' ')
            
            # Remove other Unicode punctuation
            result = []
            for char in text:
                # Don't remove underscores (part of our protection markers)
                if char == '_':
                    result.append(char)
                elif unicodedata.category(char).startswith('P'):
                    result.append(' ')
                else:
                    result.append(char)
            text = ''.join(result)
            
            # Restore protected punctuation with the actual characters
            text = text.replace('__DECIMAL__', '.')
            text = text.replace('__COMMA__', ',')
            text = text.replace('__THAI_MAI_TAI_KHU__', '์')
            
            # Restore the original placeholders
            for i in range(len(placeholder_patterns)):
                safe_marker = f'PLACEHOLDER{i}MARKER'
                original_placeholder = placeholder_patterns[i][2]
                text = text.replace(safe_marker, original_placeholder)
            
            # Clean up spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.warning(f"Punctuation removal failed: {e}")
            return text
    
    def _fix_decimal_formatting(self, text: str) -> str:
        """Fix decimal formatting that was broken by word segmentation"""
        if not text:
            return text
        
        try:
            # Fix patterns like "123 . 45" -> "123.45"
            text = re.sub(r'(\d+)\s+\.\s+(\d+)', r'\1.\2', text)
            
            # Also handle cases where there might be multiple spaces
            text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
            
            # CRITICAL FIX: Handle cases where word segmentation inserted "decimal" word
            # This happens when Thai digits are converted and then segmented
            # Pattern: "123 decimal 45" -> "123.45"
            text = re.sub(r'(\d+)\s+decimal\s+(\d+)', r'\1.\2', text)
            
            # Also handle without spaces
            text = re.sub(r'(\d+)decimal(\d+)', r'\1.\2', text)
            
            return text
        except Exception as e:
            logger.warning(f"Decimal formatting fix failed: {e}")
            return text
    
    def _protect_date_patterns(self, text: str) -> str:
        """Protect date patterns from word segmentation"""
        if not text:
            return text
        
        import re
        
        # Use a more unique placeholder pattern that won't be split by segmentation
        # Use unicode characters that are unlikely to appear in normal text
        PLACEHOLDER_START = '⟨'  # U+27E8
        PLACEHOLDER_END = '⟩'    # U+27E9
        
        # CRITICAL: First protect any existing "วันที่" as a complete unit
        # This ensures it won't be split by word segmentation
        text = re.sub(r'วัน\s*ที่', f'{PLACEHOLDER_START}DATEMARKER{PLACEHOLDER_END}', text)
        text = text.replace('วันที่', f'{PLACEHOLDER_START}DATEMARKER{PLACEHOLDER_END}')
        
        # Protect day names with วัน prefix - handle วันอาทิตย์ FIRST
        # This prevents it from being misinterpreted
        text = text.replace('วันอาทิตย์', f'{PLACEHOLDER_START}DAYMARKER_อาทิตย์{PLACEHOLDER_END}')
        
        # Then protect other day names
        day_names = ['จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์']
        for day in day_names:
            # Handle cases where วัน might already be separated
            text = re.sub(f'วัน\\s*{re.escape(day)}', f'{PLACEHOLDER_START}DAYMARKER_{day}{PLACEHOLDER_END}', text)
            text = text.replace(f'วัน{day}', f'{PLACEHOLDER_START}DAYMARKER_{day}{PLACEHOLDER_END}')
        
        # Protect day abbreviations BEFORE they go through abbreviation expansion
        day_abbrevs = {
            'จ.': 'จันทร์',
            'อ.': 'อังคาร',  
            'พ.': 'พุธ',
            'พฤ.': 'พฤหัสบดี',
            'ศ.': 'ศุกร์',
            'ส.': 'เสาร์',
            'อา.': 'อาทิตย์'
        }
        
        # Only protect standalone day abbreviations (not อ. for อำเภอ)
        for abbrev, full_day in day_abbrevs.items():
            # Pattern to match standalone day abbreviations
            if abbrev == 'อ.':
                # Special handling for อ. to distinguish from อำเภอ
                # Only treat as day if not followed by place name
                pattern = rf'\b{re.escape(abbrev)}(?!\s*[ก-ฮ]{{2,}})'
            else:
                pattern = rf'\b{re.escape(abbrev)}\b'
            
            text = re.sub(pattern, f'{PLACEHOLDER_START}DAYABBR_{full_day}{PLACEHOLDER_END}', text)
        
        # Protect month names - full names
        month_names = [
            'มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน',
            'พฤษภาคม', 'มิถุนายน', 'กรกฎาคม', 'สิงหาคม',
            'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม'
        ]
        
        for i, month in enumerate(month_names):
            text = text.replace(month, f'{PLACEHOLDER_START}MONTHNAME{i+1}{PLACEHOLDER_END}')
        
        # Protect abbreviated months
        abbrev_months = [
            'ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.',
            'พ.ค.', 'มิ.ย.', 'ก.ค.', 'ส.ค.',
            'ก.ย.', 'ต.ค.', 'พ.ย.', 'ธ.ค.'
        ]
        
        for i, abbrev in enumerate(abbrev_months):
            text = text.replace(abbrev, f'{PLACEHOLDER_START}MONTHABBR{i+1}{PLACEHOLDER_END}')
        
        # Protect colloquial month forms
        colloquial_months = [
            'มกรา', 'กุมภา', 'มีนา', 'เมษา',
            'พฤษภา', 'มิถุนา', 'กรกฎา', 'สิงหา',
            'กันยา', 'ตุลา', 'พฤศจิกา', 'ธันวา'
        ]
        
        for i, month in enumerate(colloquial_months):
            text = text.replace(month, f'{PLACEHOLDER_START}MONTHCOL{i+1}{PLACEHOLDER_END}')
        
        # Protect other date-related words
        text = text.replace('เดือน', f'{PLACEHOLDER_START}MONTHWORD{PLACEHOLDER_END}')
        text = text.replace('ปี', f'{PLACEHOLDER_START}YEARWORD{PLACEHOLDER_END}')
        text = text.replace('พ.ศ.', f'{PLACEHOLDER_START}BUDDHISTERA{PLACEHOLDER_END}')
        text = text.replace('ค.ศ.', f'{PLACEHOLDER_START}CHRISTIANERA{PLACEHOLDER_END}')
        
        return text

    def _unprotect_date_patterns(self, text: str) -> str:
        """Restore protected date patterns - FIXED FOR THAI CHARACTERS"""
        if not text:
            return text
        
        import re
        
        # Use the same unicode placeholder markers
        PLACEHOLDER_START = '⟨'
        PLACEHOLDER_END = '⟩'
        
        # First, clean up any spaces that word segmentation might have added
        # This is critical for proper restoration
        pattern = f'{re.escape(PLACEHOLDER_START)}([^{re.escape(PLACEHOLDER_END)}]+?){re.escape(PLACEHOLDER_END)}'
        
        def clean_placeholder(match):
            content = match.group(1)
            # Remove any spaces within the content
            content = content.replace(' ', '')
            return f'{PLACEHOLDER_START}{content}{PLACEHOLDER_END}'
        
        text = re.sub(pattern, clean_placeholder, text)
        
        # Restore วันที่
        text = text.replace(f'{PLACEHOLDER_START}DATEMARKER{PLACEHOLDER_END}', 'วันที่')
        
        # Restore day names
        day_names = ['จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์', 'อาทิตย์']
        for day in day_names:
            placeholder = f'{PLACEHOLDER_START}DAYMARKER_{day}{PLACEHOLDER_END}'
            if placeholder in text:
                text = text.replace(placeholder, f'วัน{day}')
        
        # Restore day abbreviations to full day names
        for day in day_names:
            placeholder = f'{PLACEHOLDER_START}DAYABBR_{day}{PLACEHOLDER_END}'
            if placeholder in text:
                text = text.replace(placeholder, day)
        
        # Restore month names (full names)
        month_names = [
            'มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน',
            'พฤษภาคม', 'มิถุนายน', 'กรกฎาคม', 'สิงหาคม',
            'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม'
        ]
        
        for i, month in enumerate(month_names):
            text = text.replace(f'{PLACEHOLDER_START}MONTHNAME{i+1}{PLACEHOLDER_END}', month)
        
        # Restore abbreviated months to full names
        month_abbrev_to_full = {
            f'{PLACEHOLDER_START}MONTHABBR1{PLACEHOLDER_END}': 'มกราคม',
            f'{PLACEHOLDER_START}MONTHABBR2{PLACEHOLDER_END}': 'กุมภาพันธ์',
            f'{PLACEHOLDER_START}MONTHABBR3{PLACEHOLDER_END}': 'มีนาคม',
            f'{PLACEHOLDER_START}MONTHABBR4{PLACEHOLDER_END}': 'เมษายน',
            f'{PLACEHOLDER_START}MONTHABBR5{PLACEHOLDER_END}': 'พฤษภาคม',
            f'{PLACEHOLDER_START}MONTHABBR6{PLACEHOLDER_END}': 'มิถุนายน',
            f'{PLACEHOLDER_START}MONTHABBR7{PLACEHOLDER_END}': 'กรกฎาคม',
            f'{PLACEHOLDER_START}MONTHABBR8{PLACEHOLDER_END}': 'สิงหาคม',
            f'{PLACEHOLDER_START}MONTHABBR9{PLACEHOLDER_END}': 'กันยายน',
            f'{PLACEHOLDER_START}MONTHABBR10{PLACEHOLDER_END}': 'ตุลาคม',
            f'{PLACEHOLDER_START}MONTHABBR11{PLACEHOLDER_END}': 'พฤศจิกายน',
            f'{PLACEHOLDER_START}MONTHABBR12{PLACEHOLDER_END}': 'ธันวาคม'
        }
        
        for placeholder, full_month in month_abbrev_to_full.items():
            text = text.replace(placeholder, full_month)
        
        # Restore colloquial months to full names
        month_col_to_full = {
            f'{PLACEHOLDER_START}MONTHCOL1{PLACEHOLDER_END}': 'มกราคม',
            f'{PLACEHOLDER_START}MONTHCOL2{PLACEHOLDER_END}': 'กุมภาพันธ์',
            f'{PLACEHOLDER_START}MONTHCOL3{PLACEHOLDER_END}': 'มีนาคม',
            f'{PLACEHOLDER_START}MONTHCOL4{PLACEHOLDER_END}': 'เมษายน',
            f'{PLACEHOLDER_START}MONTHCOL5{PLACEHOLDER_END}': 'พฤษภาคม',
            f'{PLACEHOLDER_START}MONTHCOL6{PLACEHOLDER_END}': 'มิถุนายน',
            f'{PLACEHOLDER_START}MONTHCOL7{PLACEHOLDER_END}': 'กรกฎาคม',
            f'{PLACEHOLDER_START}MONTHCOL8{PLACEHOLDER_END}': 'สิงหาคม',
            f'{PLACEHOLDER_START}MONTHCOL9{PLACEHOLDER_END}': 'กันยายน',
            f'{PLACEHOLDER_START}MONTHCOL10{PLACEHOLDER_END}': 'ตุลาคม',
            f'{PLACEHOLDER_START}MONTHCOL11{PLACEHOLDER_END}': 'พฤศจิกายน',
            f'{PLACEHOLDER_START}MONTHCOL12{PLACEHOLDER_END}': 'ธันวาคม'
        }
        
        for placeholder, full_month in month_col_to_full.items():
            text = text.replace(placeholder, full_month)
        
        # Restore other words
        text = text.replace(f'{PLACEHOLDER_START}MONTHWORD{PLACEHOLDER_END}', 'เดือน')
        text = text.replace(f'{PLACEHOLDER_START}YEARWORD{PLACEHOLDER_END}', 'ปี')
        text = text.replace(f'{PLACEHOLDER_START}BUDDHISTERA{PLACEHOLDER_END}', 'พ.ศ.')
        text = text.replace(f'{PLACEHOLDER_START}CHRISTIANERA{PLACEHOLDER_END}', 'ค.ศ.')
        
        # Final cleanup - remove any remaining placeholder markers that might have been missed
        text = re.sub(f'{re.escape(PLACEHOLDER_START)}[^{re.escape(PLACEHOLDER_END)}]*{re.escape(PLACEHOLDER_END)}', '', text)
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """Final text cleanup - ASR-focused version"""
        try:
            # Normalize whitespace
            if self.config.normalize_whitespace:
                text = re.sub(r'\s+', ' ', text).strip()
            
            # Handle repeated characters - ONLY if really needed for ASR
            if self.config.handle_repeated_chars:
                
                # Remove repeated spaces (this is always safe)
                text = re.sub(r' {2,}', ' ', text)
                
                # Optionally: Remove repeated punctuation only
                text = re.sub(r'([!?.])\1+', r'\1', text)
            
            # Convert to lowercase
            if self.config.lowercase_output:
                text = text.lower()
            
            # Final validation of Thai text
            if not self.text_validator.validate_thai_text(text):
                logger.warning("Output may not be valid Thai text")
            
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
            
            # Check for Thai character preservation
            original_thai = sum(1 for c in original if '\u0E00' <= c <= '\u0E7F')
            normalized_thai = sum(1 for c in normalized if '\u0E00' <= c <= '\u0E7F')
            
            if normalized_thai < original_thai * 0.5:
                issues.append("Significant Thai character loss")
            
            # Check for text loss
            if len(normalized.strip()) == 0 and len(original.strip()) > 0:
                issues.append("Complete text loss")
            
            # Check for excessive length change
            if len(normalized) > len(original) * 3:
                issues.append("Excessive length increase")
            
            if issues:
                logger.warning(f"Validation issues: {issues}")
                if self.config.strict_mode:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return not self.config.strict_mode
    
    def normalize(self, text: str) -> str:
        """Main normalization method - FIXED VERSION 2"""
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
            logger.debug(f"=== THAI ASR NORMALIZATION START ===")
            logger.debug(f"Input: '{text}'")
        
        try:
            original = text
            
            # Step 1: Unicode normalization and encoding fixes
            text = self._normalize_unicode(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After Unicode normalization: '{text}'")
            
            # Step 2: Convert Thai digits to Arabic
            text = self.number_converter.convert_thai_digits_to_arabic(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After Thai digit conversion: '{text}'")

            # Step 2.5: Convert Thai number words to Arabic numerals (NEW)
            text = self.number_converter.convert_thai_words_to_numbers(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After Thai number word conversion: '{text}'")
    
            # Step 3: Expand abbreviations
            text = self.abbreviation_expander.expand_abbreviations(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After abbreviation expansion: '{text}'")
            
            # Step 4: Normalize date/time FIRST (before any other processing)
            text = self.datetime_processor.normalize_datetime(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After datetime normalization: '{text}'")
            
            # Step 5: Normalize phone numbers
            text = self.phone_processor.normalize_phones(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After phone normalization: '{text}'")
            
            # Step 6: Normalize currency
            text = self.currency_processor.normalize_currency(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After currency normalization: '{text}'")
            
            # Step 7: Remove filler words
            text = self.filler_processor.remove_fillers(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After filler removal: '{text}'")
            
            # Step 8: Handle code switching
            text = self._handle_code_switching(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After code switching: '{text}'")
            
            # Step 9: Word segmentation (if enabled) - MOVED BEFORE PUNCTUATION REMOVAL
            if self.config.use_word_segmentation:
                # CRITICAL: Protect date patterns before segmentation
                text = self._protect_date_patterns(text)
                if self.config.enable_debug_logging:
                    logger.debug(f"After date protection: '{text}'")
                
                # Also protect phone patterns
                text = self._protect_phone_patterns(text)
                if self.config.enable_debug_logging:
                    logger.debug(f"After phone protection: '{text}'")
                
                # FIXED: Store compound words in a separate mapping
                # instead of using inline placeholders
                compound_mappings = {}
                compound_counter = 0
                
                # Protect compound place names
                compound_place_patterns = [
                    'บางแพ', 'บางรัก', 'บางกอก', 'บางนา', 'บางพลี', 'บางเขน',
                    'บางซื่อ', 'บางกะปิ', 'บางแค', 'บางขุนเทียน', 'บางพลัด',
                    'คลองเตย', 'ลาดพร้าว', 'สุขุมวิท', 'พญาไท', 'ดินแดง',
                ]
                
                # Protect titles and their compounds
                title_compounds = [
                    'ผู้ช่วยศาสตราจารย์', 'รองศาสตราจารย์', 'ศาสตราจารย์ดอกเตอร์',
                    'พันตำรวจเอก', 'พันตำรวจ', 'ร้อยตำรวจ', 'พลตำรวจ',
                    'ดอกเตอร์', 'มหาวิทยาลัย', 'โรงเรียน',
                ]
                
                # Protect unit compounds
                unit_compounds = [
                    'องศาเซลเซียส', 'องศาฟาเรนไฮต์', 'กิโลเมตร', 'เซนติเมตร',
                    'มิลลิเมตร', 'กิโลกรัม', 'มิลลิกรัม', 'มิลลิลิตร',
                    'ตารางเมตร', 'ตารางกิโลเมตร', 'ลูกบาศก์เมตร',
                ]
                
                # Protect "หมู่ที่" as a compound
                special_compounds = [
                    'หมู่ที่',
                ]
                
                # Combine all compound patterns
                all_compounds = compound_place_patterns + title_compounds + unit_compounds + special_compounds
                
                # Replace compounds with simple markers
                for compound in all_compounds:
                    if compound in text:
                        # Use a simple alphanumeric marker that won't be corrupted
                        marker = f'CMPND{compound_counter:03d}'
                        compound_mappings[marker] = compound
                        text = text.replace(compound, marker)
                        compound_counter += 1
                        if self.config.enable_debug_logging:
                            logger.debug(f"Protected '{compound}' as '{marker}'")
                
                # Protect any date patterns that might have been created by datetime normalization
                date_pattern = re.compile(r'(วันที่\s+\d{1,2}\s+(?:มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)\s+\d{4})')
                date_counter = 0
                for match in date_pattern.finditer(text):
                    marker = f'DTMRK{date_counter:03d}'
                    compound_mappings[marker] = match.group(0)
                    text = text.replace(match.group(0), marker)
                    date_counter += 1
                
                # Perform segmentation
                text = self.word_segmenter.segment(text)
                if self.config.enable_debug_logging:
                    logger.debug(f"After word segmentation: '{text}'")
                
                # Restore all protected patterns
                # Sort markers by length (longest first) to avoid partial replacements
                sorted_markers = sorted(compound_mappings.keys(), key=len, reverse=True)
                for marker in sorted_markers:
                    original = compound_mappings[marker]
                    # Try multiple replacement patterns in case segmenter added spaces
                    patterns_to_try = [
                        f' {marker} ',
                        f'{marker} ',
                        f' {marker}',
                        marker
                    ]
                    for pattern in patterns_to_try:
                        if pattern in text:
                            text = text.replace(pattern, original)
                            break
                
                # Restore other protected patterns
                text = self._unprotect_phone_patterns(text)
                text = self._unprotect_date_patterns(text)
                if self.config.enable_debug_logging:
                    logger.debug(f"After pattern restoration: '{text}'")
                
                # Fix decimal formatting broken by word segmentation
                text = self._fix_decimal_formatting(text)
                if self.config.enable_debug_logging:
                    logger.debug(f"After decimal fix: '{text}'")
            
            # Step 10: Remove punctuation (MOVED AFTER word segmentation)
            text = self._remove_punctuation(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After punctuation removal: '{text}'")
            
            # Step 11: Final cleanup
            text = self._final_cleanup(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After final cleanup: '{text}'")
            
            # Step 12: Validation
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

class ThaiASRNormaliser:
    """AudioBench-compatible Thai normaliser wrapper"""
    
    def __init__(self, config_or_enable=None, enable_disfluency_removal: bool = True, **kwargs):
        """Initialize the Thai normalizer with flexible constructor.

        Supports the following signatures for backward compatibility:
        - (config: ThaiASRConfig, enable_disfluency_removal: bool)
        - (enable_disfluency_removal: bool)
        - ()
        """
        # Determine configuration based on provided arguments
        config = None
        try:
            # Import type locally to avoid circulars at module import time
            _ThaiASRConfigType = ThaiASRConfig
        except Exception:
            _ThaiASRConfigType = None

        if _ThaiASRConfigType is not None and isinstance(config_or_enable, _ThaiASRConfigType):
            # Signature: (config, enable)
            config = config_or_enable
            self.enable_disfluency_removal = bool(enable_disfluency_removal)
        else:
            # Signature: (enable) or ()
            if isinstance(config_or_enable, bool):
                self.enable_disfluency_removal = config_or_enable
            else:
                self.enable_disfluency_removal = bool(enable_disfluency_removal)
            
            config = ThaiASRConfig(
                remove_filler_words=self.enable_disfluency_removal,
                handle_repeated_chars=True,
                normalize_currency=True,
                convert_numbers_to_words=False,
                lowercase_output=True,
                normalize_whitespace=True,
                handle_code_switching=True,
                use_word_segmentation=False,
                enable_validation=True
            )
        
        # Initialize the main normalizer
        self.normalizer = ThaiASRNormalizer(config)
    
    def normalize(self, text: str) -> str:
        """Compatibility method for the new normalizer system"""
        return self.normalize_for_asr_eval(text)
    
    def normalize_for_asr_eval(self, text: str, 
                               use_preprocessing: bool = True,
                               apply_asr_standardization: bool = True) -> str:
        """Main method called by AudioBench for text normalization"""
        if not text:
            return ""
        
        try:
            # Use the main ThaiASRNormalizer
            return self.normalizer.normalize(text)
        except Exception as e:
            logger.error(f"Thai normalization failed: {e}")
            # Fallback: basic text cleaning
            import re
            text = re.sub(r'\s+', ' ', text.strip())
            return text