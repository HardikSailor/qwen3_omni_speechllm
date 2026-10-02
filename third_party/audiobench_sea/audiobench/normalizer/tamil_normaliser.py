#!/usr/bin/env python3
"""
Fixed Tamil ASR Normalizer for Fair CER Calculation
Based on Indic NLP Library and Tamil language specifics
"""

import re
import logging
import string
import unicodedata
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import traceback

# Set up logger
logger = logging.getLogger(__name__)

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("Warning: jiwer not available - install with: pip install jiwer")

# Indic NLP Library
try:
    from indicnlp.tokenize import indic_tokenize
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
    from indicnlp import common, loader
    INDIC_NLP_AVAILABLE = True
except ImportError:
    INDIC_NLP_AVAILABLE = False
    print("Warning: Indic NLP not available - install with: pip install indic-nlp-library")

# Indic Num2Words for Tamil number conversion
try:
    from indic_numtowords import num2words as indic_num2words
    INDIC_NUM2WORDS_AVAILABLE = True
except ImportError:
    INDIC_NUM2WORDS_AVAILABLE = False
    print("Warning: indic-numtowords not available - install with: pip install indic-numtowords")

# Standard Num2Words (fallback, though limited Tamil support)
try:
    from num2words import num2words
    NUM2WORDS_AVAILABLE = True
except ImportError:
    NUM2WORDS_AVAILABLE = False
    print("Warning: num2words not available for enhanced number processing")

# Handle case where __name__ might not be available during direct file import
try:
    module_name = __name__
except NameError:
    module_name = "audiobench.normalizer.tamil_normaliser"

logger = logging.getLogger(module_name)

@dataclass
class TamilASRConfig:
    """Configuration for Tamil ASR normalization - FIXED FOR ASR EVALUATION"""
    
    # Core text processing
    normalize_unicode: bool = True
    remove_punctuation: bool = True
    lowercase_output: bool = True
    normalize_whitespace: bool = True
    
    # Tamil-specific processing - FIXED FOR ASR
    convert_tamil_numerals: bool = True  # Convert ௧௨௩ → 123
    convert_numbers_to_words: bool = False  # CHANGED: Keep as digits for ASR
    convert_number_words_to_digits: bool = True  # NEW: Convert words to digits
    normalize_currency: bool = True
    normalize_phone_numbers: bool = True
    normalize_time_format: bool = True
    normalize_date_format: bool = True
    
    # Text cleaning
    remove_filler_words: bool = True
    remove_discourse_markers: bool = True
    merge_repeated_chars: bool = True
    handle_english_mixed_text: bool = True
    
    # Advanced options
    preserve_decimal_points: bool = False
    preserve_percentages: bool = True
    enable_transliteration_cleanup: bool = True
    enable_validation: bool = True
    enable_debug_logging: bool = False
    strict_mode: bool = False
    max_text_length: int = 500000

class TamilNumberConverter:
    """Tamil number converter - FIXED VERSION: Better number word conversion"""
    
    def __init__(self, config: TamilASRConfig):
        self.config = config
        self._setup_external_library()
        self._setup_static_mappings()
        self._compile_patterns()
        
        # Initialize caching if enabled
        self.cache_enabled = getattr(config, 'enable_caching', True)
        self.max_cache_size = getattr(config, 'max_cache_size', 1000)
        self.conversion_cache = {}
    
    def _setup_external_library(self):
        """Setup indic-numtowords for Tamil number conversion"""
        self.num2words_available = False
        self.tamil_converter = None
        
        try:
            from indic_numtowords import num2words as indic_num2words
            test_result = indic_num2words(42, lang='ta', variations=False)
            self.tamil_converter = lambda n: indic_num2words(n, lang='ta', variations=False)
            self.num2words_available = True
            
            if self.config.enable_debug_logging:
                logger.debug(f"✅ indic-numtowords loaded: 42 → '{test_result}'")
                
        except ImportError:
            logger.warning("❌ indic-numtowords not available. Using built-in converter.")
        except Exception as e:
            logger.warning(f"Failed to initialize indic-numtowords: {e}")
        
        if not self.num2words_available:
            self.tamil_converter = self._fallback_converter


    def _setup_static_mappings(self):
            """Setup COMPLETE static mappings - ENHANCED VERSION WITH ஒரு FIXED"""
            
            # Tamil Unicode numerals → Arabic digits (always safe)
            self.tamil_numerals_map = {
                '௦': '0', '௧': '1', '௨': '2', '௩': '3', '௪': '4',
                '௫': '5', '௬': '6', '௭': '7', '௮': '8', '௯': '9',
                '௰': '10', '௱': '100', '௲': '1000'
            }
            
            # COMPREHENSIVE: All unambiguous number words - SIGNIFICANTLY EXPANDED WITH ஒரு FIXED
            self.static_word_mappings = {
                # Zero variations
                'சுழியம்': 0, 'சுன்னியம்': 0, 'பூஜ்ஜியம்': 0, 'சுன்னம்': 0,
                
                # Basic numbers 1-10 - CRITICAL FIX: Added ஒரு
                'ஒரு': 1, 'ஒன்று': 1, 'இரண்டு': 2, 'மூன்று': 3, 'நான்கு': 4, 'ஐந்து': 5,
                'ஆறு': 6, 'ஏழு': 7, 'எட்டு': 8, 'ஒன்பது': 9, 'பத்து': 10,
                
                # Colloquial forms
                'ரெண்டு': 2, 'மூணு': 3, 'நாலு': 4, 'அஞ்சு': 5,
                
                # Teen numbers
                'பதினொன்று': 11, 'பன்னிரண்டு': 12, 'பதிரண்டு': 12,
                'பதின்மூன்று': 13, 'பதினான்கு': 14, 'பதினைந்து': 15,
                'பதினாறு': 16, 'பதினேழு': 17, 'பதினெட்டு': 18, 'பத்தொன்பது': 19,
                
                # Tens
                'இருபது': 20, 'முப்பது': 30, 'நாற்பது': 40, 'ஐம்பது': 50,
                'அறுபது': 60, 'எழுபது': 70, 'எண்பது': 80, 'தொண்ணூறு': 90,
                
                # Scale words
                'நூறு': 100, 'ஆயிரம்': 1000, 'லட்சம்': 100000, 'இலட்சம்': 100000,
                'கோடி': 10000000, 'அர்புதம்': 1000000000, 'நீலம்': 100000000000,
                
                # ENHANCED: Common compound scale words that appear as single tokens
                'ஐநூறு': 500,  # This was the main missing word!
                'இருநூறு': 200,
                'மூன்றூறு': 300,
                'நான்நூறு': 400,
                'ஆறூறு': 600,
                'எழுநூறு': 700,
                'எண்ணூறு': 800,
                'தொள்ளாயிரம்': 900,
                
                # Additional compound thousands
                'இரண்டாயிரம்': 2000,
                'மூன்றாயிரம்': 3000,
                'நான்காயிரம்': 4000,
                'ஐந்தாயிரம்': 5000,
                'ஆறாயிரம்': 6000,
                'ஏழாயிரம்': 7000,
                'எட்டாயிரம்': 8000,
                'ஒன்பதாயிரம்': 9000,
                'பத்தாயிரம்': 10000,
                
                # ADDITIONAL: More scale combinations commonly used
                'அரை': 0.5,  # Half
                'கால்': 0.25,  # Quarter
                'முக்கால்': 0.75,  # Three quarters
                
                # Special number words - ENHANCED WITH ஒரு VARIANTS
                'இரு': 2,   # Alternative form of "two"
                
                # Special indicators (punctuation-like)
                'புள்ளி': '.', 'தசமம்': '.', 'போயிண்ட்': '.',
                'சதவீதம்': '%', 'விழுக்காடு': '%', 'பர்சென்ட்': '%'
            }
            
            # Compound number mappings for "இருபத்தி ஐந்து" style numbers
            self.compound_base_map = {
                'இருபத்தி': 20, 'முப்பத்தி': 30, 'நாற்பத்தி': 40, 'ஐம்பத்தி': 50,
                'அறுபத்தி': 60, 'எழுபத்தி': 70, 'எண்பத்தி': 80, 'தொண்ணூற்றி': 90,
                'இருபத்து': 20, 'முப்பத்து': 30, 'நாற்பத்து': 40, 'ஐம்பத்து': 50,
                'அறுபத்து': 60, 'எழுபத்து': 70, 'எண்பத்து': 80, 'தொண்ணூற்று': 90,
            }
            
            self.compound_unit_map = {
                'ஒன்று': 1, 'இரண்டு': 2, 'மூன்று': 3, 'நான்கு': 4, 'ஐந்து': 5,
                'ஆறு': 6, 'ஏழு': 7, 'எட்டு': 8, 'ஒன்பது': 9,
                'ரெண்டு': 2, 'மூணு': 3, 'நாலு': 4, 'அஞ்சு': 5,
                'ஒரு': 1,  # CRITICAL: Added ஒரு here too
            }
            
            # Scale hierarchy for large number processing
            self.scale_hierarchy = [
                ('நீலம்', 100000000000), ('அர்புதம்', 1000000000), ('கோடி', 10000000),
                ('லட்சம்', 100000), ('இலட்சம்', 100000), ('ஆயிரம்', 1000), ('நூறு', 100),
            ]
            
            logger.debug(f"Static mappings setup complete")
            logger.debug(f"{len(self.static_word_mappings)} word mappings")
            logger.debug(f"Sample mappings: {list(self.static_word_mappings.items())[:10]}")
            
            # Verify critical words are included - UPDATED TO INCLUDE ஒரு
            critical_words = ['ஒரு', 'ஐநூறு', 'இரண்டு', 'லட்சம்', 'ஆயிரம்', 'ஒன்று', 'நூறு', 'பத்து']
            for word in critical_words:
                if word in self.static_word_mappings:
                    logger.debug(f"✅ Critical word '{word}' → {self.static_word_mappings[word]}")
                else:
                    logger.debug(f"❌ Missing critical word '{word}'")
    
    def _compile_patterns(self):
        """Compile regex patterns - FIXED VERSION for Tamil Unicode WITH COLON SUPPORT"""
        
        logger.debug("Compiling patterns...")
        
        # ALL number words that should be converted - FIXED TO INCLUDE MISSING WORDS
        all_convertible_words = [
            # Basic numbers 1-10
            'ஒன்று', 'இரண்டு', 'மூன்று', 'நான்கு', 'ஐந்து', 
            'ஆறு', 'ஏழு', 'எட்டு', 'ஒன்பது', 'பத்து',
            
            # Colloquial variants
            'ரெண்டு', 'மூணு', 'நாலு', 'அஞ்சு',
            
            # Teen numbers
            'பதினொன்று', 'பன்னிரண்டு', 'பதிரண்டு', 'பதின்மூன்று', 
            'பதினான்கு', 'பதினைந்து', 'பதினாறு', 'பதினேழு', 
            'பதினெட்டு', 'பத்தொன்பது',
            
            # Tens
            'இருபது', 'முப்பது', 'நாற்பது', 'ஐம்பது',
            'அறுபது', 'எழுபது', 'எண்பது', 'தொண்ணூறு',
            
            # Scale words
            'நூறு', 'ஆயிரம்', 'லட்சம்', 'இலட்சம்', 'கோடி',
            'அர்புதம்', 'நீலம்',
            
            # FIXED: Add compound scale words that were missing
            'ஐநூறு', 'இருநூறு', 'மூன்றூறு', 'நான்நூறு', 
            'ஆறூறு', 'எழுநூறு', 'எண்ணூறு', 'தொள்ளாயிரம்',
            'இரண்டாயிரம்', 'மூன்றாயிரம்', 'நான்காயிரம்', 'ஐந்தாயிரம்',
            
            # Zero variants
            'சுழியம்', 'சுன்னியம்', 'பூஜ்ஜியம்', 'சுன்னம்'
        ]
        
        # ENHANCEMENT: Add any other words from static_word_mappings that aren't in the list
        for word in self.static_word_mappings.keys():
            if (isinstance(self.static_word_mappings[word], (int, float)) and 
                word not in all_convertible_words and
                word not in ['.', '%']):  # Skip punctuation-like mappings
                all_convertible_words.append(word)
        
        # Sort by length (longest first) to avoid partial matches
        all_convertible_words.sort(key=len, reverse=True)
        
        logger.debug(f"{len(all_convertible_words)} convertible words compiled")
        logger.debug(f"Sample words: {all_convertible_words[:10]}")
        
        # Compound number bases and units - FIXED
        compound_bases = ['இருபத்தி', 'முப்பத்தி', 'நாற்பத்தி', 'ஐம்பத்தி', 
                        'அறுபத்தி', 'எழுபத்தி', 'எண்பத்தி', 'தொண்ணூற்றி',
                        'இருபத்து', 'முப்பத்து', 'நாற்பத்து', 'ஐம்பத்து',
                        'அறுபத்து', 'எழுபத்து', 'எண்பத்து', 'தொண்ணூற்று']
        
        compound_units = ['ஒன்று', 'இரண்டு', 'மூன்று', 'நான்கு', 'ஐந்து', 
                        'ஆறு', 'ஏழு', 'எட்டு', 'ஒன்பது',
                        'ரெண்டு', 'மூணு', 'நாலு', 'அஞ்சு']
        
        # Create patterns in order of priority
        self.patterns = []
        
        # Pattern 1: Tamil Unicode numerals (CRITICAL FIX: Include colons for time expressions)
        self.patterns.append((
            re.compile(r'[௦-௯௰௱௲]+(?::[௦-௯௰௱௲]+)?'), 
            self._convert_tamil_numerals
        ))
        
        # Pattern 2: Compound numbers like "இருபத்தி ஐந்து" - FIXED UNICODE BOUNDARIES
        compound_bases_escaped = [re.escape(base) for base in compound_bases]
        compound_units_escaped = [re.escape(unit) for unit in compound_units]
        
        # FIXED: Better word boundaries for Tamil with proper Unicode ranges
        # Tamil Unicode range: \u0B80-\u0BFF
        compound_pattern = r'(?<![a-zA-Z0-9\u0B80-\u0BFF])(' + '|'.join(compound_bases_escaped) + r')\s+(' + '|'.join(compound_units_escaped) + r')(?![a-zA-Z0-9\u0B80-\u0BFF])'
        
        self.patterns.append((
            re.compile(compound_pattern, re.UNICODE), 
            self._convert_compound_number
        ))
        
        # Pattern 3: Number + scale (like "ஐந்து நூறு", "இரண்டு லட்சம்") - ENHANCED
        scale_compatible_numbers = [
            'ஒன்று', 'இரண்டு', 'மூன்று', 'நான்கு', 'ஐந்து', 
            'ஆறு', 'ஏழு', 'எட்டு', 'ஒன்பது', 'பத்து',
            'பதினொன்று', 'பன்னிரண்டு', 'பதின்மூன்று', 'பதினான்கு', 'பதினைந்து',
            'பதினாறு', 'பதினேழு', 'பதினெட்டு', 'பத்தொன்பது',
            'இருபது', 'முப்பது', 'நாற்பது', 'ஐம்பது',
            'அறுபது', 'எழுபது', 'எண்பது', 'தொண்ணூறு',
            'நூறு', 'ரெண்டு', 'மூணு', 'நாலு', 'அஞ்சு'
        ]
        
        scale_words = ['நூறு', 'ஆயிரம்', 'லட்சம்', 'இலட்சம்', 'கோடி', 'அர்புதம்', 'நீலம்']
        
        scale_compatible_numbers_escaped = [re.escape(num) for num in scale_compatible_numbers]
        scale_words_escaped = [re.escape(scale) for scale in scale_words]
        
        scale_pattern = r'(?<![a-zA-Z0-9\u0B80-\u0BFF])(' + '|'.join(scale_compatible_numbers_escaped) + r')\s+(' + '|'.join(scale_words_escaped) + r')(?![a-zA-Z0-9\u0B80-\u0BFF])'
        self.patterns.append((
            re.compile(scale_pattern, re.UNICODE), 
            self._convert_scale_number
        ))
        
        # Pattern 4: All individual number words - ENHANCED
        all_words_escaped = [re.escape(word) for word in all_convertible_words]
        simple_pattern = r'(?<![a-zA-Z0-9\u0B80-\u0BFF])(' + '|'.join(all_words_escaped) + r')(?![a-zA-Z0-9\u0B80-\u0BFF])'
        self.patterns.append((
            re.compile(simple_pattern, re.UNICODE), 
            self._convert_simple_number_word
        ))
        
        logger.debug(f" Compiled {len(self.patterns)} patterns")
        
        # Test the patterns - IMPROVED TESTING (only if debug logging is enabled)
        if self.config.enable_debug_logging:
            test_text = "ஒன்று இரண்டு இருபத்தி ஐந்து ஆயிரம் ஐநூறு ௯:௩௦"
            logger.debug(f" Testing patterns on: '{test_text}'")
            
            for i, (pattern, converter) in enumerate(self.patterns):
                matches = list(pattern.finditer(test_text))
                logger.debug(f"  Pattern {i+1}: {len(matches)} matches")
                for match in matches:
                    logger.debug(f"    - '{match.group(0)}' at {match.span()}")
    
    def _convert_simple_number_word(self, match):
        """Convert simple Tamil number words to digits - ENHANCED VERSION"""
        word = match.group(1)  # Get the captured group
        
        if self.config.enable_debug_logging:
            logger.debug(f" Converting simple number word: '{word}'")
        
        # Early return for empty or None values
        if not word or not word.strip():
            return word
        
        word = word.strip()
        
        # Check static mappings first (most comprehensive)
        if word in self.static_word_mappings:
            mapped_value = self.static_word_mappings[word]
            if isinstance(mapped_value, (int, float)):
                result = str(mapped_value)
                if self.config.enable_debug_logging:
                    logger.debug(f" Found in static mappings: '{word}' → '{result}'")
                return result
            elif isinstance(mapped_value, str):
                # Handle special cases like punctuation
                if self.config.enable_debug_logging:
                    logger.debug(f" Found special mapping: '{word}' → '{mapped_value}'")
                return mapped_value
        
        # Check compound base mappings
        if word in self.compound_base_map:
            result = str(self.compound_base_map[word])
            if self.config.enable_debug_logging:
                logger.debug(f" Found in compound base: '{word}' → '{result}'")
            return result
        
        # Check compound unit mappings
        if word in self.compound_unit_map:
            result = str(self.compound_unit_map[word])
            if self.config.enable_debug_logging:
                logger.debug(f" Found in compound unit: '{word}' → '{result}'")
            return result
        
        # Try reverse engineering for unknown words
        reverse_result = self._reverse_engineer_number(word)
        if isinstance(reverse_result, (int, float)):
            result = str(reverse_result)
            if self.config.enable_debug_logging:
                logger.debug(f" Reverse engineered: '{word}' → '{result}'")
            return result
        
        # If all else fails, return the original word
        if self.config.enable_debug_logging:
            logger.debug(f" No conversion found for: '{word}' - keeping original")
        return word

    def _convert_tamil_numerals(self, match):
        """Convert Tamil Unicode numerals to Arabic digits - FIXED COLON HANDLING"""
        text = match.group(0)
        
        if self.config.enable_debug_logging:
            logger.debug(f"Converting Tamil numerals: '{text}'")
        
        if not text:
            return ""
        
        # CRITICAL FIX: Handle colon-separated Tamil numerals for time expressions
        if ':' in text:
            # Split by colon and convert each part separately
            parts = text.split(':')
            converted_parts = []
            
            for part in parts:
                if part:  # Skip empty parts
                    converted_part = self._convert_single_tamil_numeral_part(part)
                    converted_parts.append(converted_part)
                else:
                    converted_parts.append('')
            
            result = ':'.join(converted_parts)
            if self.config.enable_debug_logging:
                logger.debug(f"Colon-separated conversion: '{text}' → '{result}'")
            return result
        
        # Handle single numeral without colon
        result = self._convert_single_tamil_numeral_part(text)
        if self.config.enable_debug_logging:
            logger.debug(f"Single numeral conversion: '{text}' → '{result}'")
        return result

    def _convert_single_tamil_numeral_part(self, text):
        """Convert a single Tamil numeral part (without colons) to Arabic digits - ENHANCED"""
        if not text:
            return ""
        
        # Direct mapping for Tamil digits
        tamil_to_arabic = {
            '௦': '0', '௧': '1', '௨': '2', '௩': '3', '௪': '4',
            '௫': '5', '௬': '6', '௭': '7', '௮': '8', '௯': '9'
        }
        
        # For single character, use simple mapping
        if len(text) == 1 and text in tamil_to_arabic:
            result = tamil_to_arabic[text]
            if self.config.enable_debug_logging:
                logger.debug(f"Single character: '{text}' → '{result}'")
            return result
        
        try:
            # Check if it contains scale markers (௰, ௱, ௲)
            if any(c in text for c in ['௰', '௱', '௲']):
                result = self._parse_tamil_numeral_complex(text)
                if self.config.enable_debug_logging:
                    logger.debug(f"Complex parsing: '{text}' → '{result}'")
                return str(result)
            else:
                # Pure digit sequence - convert each Tamil digit to Arabic
                result = ""
                for char in text:
                    if char in tamil_to_arabic:
                        result += tamil_to_arabic[char]
                    # Do not add any other characters here to avoid corruption
                
                if self.config.enable_debug_logging:
                    logger.debug(f"Digit sequence: '{text}' → '{result}'")
                return result
                
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Tamil numeral parsing failed for '{text}': {e}")
            # Fallback to simple digit conversion
            result = ""
            for char in text:
                if char in tamil_to_arabic:
                    result += tamil_to_arabic[char]
            return result
    
    def _parse_multipart_numeral(self, tamil_numeral, digit_map, scale_map):
        """Parse complex multi-part Tamil numerals like ௧௲௫௱௧௰௫"""
        try:
            # Example: ௧௲௫௱௧௰௫ should parse as 1*1000 + 5*100 + 1*10 + 5 = 1515
            
            total = 0
            i = 0
            
            while i < len(tamil_numeral):
                char = tamil_numeral[i]
                
                if char in digit_map:
                    digit_value = digit_map[char]
                    
                    # Look ahead for scale marker
                    if i + 1 < len(tamil_numeral) and tamil_numeral[i + 1] in scale_map:
                        scale_value = scale_map[tamil_numeral[i + 1]]
                        total += digit_value * scale_value
                        i += 2  # Skip both digit and scale
                    else:
                        # Standalone digit at the end
                        total += digit_value
                        i += 1
                else:
                    # Skip any unexpected characters
                    i += 1
            
            return total
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Multipart parsing failed: {e}")
            return None

    def _parse_tamil_numeral_complex(self, tamil_numeral):
        """Parse complex Tamil numerals - ENHANCED FOR BETTER PATTERN RECOGNITION"""
        
        digit_map = {'௦': 0, '௧': 1, '௨': 2, '௩': 3, '௪': 4,
                    '௫': 5, '௬': 6, '௭': 7, '௮': 8, '௯': 9}
        scale_map = {'௰': 10, '௱': 100, '௲': 1000}
        
        if self.config.enable_debug_logging:
            logger.debug(f"Parsing complex Tamil numeral: '{tamil_numeral}'")
        
        # ENHANCED: Handle more specific patterns first
        
        # Pattern 1: ௪௰௰௰ (4000) - digit followed by multiple ௰
        if len(tamil_numeral) >= 2 and tamil_numeral.count('௰') >= 2:
            chars = list(tamil_numeral)
            if chars[0] in digit_map and all(c == '௰' for c in chars[1:]):
                # Pattern like ௪௰௰௰ = 4 * (10^3) = 4000
                base_digit = digit_map[chars[0]]
                power = len(chars) - 1
                result = base_digit * (10 ** power)
                if self.config.enable_debug_logging:
                    logger.debug(f"Multiple ௰ pattern: {tamil_numeral} → {result}")
                return result
        
        # Pattern 2: ௫௦௰ (500) - digit + ௦ + ௰
        if len(tamil_numeral) == 3:
            chars = list(tamil_numeral)
            if (chars[0] in digit_map and chars[1] == '௦' and chars[2] == '௰'):
                # This represents hundreds: ௫௦௰ = 5 * 100 = 500
                hundreds = digit_map[chars[0]]
                result = hundreds * 100
                if self.config.enable_debug_logging:
                    logger.debug(f"Hundreds pattern X௦௰: {tamil_numeral} → {result}")
                return result
        
        # Pattern 3: ௨௦௰௰ (2000) - digit + ௦ + multiple ௰
        if len(tamil_numeral) == 4:
            chars = list(tamil_numeral)
            if (chars[0] in digit_map and chars[1] == '௦' and 
                chars[2] == '௰' and chars[3] == '௰'):
                # This represents thousands: ௨௦௰௰ = 2 * 1000 = 2000
                thousands = digit_map[chars[0]]
                result = thousands * 1000
                if self.config.enable_debug_logging:
                    logger.debug(f"Thousands pattern X௦௰௰: {tamil_numeral} → {result}")
                return result
            
            # Pattern 4: Year format ௨௰௧௧ (2011)
            if (chars[0] in digit_map and chars[1] == '௰' and 
                chars[2] in digit_map and chars[3] in digit_map):
                thousands = digit_map[chars[0]]
                tens = digit_map[chars[2]]
                units = digit_map[chars[3]]
                result = thousands * 1000 + tens * 10 + units
                if self.config.enable_debug_logging:
                    logger.debug(f"Year pattern X௰YZ: {tamil_numeral} → {result}")
                return result
        
        # Pattern 5: Complex 5+ digit patterns like ௧௲௫௱௧௰௫ (1515)
        if len(tamil_numeral) >= 5:
            result = self._parse_multipart_numeral(tamil_numeral, digit_map, scale_map)
            if result is not None:
                if self.config.enable_debug_logging:
                    logger.debug(f"Multipart pattern: {tamil_numeral} → {result}")
                return result
        
        # Pattern 6: General scale-based parsing
        total = 0
        current_number = 0
        i = 0
        
        while i < len(tamil_numeral):
            char = tamil_numeral[i]
            
            if char in digit_map:
                current_number = digit_map[char]
            elif char in scale_map:
                scale_value = scale_map[char]
                if current_number == 0:
                    current_number = 1  # Implicit 1 before scale
                total += current_number * scale_value
                current_number = 0
            i += 1
        
        # Add any remaining number
        total += current_number
        
        if self.config.enable_debug_logging:
            logger.debug(f"General scale parsing: {tamil_numeral} → {total}")
        
        return total

    def _parse_simple_tamil_part(self, part):
        """Helper method to parse simple Tamil numeral parts like ௧௰௫"""
        digit_map = {'௦': 0, '௧': 1, '௨': 2, '௩': 3, '௪': 4,
                    '௫': 5, '௬': 6, '௭': 7, '௮': 8, '௯': 9}
        
        if len(part) == 3 and part[1] == '௰':
            # Pattern like ௧௰௫ = 1*10 + 5 = 15
            tens = digit_map.get(part[0], 0)
            units = digit_map.get(part[2], 0)
            return tens * 10 + units
        elif len(part) == 1:
            return digit_map.get(part[0], 0)
        else:
            # Fallback to simple digit conversion
            result = 0
            for char in part:
                if char in digit_map:
                    result = result * 10 + digit_map[char]
            return result
    
    def _convert_tamil_digit_sequence(self, text):
        """Convert a sequence of Tamil digits to Arabic digits - FIXED FOR COLON HANDLING"""
        simple_map = {'௦': '0', '௧': '1', '௨': '2', '௩': '3', '௪': '4',
                    '௫': '5', '௬': '6', '௭': '7', '௮': '8', '௯': '9'}
        
        result = ""
        for char in text:
            if char in simple_map:
                result += simple_map[char]
            else:
                # CRITICAL FIX: Preserve important characters like colon, but filter out scale markers
                if char == ':':
                    result += ':'  # Explicitly preserve colon
                elif char not in ['௰', '௱', '௲']:  # Don't include scale markers in digit sequences
                    # For any other character, check if it's a valid ASCII/Unicode character
                    if ord(char) < 128 or char.isprintable():
                        result += char
                    # Skip invalid or corrupted characters
        
        return result
    
    def _convert_compound_number(self, match):
        """Convert compound Tamil numbers like 'இருபத்தி ஐந்து' → '25' - ENHANCED"""
        base_part = match.group(1)
        unit_part = match.group(2)
        
        if self.config.enable_debug_logging:
            logger.debug(f" Converting compound number: '{base_part} {unit_part}'")
            logger.debug(f" Base part: '{base_part}', Unit part: '{unit_part}'")
        
        # Check if both parts are in our mappings
        base_value = None
        unit_value = None
        
        if base_part in self.compound_base_map:
            base_value = self.compound_base_map[base_part]
            if self.config.enable_debug_logging:
                logger.debug(f" Base value: {base_value}")
        
        if unit_part in self.compound_unit_map:
            unit_value = self.compound_unit_map[unit_part]
            if self.config.enable_debug_logging:
                logger.debug(f" Unit value: {unit_value}")
        elif unit_part in self.static_word_mappings:
            unit_value = self.static_word_mappings[unit_part]
            if self.config.enable_debug_logging:
                logger.debug(f" Unit value from static: {unit_value}")
        
        if base_value is not None and unit_value is not None:
            result = base_value + unit_value
            if self.config.enable_debug_logging:
                logger.debug(f" Compound calculation: {base_value} + {unit_value} = {result}")
            return str(result)
        
        if self.config.enable_debug_logging:
            logger.debug(f" Compound number conversion failed")
            logger.debug(f" base_part='{base_part}' in mapping: {base_part in self.compound_base_map}")
            logger.debug(f" unit_part='{unit_part}' in mapping: {unit_part in self.compound_unit_map}")
        
        return match.group(0)

    def _convert_scale_number(self, match):
            """Convert scale-based numbers like 'ஐந்து நூறு' → '500' - COMPLETELY FIXED"""
            number_text = match.group(1)
            scale_text = match.group(2)
            
            if self.config.enable_debug_logging:
                logger.debug(f" Converting scale number: '{number_text} {scale_text}'")
            
            # Convert the number part - ENHANCED to handle all cases including ஒரு
            number_value = None
            
            # CRITICAL FIX: Handle "ஒரு" specifically
            if number_text == 'ஒரு':
                number_value = 1
                if self.config.enable_debug_logging:
                    logger.debug(f" Number 'ஒரு' converted to: {number_value}")
            
            # Check static mappings (most comprehensive)
            elif number_text in self.static_word_mappings:
                mapped_value = self.static_word_mappings[number_text]
                if isinstance(mapped_value, (int, float)):
                    number_value = mapped_value
                    if self.config.enable_debug_logging:
                        logger.debug(f" Number from static mappings: '{number_text}' → {number_value}")
            
            # Then check compound unit mappings
            elif number_text in self.compound_unit_map:
                number_value = self.compound_unit_map[number_text]
                if self.config.enable_debug_logging:
                    logger.debug(f" Number from compound unit: '{number_text}' → {number_value}")
            
            # Then check compound base mappings
            elif number_text in self.compound_base_map:
                number_value = self.compound_base_map[number_text]
                if self.config.enable_debug_logging:
                    logger.debug(f" Number from compound base: '{number_text}' → {number_value}")
            
            # Finally try the generic converter
            else:
                number_value = self._convert_tamil_word_to_number(number_text)
                if self.config.enable_debug_logging:
                    logger.debug(f" Number from generic converter: '{number_text}' → {number_value}")
            
            # Get scale multiplier - ENHANCED with more scale words
            scale_map = {
                'நூறு': 100, 'ஆயிரம்': 1000, 'லட்சம்': 100000, 
                'இலட்சம்': 100000, 'கோடி': 10000000, 
                'அர்புதம்': 1000000000, 'நீலம்': 100000000000
            }
            
            # Validate both number and scale
            if isinstance(number_value, (int, float)) and scale_text in scale_map:
                result = int(number_value * scale_map[scale_text])
                if self.config.enable_debug_logging:
                    logger.debug(f" Scale calculation: {number_value} × {scale_map[scale_text]} = {result}")
                    logger.debug(f" Scale number result: '{match.group(0)}' → '{result}'")
                return str(result)
            
            # Enhanced debugging for failures
            if self.config.enable_debug_logging:
                logger.debug(f" Scale number conversion failed:")
                print(f"  number_text='{number_text}', number_value={number_value} (type: {type(number_value)})")
                print(f"  scale_text='{scale_text}', in scale_map={scale_text in scale_map}")
                if scale_text in scale_map:
                    print(f"  scale_value={scale_map[scale_text]}")
            
            return match.group(0)
    
    def _convert_complex_number(self, match):
        """Convert complex Tamil number expressions with scale words"""
        prefix_text = match.group(1).strip() if match.group(1) else ""
        scale_word = match.group(2)
        suffix_text = match.group(3).strip() if match.group(3) else ""
        
        full_expression = f"{prefix_text} {scale_word}"
        if suffix_text:
            full_expression += f" {suffix_text}"
        
        result = self._convert_tamil_text_to_number(full_expression.strip())
        
        if isinstance(result, (int, float)):
            return str(result)
        
        return match.group(0)
    
    def _convert_tamil_word_to_number(self, tamil_word):
        """Convert a single Tamil word to number - SAFE version"""
        
        if not tamil_word:
            return tamil_word
        
        tamil_word = tamil_word.strip()
        
        # Check static mappings first (safe words only)
        if tamil_word in self.static_word_mappings:
            return self.static_word_mappings[tamil_word]
        
        # Check compound mappings
        if tamil_word in self.compound_base_map:
            return self.compound_base_map[tamil_word]
        
        if tamil_word in self.compound_unit_map:
            return self.compound_unit_map[tamil_word]
        
        # Try to reverse-engineer the number
        reverse_result = self._reverse_engineer_number(tamil_word)
        if isinstance(reverse_result, (int, float)):
            return reverse_result
        
        # If all else fails, return the original word
        return tamil_word
    
    def _convert_tamil_text_to_number(self, tamil_text):
        """Convert complex Tamil text to number by parsing components"""
        
        parts = re.split(r'\s+', tamil_text.strip())
        
        total = 0
        current_number = 0
        
        for part in parts:
            if not part:
                continue
                
            if part in self.static_word_mappings:
                value = self.static_word_mappings[part]
            elif part in self.compound_base_map:
                value = self.compound_base_map[part]
            elif part in self.compound_unit_map:
                value = self.compound_unit_map[part]
            else:
                value = self._reverse_engineer_number(part)
            
            if isinstance(value, (int, float)):
                if value >= 100:  # Scale word
                    if current_number == 0:
                        current_number = 1
                    current_number *= value
                else:  # Regular number
                    current_number += value
            else:
                return tamil_text
        
        total += current_number
        return total if total > 0 else tamil_text
    
    def _reverse_engineer_number(self, tamil_word):
        """Attempt to reverse-engineer a Tamil word to find its numeric value"""
        
        # Try tens patterns: "இருபது" → 20
        tens_patterns = {
            'இருப': 20, 'முப்': 30, 'நாற்': 40, 'ஐம்': 50,
            'அறுப': 60, 'எழுப': 70, 'எண்': 80, 'தொண்': 90
        }
        
        for pattern, value in tens_patterns.items():
            if tamil_word.startswith(pattern):
                return value
        
        # Try compound patterns: "இருபத்தி" suggests 20+
        if 'த்தி' in tamil_word:
            parts = tamil_word.split('த்தி')
            if len(parts) == 2:
                base = self._reverse_engineer_number(parts[0] + 'து')
                addition = self._reverse_engineer_number(parts[1])
                if isinstance(base, int) and isinstance(addition, int):
                    return base + addition
        
        return tamil_word
    
    def _fallback_converter(self, n):
        """Dynamic fallback converter - generates Tamil words on-demand"""
        
        if not isinstance(n, int) or n < 0:
            return str(n)
        
        if self.cache_enabled and n in self.conversion_cache:
            return self.conversion_cache[n]
        
        if n == 0:
            result = 'சுழியம்'
        else:
            result = self._convert_number_to_tamil_words(n)
        
        if self.cache_enabled and len(self.conversion_cache) < self.max_cache_size:
            self.conversion_cache[n] = result
        
        return result
    
    def _create_tamil_word_boundary_pattern(self, words_list):
        """Create a regex pattern with proper Tamil word boundaries"""
        # Escape all words for regex
        escaped_words = [re.escape(word) for word in words_list]
        
        # Tamil Unicode range: U+0B80–U+0BFF
        # Also include common English letters and numbers that might be adjacent
        boundary_chars = r'[a-zA-Z0-9\u0B80-\u0BFF]'
        
        # Create pattern with negative lookbehind and lookahead
        pattern = r'(?<!' + boundary_chars + r')(' + '|'.join(escaped_words) + r')(?!' + boundary_chars + r')'
        
        return re.compile(pattern)

    def _test_word_boundary_pattern(self, pattern, test_text, expected_matches):
        """Test if the word boundary pattern works correctly"""
        matches = pattern.findall(test_text)
        
        if self.config.enable_debug_logging:
            logger.debug(f" Testing pattern on: '{test_text}'")
            logger.debug(f" Found matches: {matches}")
            logger.debug(f" Expected matches: {expected_matches}")
            
            if set(matches) == set(expected_matches):
                logger.debug(f" ✅ Pattern test passed")
                return True
            else:
                logger.debug(f" ❌ Pattern test failed")
                return False
        
        return set(matches) == set(expected_matches)

    def _convert_number_to_tamil_words(self, n):
        """Convert any integer to Tamil words dynamically"""
        
        if n == 0:
            return 'சுழியம்'
        
        def get_ones_word(num):
            ones = ['', 'ஒன்று', 'இரண்டு', 'மூன்று', 'நான்கு', 'ஐந்து', 
                   'ஆறு', 'ஏழு', 'எட்டு', 'ஒன்பது']
            return ones[num] if 0 <= num < len(ones) else ''
        
        def get_teens_word(num):
            teens = ['பத்து', 'பதினொன்று', 'பன்னிரண்டு', 'பதின்மூன்று', 
                    'பதினான்கு', 'பதினைந்து', 'பதினாறு', 'பதினேழு', 
                    'பதினெட்டு', 'பத்தொன்பது']
            return teens[num - 10] if 10 <= num < 20 else ''
        
        def get_tens_word(num):
            tens = ['', '', 'இருபது', 'முப்பது', 'நாற்பது', 'ஐம்பது',
                   'அறுபது', 'எழுபது', 'எண்பது', 'தொண்ணூறு']
            return tens[num] if 0 <= num < len(tens) else ''
        
        def convert_below_hundred(num):
            if num == 0:
                return ''
            elif num < 10:
                return get_ones_word(num)
            elif num < 20:
                return get_teens_word(num)
            elif num < 100:
                tens_part = num // 10
                ones_part = num % 10
                tens_word = get_tens_word(tens_part)
                if ones_part == 0:
                    return tens_word
                else:
                    tens_prefix = tens_word.replace('து', 'த்தி')
                    ones_word = get_ones_word(ones_part)
                    return f"{tens_prefix} {ones_word}"
            return str(num)
        
        def convert_below_thousand(num):
            if num < 100:
                return convert_below_hundred(num)
            
            hundreds = num // 100
            remainder = num % 100
            
            hundreds_word = get_ones_word(hundreds)
            result = f"{hundreds_word} நூறு"
            
            if remainder > 0:
                remainder_word = convert_below_hundred(remainder)
                result += f" {remainder_word}"
            
            return result
        
        # Process large numbers using scale hierarchy
        parts = []
        remaining = n
        
        for scale_name, scale_value in self.scale_hierarchy:
            if remaining >= scale_value:
                scale_count = remaining // scale_value
                if scale_count > 0:
                    if scale_value >= 1000:
                        scale_part = convert_below_thousand(scale_count)
                    else:
                        scale_part = convert_below_hundred(scale_count)
                    
                    parts.append(f"{scale_part} {scale_name}")
                    remaining = remaining % scale_value
        
        if remaining > 0:
            parts.append(convert_below_thousand(remaining))
        
        return ' '.join(parts) if parts else 'சுழியம்'
    
    def convert_tamil_to_numbers(self, text: str) -> str:
        
        if not text or not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # CRITICAL: Check configuration flag
        if not self.config.convert_number_words_to_digits:
            if self.config.enable_debug_logging:
                logger.debug(" Number words to digits conversion DISABLED by configuration")
            return text
        
        try:
            result = text
            
            if self.config.enable_debug_logging:
                logger.debug(f" Converting Tamil to numbers: '{text}'")
            
            # Step 1: Always convert Tamil numerals (௧௨௩ → 123) - this is always safe
            pattern_0 = self.patterns[0][0]  # Tamil numerals pattern
            converter_0 = self.patterns[0][1]
            
            matches = list(pattern_0.finditer(result))
            if matches and self.config.enable_debug_logging:
                logger.debug(f" Found {len(matches)} Tamil numeral matches")
            
            for match in reversed(matches):  # Reverse to maintain positions
                start, end = match.span()
                converted = converter_0(match)
                result = result[:start] + converted + result[end:]
                if self.config.enable_debug_logging:
                    print(f"  - Converted numeral: '{match.group(0)}' → '{converted}'")
            
            # Step 2: Convert compound numbers (இருபத்தி ஐந்து → 25)
            pattern_1 = self.patterns[1][0]
            converter_1 = self.patterns[1][1]
            
            matches = list(pattern_1.finditer(result))
            if matches and self.config.enable_debug_logging:
                logger.debug(f" Found {len(matches)} compound number matches")
            
            for match in reversed(matches):
                start, end = match.span()
                converted = converter_1(match)
                result = result[:start] + converted + result[end:]
                if self.config.enable_debug_logging:
                    print(f"  - Converted compound: '{match.group(0)}' → '{converted}'")
            
            # Step 3: Convert scale numbers (ஐந்து நூறு → 500) - ENHANCED AND FIXED
            pattern_2 = self.patterns[2][0]
            converter_2 = self.patterns[2][1]
            
            matches = list(pattern_2.finditer(result))
            if matches and self.config.enable_debug_logging:
                logger.debug(f" Found {len(matches)} scale number matches")
            
            for match in reversed(matches):
                start, end = match.span()
                converted = converter_2(match)
                if converted != match.group(0):  # Only replace if conversion was successful
                    result = result[:start] + converted + result[end:]
                    if self.config.enable_debug_logging:
                        print(f"  - Converted scale: '{match.group(0)}' → '{converted}'")
            
            # Step 4: Convert simple numbers (ஒன்று → 1) - COMPLETELY REWRITTEN
            # Use a more comprehensive approach that handles all individual words
            
            # Create a list of all individual words that should be converted
            # Sort by length (longest first) to avoid partial matches
            all_individual_words = []
            
            # Add basic numbers
            basic_numbers = ['ஒன்று', 'இரண்டு', 'மூன்று', 'நான்கு', 'ஐந்து', 
                            'ஆறு', 'ஏழு', 'எட்டு', 'ஒன்பது', 'பத்து']
            all_individual_words.extend(basic_numbers)
            
            # Add teens
            teens = ['பதினொன்று', 'பன்னிரண்டு', 'பதிரண்டு', 'பதின்மூன்று', 
                    'பதினான்கு', 'பதினைந்து', 'பதினாறு', 'பதினேழு', 
                    'பதினெட்டு', 'பத்தொன்பது']
            all_individual_words.extend(teens)
            
            # Add tens
            tens = ['இருபது', 'முப்பது', 'நாற்பது', 'ஐம்பது',
                'அறுபது', 'எழுபது', 'எண்பது', 'தொண்ணூறு']
            all_individual_words.extend(tens)
            
            # Add scale words
            scales = ['நூறு', 'ஆயிரம்', 'லட்சம்', 'இலட்சம்', 'கோடி']
            all_individual_words.extend(scales)
            
            # Add compound scale words (CRITICAL: These were missing!)
            compound_scales = ['ஐநூறு', 'இருநூறு', 'மூன்றூறு', 'நான்நூறு', 
                            'ஆறூறு', 'எழுநூறு', 'எண்ணூறு', 'தொள்ளாயிரம்']
            all_individual_words.extend(compound_scales)
            
            # Sort by length (longest first) to handle overlapping words correctly
            all_individual_words.sort(key=len, reverse=True)
            
            # Apply conversions for each word
            for word in all_individual_words:
                if word in result:
                    # Check if this word exists in our mappings
                    if word in self.static_word_mappings:
                        value = self.static_word_mappings[word]
                        if isinstance(value, (int, float)):
                            # Create a regex pattern with proper word boundaries
                            # Tamil Unicode range: \u0B80-\u0BFF
                            pattern = r'(?<![a-zA-Z0-9\u0B80-\u0BFF])' + re.escape(word) + r'(?![a-zA-Z0-9\u0B80-\u0BFF])'
                            
                            def replace_func(match):
                                return str(value)
                            
                            new_result = re.sub(pattern, replace_func, result)
                            if new_result != result:
                                if self.config.enable_debug_logging:
                                    print(f"  - Converted individual word: '{word}' → '{value}'")
                                result = new_result
            
            if self.config.enable_debug_logging:
                logger.debug(f" Final result: '{result}'")
            
            return result
            
        except Exception as e:
            logger.debug(f" Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return text


    def convert_number_to_tamil(self, number) -> str:
        """Convert any number to Tamil words"""
        
        try:
            if isinstance(number, str):
                if '/' in number:  # Fraction
                    parts = number.split('/')
                    if len(parts) == 2:
                        num_word = self.tamil_converter(int(parts[0]))
                        denom_word = self.tamil_converter(int(parts[1]))
                        return f"{denom_word}ல் {num_word}"
                
                if '.' in number:  # Decimal
                    parts = number.split('.')
                    if len(parts) == 2:
                        int_word = self.tamil_converter(int(parts[0]))
                        decimal_words = [self.tamil_converter(int(d)) for d in parts[1]]
                        return f"{int_word} புள்ளி {' '.join(decimal_words)}"
                
                if number.endswith('%'):  # Percentage
                    num_word = self.tamil_converter(int(number[:-1]))
                    return f"{num_word} சதவீதம்"
                
                number = int(float(number))
            
            return self.tamil_converter(number)
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Conversion failed for {number}: {e}")
            return str(number)

class TamilCurrencyProcessor:
    """Process Tamil currency formats - COMPLETELY FIXED VERSION"""
    
    def __init__(self, config: TamilASRConfig):
        self.config = config
        self._compile_patterns()
        self._setup_number_mappings()
    
    def _setup_number_mappings(self):
        """Setup number mappings for currency conversion"""
        self.number_word_mappings = {
            'ஒரு': 1, 'ஒன்று': 1, 'இரண்டு': 2, 'மூன்று': 3, 'நான்கு': 4, 'ஐந்து': 5,
            'ஆறு': 6, 'ஏழு': 7, 'எட்டு': 8, 'ஒன்பது': 9, 'பத்து': 10,
            'இருபது': 20, 'முப்பது': 30, 'நாற்பது': 40, 'ஐம்பது': 50,
            'அறுபது': 60, 'எழுபது': 70, 'எண்பது': 80, 'தொண்ணூறு': 90,
            'நூறு': 100, 'ஆயிரம்': 1000, 'லட்சம்': 100000, 'இலட்சம்': 100000, 'கோடி': 10000000
        }
    
    def _convert_tamil_numerals_to_digits(self, tamil_num):
        """Convert Tamil numerals to Arabic digits - ENHANCED VERSION"""
        if not tamil_num:
            return "0"
            
        simple_map = {'௦': '0', '௧': '1', '௨': '2', '௩': '3', '௪': '4',
                     '௫': '5', '௬': '6', '௭': '7', '௮': '8', '௯': '9'}
        
        # Handle complex Tamil numerals first
        if len(tamil_num) > 1 and any(c in tamil_num for c in ['௰', '௱', '௲']):
            try:
                # Use the same logic as the number converter for complex numerals
                total = 0
                current = 0
                
                digit_map = {'௦': 0, '௧': 1, '௨': 2, '௩': 3, '௪': 4,
                           '௫': 5, '௬': 6, '௭': 7, '௮': 8, '௯': 9}
                scale_map = {'௰': 10, '௱': 100, '௲': 1000}
                
                for char in tamil_num:
                    if char in digit_map:
                        current = digit_map[char]
                    elif char in scale_map:
                        if current == 0:
                            current = 1
                        total += current * scale_map[char]
                        current = 0
                
                total += current
                return str(total)
            except:
                pass
        
        # Simple digit-by-digit conversion
        result = ""
        for char in tamil_num:
            if char in simple_map:
                result += simple_map[char]
            else:
                result += char
        
        return result if result else "0"
    
    def _compile_patterns(self):
        """Compile currency patterns - FIXED VERSION WITH BETTER PATTERN MATCHING"""
        self.patterns = [
            # CRITICAL FIX 1: Tamil rupee symbol ௹ with ANY Tamil numerals (including complex ones)
            (re.compile(r'௹([௦-௯௰௱௲]+)'), self._convert_tamil_rupee_symbol),
            
            # CRITICAL FIX 2: Handle standalone Tamil rupee symbol with Arabic digits  
            (re.compile(r'௹(\d+(?:\.\d+)?)'), self._convert_rupee_symbol_arabic),
            
            # Regular rupee symbol with digits
            (re.compile(r'₹(\d+(?:\.\d+)?)'), r'\1 ரூபாய்'),
            
            # CRITICAL FIX 3: "ஒரு + scale + ரூபாய்" patterns - more comprehensive
            (re.compile(r'ஒரு\s+(கோடி|லட்சம்|இலட்சம்|ஆயிரம்|நூறு)\s*ரூபாய்'), self._convert_oru_scale_currency),
            
            # Number words + scale + currency - ENHANCED
            (re.compile(r'(ஒன்று|இரண்டு|மூன்று|நான்கு|ஐந்து|ஆறு|ஏழு|எட்டு|ஒன்பது|பத்து|இருபது|முப்பது|நாற்பது|ஐம்பது|அறுபது|எழுபது|எண்பது|தொண்ணூறு|நூறு)\s+(லட்சம்|இலட்சம்|கோடி|ஆயிரம்|நூறு)\s*ரூபாய்'), self._convert_number_scale_currency),
            
            # NEW: Handle compound numbers + scale + currency (like "இருபத்தி ஐந்து ஆயிரம் ரூபாய்")
            (re.compile(r'(இருபத்தி\s+\w+|முப்பத்தி\s+\w+|நாற்பத்தி\s+\w+|ஐம்பத்தி\s+\w+|அறுபத்தி\s+\w+|எழுபத்தி\s+\w+|எண்பத்தி\s+\w+|தொண்ணூற்றி\s+\w+)\s+(லட்சம்|இலட்சம்|கோடி|ஆயிரம்|நூறு)\s*ரூபாய்'), self._convert_compound_scale_currency),
            
            # Clean up patterns - more robust
            (re.compile(r'(\d+(?:\.\d+)?)\s*ரூபாய்கள்'), r'\1 ரூபாய்'),
            (re.compile(r'(\d+(?:\.\d+)?)\s*ரூபாய்\s*ரூபாய்'), r'\1 ரூபாய்'),
            (re.compile(r'(\d+(?:\.\d+)?)\s+ரூபாய்'), r'\1 ரூபாய்'),  # Normalize spacing
        ]
    
    def _convert_tamil_rupee_symbol(self, match):
        """Convert Tamil rupee symbol with Tamil numerals - COMPLETELY FIXED"""
        tamil_numerals = match.group(1)
        
        if self.config.enable_debug_logging:
            logger.debug(f" Converting Tamil rupee symbol: ௹{tamil_numerals}")
        
        # Convert Tamil numerals to Arabic digits
        arabic_digits = self._convert_tamil_numerals_to_digits(tamil_numerals)
        
        result = f"{arabic_digits} ரூபாய்"
        
        if self.config.enable_debug_logging:
            logger.debug(f" Tamil rupee result: ௹{tamil_numerals} → {result}")
        
        return result
    
    def _convert_rupee_symbol_arabic(self, match):
        """Convert Tamil rupee symbol with Arabic digits - NEW METHOD"""
        arabic_digits = match.group(1)
        result = f"{arabic_digits} ரூபாய்"
        
        if self.config.enable_debug_logging:
            logger.debug(f" Arabic rupee result: ௹{arabic_digits} → {result}")
        
        return result
    
    def _convert_oru_scale_currency(self, match):
        """Convert 'ஒரு + scale + ரூபாய்' to proper currency format - ENHANCED"""
        scale_word = match.group(1)
        scale_map = {'நூறு': 100, 'ஆயிரம்': 1000, 'லட்சம்': 100000, 'இலட்சம்': 100000, 'கோடி': 10000000}
        
        if scale_word in scale_map:
            scale_value = scale_map[scale_word]
            result = f"{scale_value} ரூபாய்"
            
            if self.config.enable_debug_logging:
                logger.debug(f" Oru scale currency: ஒரு {scale_word} ரூபாய் → {result}")
            
            return result
        
        return match.group(0)  # Return original if no match
    
    def _convert_number_scale_currency(self, match):
        """Convert 'number + scale + ரூபாய்' to proper currency format - ENHANCED"""
        number_word = match.group(1)
        scale_word = match.group(2)
        
        # Get number value
        number_value = self.number_word_mappings.get(number_word, 1)
        
        # Get scale value
        scale_map = {'நூறு': 100, 'ஆயிரம்': 1000, 'லட்சம்': 100000, 'இலட்சம்': 100000, 'கோடி': 10000000}
        scale_value = scale_map.get(scale_word, 1)
        
        if isinstance(number_value, (int, float)) and scale_word in scale_map:
            total = int(number_value * scale_value)
            result = f"{total} ரூபாய்"
            
            if self.config.enable_debug_logging:
                logger.debug(f" Number scale currency: {number_word} {scale_word} ரூபாய் → {result}")
            
            return result
        
        return match.group(0)  # Return original if conversion fails
    
    def _convert_compound_scale_currency(self, match):
        """Convert compound number + scale + currency - NEW METHOD"""
        compound_part = match.group(1).strip()
        scale_word = match.group(2)
        
        if self.config.enable_debug_logging:
            logger.debug(f" Converting compound scale currency: {compound_part} {scale_word} ரூபாய்")
        
        # Parse compound number (like "இருபத்தி ஐந்து")
        compound_value = self._parse_compound_number(compound_part)
        
        # Get scale value
        scale_map = {'நூறு': 100, 'ஆயிரம்': 1000, 'லட்சம்': 100000, 'இலட்சம்': 100000, 'கோடி': 10000000}
        scale_value = scale_map.get(scale_word, 1)
        
        if isinstance(compound_value, (int, float)) and scale_word in scale_map:
            total = int(compound_value * scale_value)
            result = f"{total} ரூபாய்"
            
            if self.config.enable_debug_logging:
                logger.debug(f" Compound scale result: {compound_part} {scale_word} → {result}")
            
            return result
        
        return match.group(0)
    
    def _parse_compound_number(self, compound_text):
        """Parse compound numbers like 'இருபத்தி ஐந்து' → 25"""
        try:
            # Split by space to get base and unit
            parts = compound_text.split()
            if len(parts) == 2:
                base_part = parts[0]
                unit_part = parts[1]
                
                # Base mapping for compound numbers
                base_map = {
                    'இருபத்தி': 20, 'முப்பத்தி': 30, 'நாற்பத்தி': 40, 'ஐம்பத்தி': 50,
                    'அறுபத்தி': 60, 'எழுபத்தி': 70, 'எண்பத்தி': 80, 'தொண்ணூற்றி': 90
                }
                
                # Unit mapping
                unit_map = {
                    'ஒன்று': 1, 'இரண்டு': 2, 'மூன்று': 3, 'நான்கு': 4, 'ஐந்து': 5,
                    'ஆறு': 6, 'ஏழு': 7, 'எட்டு': 8, 'ஒன்பது': 9
                }
                
                base_value = base_map.get(base_part, 0)
                unit_value = unit_map.get(unit_part, 0)
                
                if base_value > 0 and unit_value > 0:
                    return base_value + unit_value
            
            # If not compound, try to get single number value
            return self.number_word_mappings.get(compound_text.strip(), compound_text)
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f" Compound parsing failed for '{compound_text}': {e}")
            return compound_text
    
    def normalize_currency(self, text: str) -> str:
        """Normalize Tamil currency formats - ENHANCED VERSION"""
        if not self.config.normalize_currency or not text:
            return text or ""
        
        try:
            original = text
            
            if self.config.enable_debug_logging:
                logger.debug(f" Currency normalization input: '{text}'")
            
            # Apply all patterns in order of priority
            for i, (pattern, converter) in enumerate(self.patterns):
                old_text = text
                if callable(converter):
                    text = pattern.sub(converter, text)
                else:
                    text = pattern.sub(converter, text)
                
                if old_text != text and self.config.enable_debug_logging:
                    logger.debug(f" Pattern {i+1} applied: '{old_text}' → '{text}'")
            
            if self.config.enable_debug_logging and original != text:
                logger.debug(f" Final currency normalization: '{original}' → '{text}'")
            
            return text
            
        except Exception as e:
            logger.warning(f"Currency normalization failed: {e}")
            return text

class TamilPhoneProcessor:
    """Process Tamil phone number formats - FIXED"""
    
    def __init__(self, config: TamilASRConfig):
        self.config = config
        self._setup_digit_mapping()
        self._compile_patterns()
    
    def _setup_digit_mapping(self):
        """Setup digit mapping for phone numbers"""
        # Tamil digits for phone numbers
        self.digit_map = {
            '0': 'சுழியம்', '1': 'ஒன்று', '2': 'இரண்டு', '3': 'மூன்று', '4': 'நான்கு',
            '5': 'ஐந்து', '6': 'ஆறு', '7': 'ஏழு', '8': 'எட்டு', '9': 'ஒன்பது'
        }
    
    def _compile_patterns(self):
        """Compile phone number patterns - ENHANCED TO HANDLE LANDLINES"""
        self.patterns = [
            # +91 international format
            (re.compile(r'\+91[\s-]*(\d{10})\b'), self._convert_international_continuous),
            (re.compile(r'\+91[\s-]*(\d{3})[\s-]+(\d{3})[\s-]+(\d{4})'), self._convert_international_formatted),
            
            # Indian mobile numbers - only match valid mobile numbers
            (re.compile(r'\b([6-9]\d{9})\b'), self._convert_mobile),
            
            # ENHANCED: Handle landline numbers (11 digits starting with 0)
            (re.compile(r'\b(0\d{10})\b'), self._convert_landline),
            
            # Formatted Indian numbers
            (re.compile(r'\b(\d{3})[\s-]+(\d{3})[\s-]+(\d{4})\b'), self._convert_formatted),
            
            # Tamil context phone numbers
            (re.compile(r'(தொலைபேசி|போன்|மொபைல்|தொலைபேசி எண்)[\s:]*(\d{10,11})'), self._convert_with_context),
        ]

    def _convert_landline(self, match):
        """NEW: Convert landline numbers to Tamil words"""
        digits = match.group(1)
        if len(digits) == 11 and digits[0] == '0':
            return self._digits_to_tamil(digits)
        return match.group(0)

    
    def _digits_to_tamil(self, digits: str) -> str:
        """Convert digits to Tamil words"""
        result = []
        for digit in digits:
            if digit.isdigit():
                result.append(self.digit_map.get(digit, digit))
            else:
                result.append(digit)
        return ' '.join(result)
    
    def _convert_international_continuous(self, match):
        """Convert international format +91xxxxxxxxxx - FIXED"""
        digits = match.group(1)
        # Convert all digits including country code
        full_number = '91' + digits
        return self._digits_to_tamil(full_number)
    
    def _convert_international_formatted(self, match):
        """Convert international formatted +91-xxx-xxx-xxxx - FIXED"""
        digit1, digit2, digit3 = match.group(1), match.group(2), match.group(3)
        all_digits = '91' + digit1 + digit2 + digit3
        return self._digits_to_tamil(all_digits)
    
    def _convert_mobile(self, match):
        """Convert mobile number to Tamil words"""
        digits = match.group(1) if len(match.groups()) > 0 else match.group(0)
        # Only convert if it's a valid mobile number
        if len(digits) == 10 and digits[0] in '6789':
            return self._digits_to_tamil(digits)
        return match.group(0)  # Return unchanged if not valid mobile
    
    def _convert_formatted(self, match):
        """Convert formatted number to Tamil words"""
        if len(match.groups()) == 3:
            digits = match.group(1) + match.group(2) + match.group(3)
            # Check if it's a valid 10-digit mobile number
            if len(digits) == 10 and digits[0] in '6789':
                return self._digits_to_tamil(digits)
        return match.group(0)  # Return unchanged if not valid
    
    def _convert_with_context(self, match):
        """Convert phone number with Tamil context"""
        context = match.group(1)
        digits = match.group(2)
        converted = self._digits_to_tamil(digits)
        return f"{context} {converted}"
    
    def normalize_phones(self, text: str) -> str:
        """Normalize phone numbers to Tamil words - IMPROVED"""
        if not self.config.normalize_phone_numbers or not text:
            return text or ""
        
        try:
            original = text
            
            for pattern, converter in self.patterns:
                text = pattern.sub(converter, text)
            
            if self.config.enable_debug_logging and original != text:
                logger.debug(f"Phone normalization: '{original}' → '{text}'")
            
            return text
            
        except Exception as e:
            logger.warning(f"Phone normalization failed: {e}")
            return text


class TamilFillerProcessor:
    def __init__(self, config: TamilASRConfig):
        self.config = config
        self._setup_filler_lists()
    
    def _setup_filler_lists(self):
        self.safe_fillers = []
        
        # Hesitation sounds that are safe to remove
        self.hesitation_sounds = [
            'அம்ம்', 'ஏம்ம்', 'ஓம்ம்', 'உம்ம்', 'ஹ்ம்ம்'
        ]
        
        # Repeated patterns
        self.repeated_patterns = [
            ('அது அது', 'அது'), ('இது இது', 'இது'),  # Keep one instance
            ('அப்படி அப்படி', 'அப்படி'), ('அப்புறம் அப்புறம்', 'அப்புறம்'),
            ('என்ன என்ன', 'என்ன'), ('ஆனா ஆனா', 'ஆனா'),
        ]
        
        # Context-sensitive words that need special handling
        '''
        self.context_sensitive_words = {
            'என்ன': 'interrogative',  # Can mean "what" in questions
            'இல்லை': 'negation',     # Critical negation word
            'ஆனா': 'conjunction',     # Important conjunction
            'அது': 'demonstrative',   # Can be important demonstrative
            'இது': 'demonstrative',   # Can be important demonstrative
            'அப்படி': 'manner',       # Can indicate manner/method
            'அப்புறம்': 'temporal'    # Can indicate time sequence
        }
        '''
    
    def _enhanced_filler_removal(self, text: str, filler: str) -> str:
        if not filler or filler not in text:
            return text
        
        original_text = text
        
        # SPECIAL HANDLING for specific fillers
        if filler == 'என்ன':
            # Protect "என்னா" and other variations
            for protected in ['என்னா', 'என்னோட', 'என்னால', 'என்னுடைய']:
                if protected in text:
                    # Replace protected word with placeholder
                    text = text.replace(protected, f'__PROTECTED_{protected}__')
        
        # Pattern 1: Remove filler at end of sentence (before period, end of text, or ellipsis)
        text = re.sub(r'\s+' + re.escape(filler) + r'\s*\.$', '.', text)
        text = re.sub(r'\s+' + re.escape(filler) + r'$', '', text)
        text = re.sub(r'\s+' + re.escape(filler) + r'\s*\.\.\.$', '...', text)
        
        # Pattern 2: Remove filler at start of sentence
        text = re.sub(r'^' + re.escape(filler) + r'\s+', '', text)
        text = re.sub(r'^' + re.escape(filler) + r'\.\.\.\s*', '', text)
        
        # Pattern 3: Remove filler surrounded by spaces or punctuation
        text = re.sub(r'\s+' + re.escape(filler) + r'\s+', ' ', text)
        text = re.sub(r'\.\.\.\s*' + re.escape(filler) + r'\s+', '... ', text) 
        text = re.sub(r'\s+' + re.escape(filler) + r'\s*\.\.\.\s*', ' ', text)
        
        # Pattern 4: Remove filler with any punctuation
        punctuation_marks = ['.', ',', '!', '?', ':', ';', '।', '॥']
        for punct in punctuation_marks:
            # Before punctuation
            text = re.sub(r'\s+' + re.escape(filler) + r'\s*' + re.escape(punct), punct, text)
            # After punctuation
            text = re.sub(re.escape(punct) + r'\s*' + re.escape(filler) + r'\s+', punct + ' ', text)
            # Standalone after punctuation
            text = re.sub(re.escape(punct) + r'\s*' + re.escape(filler) + r'$', punct, text)
        
        # Pattern 5: Special handling for sentence-ending fillers with period
        # This catches cases like "கொடுங்கள் என்ன." → "கொடுங்கள்"
        if text.endswith(' ' + filler + '.'):
            text = text[:-len(filler)-2] + '.'
        elif text.endswith(' ' + filler):
            text = text[:-len(filler)-1]
        elif text.endswith(filler + '.'):
            text = text[:-len(filler)-1] + '.'
        elif text.endswith(filler):
            if len(text) > len(filler) and text[-len(filler)-1:-len(filler)] == ' ':
                text = text[:-len(filler)].rstrip()
        
        # Restore protected words
        if filler == 'என்ன':
            for protected in ['என்னா', 'என்னோட', 'என்னால', 'என்னுடைய']:
                text = text.replace(f'__PROTECTED_{protected}__', protected)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        if self.config.enable_debug_logging and original_text != text:
            logger.debug(f"Enhanced filler removal '{filler}': '{original_text}' → '{text}'")
        
        return text


    # Remove this method
    def _context_aware_filler_removal(self, text: str, filler: str) -> str:
        """Context-aware filler removal for sensitive words like 'என்ன'"""
        if not filler or filler not in text:
            return text
        
        # Special handling for interrogative words like "என்ன"
        if filler == "என்ன":
            # Patterns where "என்ன" should be preserved
            preserve_patterns = [
                r'பெயர்\s+என்ன',      # "name what" = "what is the name"
                r'விலை\s+என்ன',       # "price what" = "what is the price"
                r'நேரம்\s+என்ன',      # "time what" = "what is the time"
                r'வயது\s+என்ன',       # "age what" = "what is the age"
                r'எண்\s+என்ன',        # "number what" = "what is the number"
                r'காரணம்\s+என்ன',     # "reason what" = "what is the reason"
                r'\w+\s+என்ன\s*[?]',   # Any word + என்ன + ? (question mark)
                r'\w+\s+என்ன$',        # Any word + என்ன at end of text
            ]
            
            # Check if any preserve pattern matches
            for pattern in preserve_patterns:
                if re.search(pattern, text):
                    # This is an interrogative use, don't remove it
                    return text
            
            # If it's at the end of a sentence after punctuation removal, check more carefully
            # Split text into segments and check each one
            segments = re.split(r'[.!?।॥]', text)
            new_segments = []
            
            for segment in segments:
                segment = segment.strip()
                if not segment:
                    continue
                    
                # Check if this segment ends with a pattern that needs என்ன
                should_preserve = False
                for pattern in preserve_patterns:
                    if re.search(pattern, segment):
                        should_preserve = True
                        break
                
                if should_preserve:
                    new_segments.append(segment)
                else:
                    # Apply normal filler removal
                    new_segments.append(self._enhanced_filler_removal(segment, filler))
            
            # Rejoin segments
            result = ' '.join(new_segments)
            return result
        
        # For other context-sensitive words, check if they should be preserved
        else:
            # Find all occurrences and check each one
            pattern = r'\b' + re.escape(filler) + r'\b'
            matches = list(re.finditer(pattern, text))
            
            # Process matches in reverse order to maintain positions
            for match in reversed(matches):
                start, end = match.span()
            
            # Clean up multiple spaces
            text = re.sub(r'\s+', ' ', text).strip()
            return text

    def remove_fillers(self, text: str) -> str:
        if not self.config.remove_filler_words or not text:
            return text or ""
        
        if not isinstance(text, str):
            text = str(text)
        
        original = text
        
        try:
            # Multiple passes to ensure thorough removal
            for iteration in range(5):  # Increased iterations
                old_text = text
                
                # Pass 1: Remove repeated patterns first
                for pattern, replacement in self.repeated_patterns:
                    text = text.replace(pattern, replacement)
                
                # Pass 2: Remove hesitation sounds
                for filler in self.hesitation_sounds:
                    text = self._enhanced_filler_removal(text, filler)
                
                # Pass 3: Remove main filler words with enhanced method
                filler_order = ['அது', 'இது', 'அதாவது', 'என்ன', 'அப்படி', 'அப்புறம்', 'ஆனா', 'இல்லை']
                for filler in filler_order:
                    if filler == 'என்ன':
                        preserve_patterns = [
                            r'பெயர்\s+என்ன',      # "name what" = "what is the name"
                            r'விலை\s+என்ன',       # "price what" = "what is the price"
                            r'நேரம்\s+என்ன',      # "time what" = "what is the time"
                            r'வயது\s+என்ன',       # "age what" = "what is the age"
                            r'எண்\s+என்ன',        # "number what" = "what is the number"
                            r'காரணம்\s+என்ன',     # "reason what" = "what is the reason"
                            r'\w+\s+என்ன\s*[?]',   # Any word + என்ன + ? (question mark)
                            r'\w+\s+என்ன$',        # Any word + என்ன at end of text
                        ]
                        
                        should_preserve = False
                        for pattern in preserve_patterns:
                            if re.search(pattern, text):
                                should_preserve = True
                                break
                        
                        if should_preserve:
                            continue
                    
                    # For all other cases (including non-interrogative "என்ன"), remove aggressively
                    text = self._enhanced_filler_removal(text, filler)
                
                # Pass 4: Clean up artifacts
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Remove any trailing/leading punctuation issues
                text = re.sub(r'^\s*[,.!?]+\s*', '', text)  # Remove leading punctuation
                text = re.sub(r'\s*\.\s*\.\s*\.\s*$', '', text)  # Remove trailing "..."
                text = re.sub(r'\s+\.', '.', text)  # Fix " ." → "."
                text = re.sub(r'\.\s*$', '', text)  # Remove trailing period if needed
                
                if text == old_text:
                    break
            
            # Final cleanup
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Final check for any remaining sentence-ending fillers
            for filler in self.safe_fillers:
                if text.endswith(' ' + filler):
                    text = text[:-len(filler)-1].rstrip()
            
            if self.config.enable_debug_logging and original != text:
                logger.debug(f"Enhanced filler removal ({iteration+1} iterations): '{original}' → '{text}'")
            
            return text
            
        except Exception as e:
            logger.warning(f"Filler removal failed: {e}")
            return original


class TamilTimeProcessor:
    def __init__(self, config: TamilASRConfig):
        self.config = config
        self._compile_patterns()
    
    def _compile_patterns(self):
        self.time_patterns = [
            # PRIORITY 1: Handle Tamil numerals with colon in time format FIRST (convert to structured format)
            (re.compile(r'([௦-௯௰௱௲]+):([௦-௯௰௱௲]+)\s*மணிக்கு'), self._convert_tamil_time_with_suffix),
            (re.compile(r'([௦-௯௰௱௲]+):([௦-௯௰௱௲]+)(?!\s*மணி)'), self._convert_tamil_time_simple),
            
            # PRIORITY 2: Handle Arabic numerals with colon (convert to structured format)
            (re.compile(r'(\d{1,2}):(\d{2})\s*மணிக்கு'), self._convert_time_with_suffix_preserve),
            (re.compile(r'(\d{1,2}):(\d{2})(?!\s*மணி)'), self._convert_time_simple),
            
            # PRIORITY 3: Handle simple Tamil numeral time expressions WITHOUT colons (CONVERT மணிக்கு to மணி)
            (re.compile(r'([௦-௯௰௱௲]+)\s*மணிக்கு(?![a-zA-Z0-9\u0B80-\u0BFF])', re.UNICODE), self._convert_tamil_simple_time_to_mani),
            
            # PRIORITY 4: Handle simple Arabic numeral time expressions WITHOUT colons (PRESERVE மணிக்கு)
            (re.compile(r'(\d{1,2})\s*மணிக்கு(?![a-zA-Z0-9\u0B80-\u0BFF])', re.UNICODE), self._convert_arabic_simple_time_preserve_suffix),
            
            # PRIORITY 5: Handle existing மணி patterns - be more careful about context
            (re.compile(r'(\d{1,2})\s*மணி\s*(\d{1,2})\s*நிமிடம்'), lambda m: f"{m.group(1)} மணி {m.group(2)} நிமிடம்"),
            (re.compile(r'(\d{1,2}):(\d{2})\s*மணி(?!\s*நிமிடம்)'), self._convert_time_to_structured),
            
            # Date formats
            (re.compile(r'(\d{1,2})/(\d{1,2})/(\d{4})'), lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}"),
            (re.compile(r'(\d{1,2})-(\d{1,2})-(\d{4})'), lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}"),
        ]

    def _convert_tamil_simple_time_to_mani(self, match):
        """Convert simple Tamil numeral time like '௯ மணிக்கு' to '9 மணி' - CONVERT மணிக்கு to மணி"""
        tamil_hour = match.group(1)
        
        if self.config.enable_debug_logging:
            logger.debug(f" Converting Tamil numeral time (மணிக்கு→மணி): '{tamil_hour} மணிக்கு'")
        
        hour = self._convert_tamil_numerals_in_time(tamil_hour)
        
        result = f"{hour} மணி"
        
        if self.config.enable_debug_logging:
            logger.debug(f" Tamil numeral time result (மணிக்கு→மணி): '{result}'")
        
        return result

    def _convert_arabic_simple_time_preserve_suffix(self, match):
        """Convert simple Arabic numeral time like '10 மணிக்கு' to '10 மணிக்கு' - PRESERVE மணிக்கு WITH NORMALIZATION"""
        hour = match.group(1)
        
        if self.config.enable_debug_logging:
            logger.debug(f" Converting Arabic numeral time (preserve மணிக்கு): '{hour} மணிக்கு'")
        
        normalized_hour = hour
        if hour.isdigit():
            try:
                normalized_hour = str(int(hour))
            except (ValueError, TypeError, OverflowError):
                pass
        
        result = f"{normalized_hour} மணிக்கு"
        
        if self.config.enable_debug_logging:
            logger.debug(f" Arabic numeral time result (preserved): '{result}'")
        
        return result

    def _convert_tamil_numerals_in_time(self, tamil_text):
        if not tamil_text:
            return "0"
        
        if self.config.enable_debug_logging:
            logger.debug(f" Converting Tamil numerals in time: '{tamil_text}'")
        
        # Handle complex Tamil numerals (like ௯௰ for 90, etc.)
        if len(tamil_text) > 1 and any(c in tamil_text for c in ['௰', '௱', '௲']):
            try:
                # Handle simple cases like ௯௰ = 90
                if tamil_text == '௧௰':
                    return "10"
                elif tamil_text == '௨௰':
                    return "20"
                elif tamil_text == '௩௰':
                    return "30"
                elif tamil_text == '௪௰':
                    return "40"
                elif tamil_text == '௫௰':
                    return "50"
                elif tamil_text == '௬௰':
                    return "60"
                elif tamil_text == '௭௰':
                    return "70"
                elif tamil_text == '௮௰':
                    return "80"
                elif tamil_text == '௯௰':
                    return "90"
                
                # For complex patterns, try to parse manually
                total = 0
                current = 0
                
                digit_map = {'௦': 0, '௧': 1, '௨': 2, '௩': 3, '௪': 4,
                        '௫': 5, '௬': 6, '௭': 7, '௮': 8, '௯': 9}
                scale_map = {'௰': 10, '௱': 100, '௲': 1000}
                
                for char in tamil_text:
                    if char in digit_map:
                        current = digit_map[char]
                    elif char in scale_map:
                        if current == 0:
                            current = 1
                        total += current * scale_map[char]
                        current = 0
                
                total += current
                result = str(total)
                
                if self.config.enable_debug_logging:
                    logger.debug(f" Complex Tamil numeral conversion: '{tamil_text}' → '{result}'")
                
                return result
                
            except Exception as e:
                if self.config.enable_debug_logging:
                    logger.debug(f" Complex Tamil numeral parsing failed: {e}")
                pass
        
        # Simple Tamil digit conversion
        simple_map = {'௦': '0', '௧': '1', '௨': '2', '௩': '3', '௪': '4',
                    '௫': '5', '௬': '6', '௭': '7', '௮': '8', '௯': '9'}
        
        result = ""
        for char in tamil_text:
            if char in simple_map:
                result += simple_map[char]
            else:
                # Don't include scale markers in simple conversion
                if char not in ['௰', '௱', '௲']:
                    result += char
        
        if result:
            if result.isdigit():
                try:
                    # Convert to int to remove leading zeros, then back to string
                    normalized_int = int(result)
                    normalized_result = str(normalized_int)
                    if self.config.enable_debug_logging:
                        logger.debug(f" Normalized time digits: '{result}' → '{normalized_result}'")
                    result = normalized_result
                except (ValueError, TypeError, OverflowError):
                    # If conversion fails, keep the original result
                    if self.config.enable_debug_logging:
                        logger.debug(f" Failed to normalize '{result}', keeping original")
                    pass
            # Handle empty or whitespace-only result
            elif not result.strip():
                result = "0"
        else:
            # Handle None or empty result
            result = "0"
        
        # Final validation and return
        final_result = result if result and (result.isdigit() or result.replace('.', '').isdigit()) else tamil_text
        
        if self.config.enable_debug_logging:
            logger.debug(f" Simple Tamil numeral conversion: '{tamil_text}' → '{final_result}'")
        
        return final_result

    def _convert_tamil_time_with_suffix(self, match):
        tamil_hour, tamil_minute = match.groups()
        
        if self.config.enable_debug_logging:
            logger.debug(f" Converting Tamil time with suffix: '{tamil_hour}:{tamil_minute} மணிக்கு'")
        
        hour = self._convert_tamil_numerals_in_time(tamil_hour)
        minute = self._convert_tamil_numerals_in_time(tamil_minute)
        
        result = f"{hour} மணி {minute} நிமிடம்"
        
        if self.config.enable_debug_logging:
            logger.debug(f" Tamil time with suffix result: '{result}'")
        
        return result
    
    def _convert_tamil_time_simple(self, match):
        """Convert simple Tamil numeral time format - FIXED"""
        tamil_hour, tamil_minute = match.groups()
        
        if self.config.enable_debug_logging:
            logger.debug(f" Converting simple Tamil time: '{tamil_hour}:{tamil_minute}'")
        
        hour = self._convert_tamil_numerals_in_time(tamil_hour)
        minute = self._convert_tamil_numerals_in_time(tamil_minute)
        
        result = f"{hour} மணி {minute} நிமிடம்"
        
        if self.config.enable_debug_logging:
            logger.debug(f" Simple Tamil time result: '{result}'")
        
        return result

    def _convert_time_with_suffix_preserve(self, match):
        hour, minute = match.groups()
        
        if self.config.enable_debug_logging:
            logger.debug(f" Converting Arabic time with suffix: '{hour}:{minute} மணிக்கு'")
        
        normalized_hour = hour
        normalized_minute = minute
        
        if hour.isdigit():
            try:
                normalized_hour = str(int(hour))
            except (ValueError, TypeError, OverflowError):
                pass
        
        if minute.isdigit():
            try:
                normalized_minute = str(int(minute))
            except (ValueError, TypeError, OverflowError):
                pass
        
        result = f"{normalized_hour} மணி {normalized_minute} நிமிடம்"
        
        if self.config.enable_debug_logging:
            logger.debug(f" Arabic time with suffix result: '{result}'")
        
        return result

    def _convert_time_simple(self, match):
        hour, minute = match.groups()
        
        if self.config.enable_debug_logging:
            logger.debug(f" Converting simple Arabic time: '{hour}:{minute}'")
        
        normalized_hour = hour
        normalized_minute = minute
        
        # Normalize hour (remove leading zeros)
        if hour.isdigit():
            try:
                normalized_hour = str(int(hour))
            except (ValueError, TypeError, OverflowError):
                pass
        
        # Normalize minute (remove leading zeros)
        if minute.isdigit():
            try:
                normalized_minute = str(int(minute))
            except (ValueError, TypeError, OverflowError):
                pass
        
        result = f"{normalized_hour} மணி {normalized_minute} நிமிடம்"
        
        if self.config.enable_debug_logging:
            logger.debug(f" Simple Arabic time result: '{result}'")
        
        return result

    def _convert_time_to_structured(self, match):
        hour, minute = match.groups()
        
        if self.config.enable_debug_logging:
            logger.debug(f" Converting structured time: '{hour}:{minute} மணி'")
        
        # CRITICAL FIX: Normalize leading zeros for both hour and minute
        normalized_hour = hour
        normalized_minute = minute
        
        if hour.isdigit():
            try:
                normalized_hour = str(int(hour))
            except (ValueError, TypeError, OverflowError):
                pass
        
        if minute.isdigit():
            try:
                normalized_minute = str(int(minute))
            except (ValueError, TypeError, OverflowError):
                pass
        
        result = f"{normalized_hour} மணி {normalized_minute} நிமிடம்"
        
        if self.config.enable_debug_logging:
            logger.debug(f" Structured time result: '{result}'")
        
        return result

    def normalize_datetime(self, text: str) -> str:
        """Normalize Tamil date and time formats - SAFER AND MORE PRECISE"""
        if not (self.config.normalize_time_format or self.config.normalize_date_format) or not text:
            return text or ""
        
        try:
            original = text
            
            if self.config.enable_debug_logging:
                logger.debug(f" DateTime normalization input: '{text}'")
            
            for i, (pattern, converter) in enumerate(self.time_patterns):
                old_text = text
                if callable(converter):
                    text = pattern.sub(converter, text)
                else:
                    text = pattern.sub(converter, text)
                
                if old_text != text and self.config.enable_debug_logging:
                    logger.debug(f" Time pattern {i+1} applied: '{old_text}' → '{text}'")
            
            if self.config.enable_debug_logging and original != text:
                logger.debug(f" Final datetime normalization: '{original}' → '{text}'")
            
            return text
            
        except Exception as e:
            logger.warning(f"DateTime normalization failed: {e}")
            return original

    def convert_tamil_numeral_colon_time(self, match):
        tamil_hour = match.group(1)
        tamil_minute = match.group(2)
        
        if self.config.enable_debug_logging:
            logger.debug(f" Converting Tamil numeral colon time: '{match.group(0)}'")
        
        digit_map = {'௦': '0', '௧': '1', '௨': '2', '௩': '3', '௪': '4',
                    '௫': '5', '௬': '6', '௭': '7', '௮': '8', '௯': '9'}
        
        arabic_hour = ''.join(digit_map.get(c, c) for c in tamil_hour)
        arabic_minute = ''.join(digit_map.get(c, c) for c in tamil_minute)
        
        if arabic_hour.isdigit():
            try:
                normalized_hour = str(int(arabic_hour))
                arabic_hour = normalized_hour
            except (ValueError, TypeError, OverflowError):
                pass
        
        if arabic_minute.isdigit():
            try:
                normalized_minute = str(int(arabic_minute))
                arabic_minute = normalized_minute
            except (ValueError, TypeError, OverflowError):
                pass
        
        result = f"{arabic_hour} மணி {arabic_minute} நிமிடம்"
        
        if self.config.enable_debug_logging:
            logger.debug(f" Tamil numeral colon time result: '{result}'")
        
        return result

    def convert_tamil_numeral_simple_time(self, match):
        tamil_hour = match.group(1)
        
        if self.config.enable_debug_logging:
            logger.debug(f" Converting Tamil numeral simple time: '{match.group(0)}'")
        
        # Convert Tamil numerals to Arabic
        digit_map = {'௦': '0', '௧': '1', '௨': '2', '௩': '3', '௪': '4',
                    '௫': '5', '௬': '6', '௭': '7', '௮': '8', '௯': '9'}
        arabic_hour = ''.join(digit_map.get(c, c) for c in tamil_hour)
        
        # Normalize leading zeros
        if arabic_hour.isdigit():
            try:
                normalized_hour = str(int(arabic_hour))
                arabic_hour = normalized_hour
            except (ValueError, TypeError, OverflowError):
                pass
        
        result = f"{arabic_hour} மணி"
        
        if self.config.enable_debug_logging:
            logger.debug(f" Tamil numeral simple time result: '{result}'")
        
        return result


class TamilASRNormalizer:
    
    def __init__(self, config: Optional[TamilASRConfig] = None, resources_path: Optional[str] = None):
        self.config = config or TamilASRConfig()

        if INDIC_NLP_AVAILABLE and resources_path:
            try:
                common.set_resources_path(resources_path)
                loader.load()
                self.indic_normalizer_factory = IndicNormalizerFactory()
                self.indic_normalizer = self.indic_normalizer_factory.get_normalizer("ta")
                self.indic_available = True
                logger.info("Indic NLP Library initialized for Tamil")
            except Exception as e:
                logger.warning(f"Failed to initialize Indic NLP: {e}")
                self.indic_available = False
        else:
            self.indic_available = False
        
        # Initialize processors
        self.number_converter = TamilNumberConverter(self.config)
        self.currency_processor = TamilCurrencyProcessor(self.config)
        self.phone_processor = TamilPhoneProcessor(self.config)
        self.filler_processor = TamilFillerProcessor(self.config)
        self.time_processor = TamilTimeProcessor(self.config)
        
        if self.config.enable_debug_logging:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.debug("Tamil ASR Normalizer initialized successfully")
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters - WITH COLON PROTECTION"""
        if not self.config.normalize_unicode or not text:
            return text or ""
        
        try:
            if self.indic_available:
                import re
                protected_text = text
                colon_pattern = re.compile(r'([௦-௯]+):([௦-௯]+)')
                
                # Replace colons with placeholders
                protected_text = colon_pattern.sub(r'\1__COLON__\2', protected_text)
                
                # Run Indic normalization
                normalized = self.indic_normalizer.normalize(protected_text)
                
                # Restore colons
                text = normalized.replace('__COLON__', ':')
                
                # Fix any visarga that still got through
                text = re.sub(r'([0-9௦-௯])ஃ([0-9௦-௯])', r'\1:\2', text)
                
            else:
                text = unicodedata.normalize('NFC', text)
            
            return text
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}")
            return text

    def _remove_punctuation(self, text: str) -> str:
        if not self.config.remove_punctuation or not text:
            return text or ""
        
        try:
            # Tamil punctuation marks
            tamil_punct = '।॥‌‍''""…–—'
            
            punct_to_remove = string.punctuation + tamil_punct
            
            preserved_chars = ""
            if self.config.preserve_decimal_points:
                preserved_chars += "."
            if self.config.preserve_percentages:
                preserved_chars += "%"
            
            for char in preserved_chars:
                punct_to_remove = punct_to_remove.replace(char, '')
            
            translator = str.maketrans(punct_to_remove, ' ' * len(punct_to_remove))
            result = text.translate(translator)
            
            result = re.sub(r'\s+', ' ', result).strip()
            
            return result
        except Exception as e:
            logger.warning(f"Punctuation removal failed: {e}")
            return text
        

    def _post_process_cleanup(self, text: str) -> str:
        if not text:
            return text
        
        try:
            original = text
            
            if self.config.enable_debug_logging:
                logger.debug(f" Post-processing cleanup input: '{text}'")

            
            # Pattern 1: Convert standalone "ஒரு" to "1"
            text = re.sub(r'\bஒரு\b', '1', text)
            
            oru_number_pattern = re.compile(r'\bஒரு\s+(\d+)\b')
            text = oru_number_pattern.sub(r'\1', text)
            
            # Clean up any double spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Fix any remaining currency format issues
            text = re.sub(r'(\d+)\s*ரூபாய்\s*ரூபாய்', r'\1 ரூபாய்', text)
            text = re.sub(r'(\d+)\s*ரூபாய்', r'\1 ரூபாய்', text)
            
            # Handle any orphaned Tamil rupee symbols
            text = re.sub(r'௹(\d+)', r'\1 ரூபாய்', text)
            
            # Final whitespace cleanup
            text = re.sub(r'\s+', ' ', text).strip()
            
            if self.config.enable_debug_logging and original != text:
                logger.debug(f" Post-processing cleanup result: '{original}' → '{text}'")
            
            return text
            
        except Exception as e:
            logger.warning(f"Post-processing cleanup failed: {e}")
            return text

    
    def _final_cleanup(self, text: str) -> str:
        try:
            if self.config.normalize_whitespace:
                text = re.sub(r'\s+', ' ', text).strip()
            
            if self.config.merge_repeated_chars:
                text = re.sub(r'([.!?])\1{2,}', r'\1', text)
            
            if self.config.lowercase_output:
                text = text.lower()
            
            return text
            
        except Exception as e:
            logger.warning(f"Final cleanup failed: {e}")
            return str(text).lower().strip() if text else ""
    
    def _validate_output(self, original: str, normalized: str) -> bool:
        if not self.config.enable_validation:
            return True
        
        try:
            issues = []
            
            if len(normalized) > len(original) * 3:
                issues.append("Excessive length increase")
            
            if len(normalized.strip()) == 0 and len(original.strip()) > 5:
                issues.append("Complete text loss")
            
            if issues:
                logger.warning(f"Validation issues (non-critical): {issues}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return True


    def normalize(self, text: str) -> str:
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
            logger.debug(f"=== TAMIL ASR NORMALIZATION START ===")
            logger.debug(f"Input: '{text}'")
        
        try:
            original = text
            
            # Step 1: Unicode normalization
            text = self._normalize_unicode(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After Unicode normalization: '{text}'")
            
            # Pattern 1: Handle colon-separated Tamil numerals with மணிக்கு (like ௯:௦௦ மணிக்கு)
            tamil_numeral_colon_time_pattern = re.compile(r'([௦-௯]+):([௦-௯]+)\s*மணிக்கு')
            text = tamil_numeral_colon_time_pattern.sub(self.time_processor.convert_tamil_numeral_colon_time, text)
            if self.config.enable_debug_logging:
                logger.debug(f"After Tamil numeral colon time pre-processing: '{text}'")
            
            # Pattern 2: Handle single Tamil numerals with மணிக்கு (like ௯ மணிக்கு)
            tamil_numeral_simple_time_pattern = re.compile(r'([௦-௯]+)\s*மணிக்கு')
            text = tamil_numeral_simple_time_pattern.sub(self.time_processor.convert_tamil_numeral_simple_time, text)
            if self.config.enable_debug_logging:
                logger.debug(f"After Tamil numeral simple time pre-processing: '{text}'")
            
            # Step 2: Convert Tamil numerals and numbers AFTER handling special time cases
            text = self.number_converter.convert_tamil_to_numbers(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After number conversion: '{text}'")
            
            # Step 3: Normalize currency
            text = self.currency_processor.normalize_currency(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After currency normalization: '{text}'")
            
            # Step 4: Normalize phone numbers
            text = self.phone_processor.normalize_phones(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After phone normalization: '{text}'")
            
            # Step 5: Normalize time and date
            text = self.time_processor.normalize_datetime(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After datetime normalization: '{text}'")
            
            # Step 6: Enhanced post-processing cleanup
            text = self._post_process_cleanup(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After enhanced post-processing cleanup: '{text}'")
            
            # Step 7: Remove filler words AFTER all conversions
            text = self.filler_processor.remove_fillers(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After filler removal: '{text}'")
            
            # Step 8: Remove punctuation AFTER filler removal
            text = self._remove_punctuation(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After punctuation removal: '{text}'")
            
            # Step 9: Final cleanup (case, whitespace)
            text = self._final_cleanup(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After final cleanup: '{text}'")
            
            # Step 10: Validation
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

class TamilASRNormaliser:
    """AudioBench-compatible Tamil normaliser wrapper"""
    
    def __init__(self, enable_disfluency_removal: bool = True, **kwargs):
        """Initialize the Tamil normalizer with AudioBench parameters"""
        self.enable_disfluency_removal = enable_disfluency_removal
        
        # Initialize your Tamil normalizer config (adjust based on your implementation)
        config = TamilASRConfig(
            remove_filler_words=enable_disfluency_removal,
            normalize_currency=True,
            convert_numbers_to_words=False,  # Keep as digits for ASR
            lowercase_output=True,
            normalize_whitespace=True,
            enable_validation=True
        )
        
        # Initialize the main normalizer
        self.normalizer = TamilASRNormalizer(config)
    
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
            # Use the main TamilASRNormalizer
            return self.normalizer.normalize(text)
        except Exception as e:
            logger.error(f"Tamil normalization failed: {e}")
            # Fallback: basic text cleaning
            import re
            text = re.sub(r'\s+', ' ', text.strip())
            return text