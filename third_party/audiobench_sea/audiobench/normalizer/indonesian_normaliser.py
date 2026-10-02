#!/usr/bin/env python3
"""
Indonesian ASR Normalizer for Fair WER Calculation - FIXED VERSION
Based on research and optimized for ASR evaluation tasks
"""

import re
import logging
import string
import unicodedata
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import traceback

# External libraries for Indonesian processing
try:
    from num2words import num2words
    NUM2WORDS_AVAILABLE = True
except ImportError:
    NUM2WORDS_AVAILABLE = False
    print("Warning: num2words not available - install with: pip install num2words")

try:
    # CORRECTED: Use the proper factory pattern import
    from indonesian_number_normalizer import create_normalizer
    # Test the import by creating a normalizer instance
    _test_normalizer = create_normalizer()
    INDO_NORM_AVAILABLE = True
    print("Successfully imported indonesian-number-normalizer")
except ImportError:
    INDO_NORM_AVAILABLE = False
    print("Warning: indonesian-number-normalizer not available - install with: pip install indonesian-number-normalizer")
except Exception as e:
    INDO_NORM_AVAILABLE = False
    print(f"Warning: indonesian-number-normalizer import failed: {e}")

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    print("Warning: Sastrawi not available - install with: pip install Sastrawi")

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("Warning: jiwer not available - install with: pip install jiwer")

logger = logging.getLogger(__name__)

@dataclass
class IndonesianASRConfig:
    """Configuration for Indonesian ASR normalization"""
    
    # Core text processing
    normalize_unicode: bool = True
    remove_punctuation: bool = True
    lowercase_output: bool = True
    normalize_whitespace: bool = True
    
    # Number and numeric entities
    convert_numbers_to_words: bool = True
    preserve_decimal_points: bool = True
    preserve_percentages: bool = True
    normalize_currency: bool = True
    normalize_phone_numbers: bool = True
    normalize_time_format: bool = True
    normalize_date_format: bool = True
    
    # Indonesian-specific features
    handle_informal_text: bool = True
    remove_filler_words: bool = True
    handle_code_mixing: bool = True
    handle_repeated_chars: bool = True
    normalize_old_spelling: bool = True
    
    # Advanced options
    use_external_normalizer: bool = True  # Use indonesian-number-normalizer if available
    enable_stemming: bool = False  # Usually not needed for ASR
    enable_validation: bool = True
    enable_debug_logging: bool = True
    strict_mode: bool = False
    max_text_length: int = 500000


class IndonesianNumberConverter:
    """Convert Indonesian numbers between text and digits - FIXED VERSION"""
    
    def __init__(self, config: IndonesianASRConfig):
        self.config = config
        self._setup_mappings()
        self._compile_patterns()
        
        # Initialize indonesian-number-normalizer if available
        if INDO_NORM_AVAILABLE and self.config.use_external_normalizer:
            try:
                from indonesian_number_normalizer import create_normalizer
                self.indo_normalizer = create_normalizer()
            except Exception as e:
                logger.warning(f"Failed to initialize indonesian-number-normalizer: {e}")
                self.indo_normalizer = None
        else:
            self.indo_normalizer = None
    
    def _setup_mappings(self):
        """Setup number word mappings"""
        # Basic number words
        self.number_words = {
            'nol': 0, 'kosong': 0,
            'satu': 1, 'dua': 2, 'tiga': 3, 'empat': 4, 'lima': 5,
            'enam': 6, 'tujuh': 7, 'delapan': 8, 'sembilan': 9,
            'sepuluh': 10, 'sebelas': 11,
            'belas': 10,  # For teen numbers
            'puluh': 10, 'ratus': 100, 'ribu': 1000,
            'juta': 1000000, 'milyar': 1000000000, 'miliar': 1000000000,
            'triliun': 1000000000000, 'trilyun': 1000000000000
        }
        
        # Special number mappings
        self.special_numbers = {
            'seratus': 100, 'seribu': 1000, 'sejuta': 1000000,
            'setengah': 0.5, 'seperempat': 0.25, 'sepertiga': 0.33,
            'dua per tiga': 0.67, 'tiga perempat': 0.75
        }
        
        # Ordinal suffixes
        self.ordinal_patterns = [
            ('pertama', '1'), ('kedua', '2'), ('ketiga', '3'),
            ('keempat', '4'), ('kelima', '5'), ('keenam', '6'),
            ('ketujuh', '7'), ('kedelapan', '8'), ('kesembilan', '9'),
            ('kesepuluh', '10'), ('kesebelas', '11'), ('kedua belas', '12'),
            ('keseratus', '100'), ('keseribu', '1000')
        ]
    
    def _compile_patterns(self):
        """Compile regex patterns for number detection"""
        self.patterns = []
        
        # Pattern for digit sequences - FIXED to handle multiple dots correctly
        # This pattern matches:
        # - Simple numbers: 123
        # - Numbers with thousand separators: 1.500.000
        # - Numbers with decimal comma: 100,50
        # - Mixed format: 1.500,75
        self.patterns.append((
            re.compile(r'\b\d+(?:\.\d{3})*(?:,\d+)?\b'),
            self._convert_digits_to_words
        ))
        
        # Pattern for number words (convert to digits if needed)
        number_word_pattern = r'\b(?:' + '|'.join(self.number_words.keys()) + r')\b'
        self.patterns.append((
            re.compile(number_word_pattern, re.IGNORECASE),
            self._convert_words_to_digits
        ))
        
        # Pattern for compound numbers (e.g., "dua puluh lima")
        compound_pattern = r'\b(\w+)\s+(puluh|ratus|ribu|juta|milyar|miliar|triliun)\s*(\w*)\b'
        self.patterns.append((
            re.compile(compound_pattern, re.IGNORECASE),
            self._convert_compound_number
        ))
        
        # Pattern for teen numbers (e.g., "lima belas")
        teen_pattern = r'\b(\w+)\s+belas\b'
        self.patterns.append((
            re.compile(teen_pattern, re.IGNORECASE),
            self._convert_teen_number
        ))
    
    def _convert_digits_to_words(self, match):
        """Convert digits to Indonesian words - COMPLETELY FIXED VERSION"""
        number_str = match.group(0)
        
        if self.config.enable_debug_logging:
            logger.debug(f"Converting digits to words: '{number_str}'")
        
        try:
            # CRITICAL: In Indonesian, dots are thousand separators and commas are decimal separators
            # We need to check the pattern to determine what we're dealing with
            
            # Pattern 1: Numbers with dots as thousand separators (e.g., 1.500.000)
            # These should have dots followed by exactly 3 digits
            if '.' in number_str and ',' not in number_str:
                # Check if this follows thousand separator pattern
                parts = number_str.split('.')
                is_thousand_separator = True
                
                # First part can be any length, subsequent parts must be exactly 3 digits
                for i, part in enumerate(parts):
                    if i > 0 and len(part) != 3:
                        is_thousand_separator = False
                        break
                    if not part.isdigit():
                        is_thousand_separator = False
                        break
                
                if is_thousand_separator:
                    # This is a number with thousand separators
                    clean_number = number_str.replace('.', '')
                    try:
                        integer_value = int(clean_number)
                        result = self._number_to_words(integer_value)
                        if self.config.enable_debug_logging:
                            logger.debug(f"Thousand separator format '{number_str}' → '{result}'")
                        return result
                    except (ValueError, TypeError):
                        return number_str
            
            # Pattern 2: Numbers with comma as decimal separator (e.g., 100,50)
            if ',' in number_str:
                parts = number_str.split(',')
                if len(parts) == 2:
                    integer_part = parts[0]
                    decimal_part = parts[1]
                    
                    if self.config.enable_debug_logging:
                        logger.debug(f"Decimal number: {integer_part},{decimal_part}")
                    
                    try:
                        # Remove dots from integer part (thousand separators)
                        clean_integer = integer_part.replace('.', '')
                        
                        # Convert integer part
                        integer_value = int(clean_integer)
                        integer_words = self._number_to_words(integer_value)
                        
                        # Convert decimal part digit by digit (INCLUDING trailing zeros)
                        decimal_words_list = []
                        for digit_char in decimal_part:
                            if digit_char.isdigit():
                                digit_word = self._digit_to_word(int(digit_char))
                                decimal_words_list.append(digit_word)
                        
                        if decimal_words_list:
                            decimal_words = ' '.join(decimal_words_list)
                            result = f"{integer_words} koma {decimal_words}"
                        else:
                            result = integer_words
                        
                        if self.config.enable_debug_logging:
                            logger.debug(f"Decimal conversion: '{number_str}' → '{result}'")
                        
                        return result
                        
                    except (ValueError, TypeError) as e:
                        if self.config.enable_debug_logging:
                            logger.debug(f"Decimal conversion failed: {e}")
                        return number_str
            
            # Pattern 3: Simple integers (no dots or commas)
            if '.' not in number_str and ',' not in number_str:
                try:
                    integer_value = int(number_str)
                    result = self._number_to_words(integer_value)
                    if self.config.enable_debug_logging:
                        logger.debug(f"Simple integer: '{number_str}' → '{result}'")
                    return result
                except (ValueError, TypeError):
                    return number_str
            
            # If we get here, return original
            return number_str
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Number conversion failed for '{number_str}': {e}")
            return number_str
    
    def _number_to_words(self, number):
        """Custom Indonesian number to words conversion - FIXED VERSION"""
        if number == 0:
            return 'nol'
        
        # Handle negative numbers
        if number < 0:
            return 'negatif ' + self._number_to_words(-number)
        
        # Convert to integer if it's a float with no decimal part
        if isinstance(number, float) and number.is_integer():
            number = int(number)
        
        # Handle very large numbers first
        if number >= 1000000000000:  # Trillions
            trillions = number // 1000000000000
            remainder = number % 1000000000000
            if trillions == 1:
                result = 'satu triliun'
            else:
                result = self._number_to_words(trillions) + ' triliun'
            if remainder > 0:
                result += ' ' + self._number_to_words(remainder)
            return result
        
        # Billions
        if number >= 1000000000:
            billions = number // 1000000000
            remainder = number % 1000000000
            if billions == 1:
                result = 'satu milyar'
            else:
                result = self._number_to_words(billions) + ' milyar'
            if remainder > 0:
                result += ' ' + self._number_to_words(remainder)
            return result
        
        # Millions
        if number >= 1000000:
            millions = number // 1000000
            remainder = number % 1000000
            if millions == 1:
                result = 'satu juta'
            else:
                result = self._number_to_words(millions) + ' juta'
            if remainder > 0:
                result += ' ' + self._number_to_words(remainder)
            return result
        
        # Thousands
        if number >= 1000:
            thousands = number // 1000
            remainder = number % 1000
            if thousands == 1:
                result = 'seribu'
            else:
                result = self._number_to_words(thousands) + ' ribu'
            if remainder > 0:
                result += ' ' + self._number_to_words(remainder)
            return result
        
        # Hundreds
        if number >= 100:
            hundreds = number // 100
            remainder = number % 100
            if hundreds == 1:
                result = 'seratus'
            else:
                result = self._number_to_words(hundreds) + ' ratus'
            if remainder > 0:
                result += ' ' + self._number_to_words(remainder)
            return result
        
        # Handle 10-99
        if number >= 20:
            tens = number // 10
            ones = number % 10
            tens_word = self._digit_to_word(tens) + ' puluh'
            if ones > 0:
                return tens_word + ' ' + self._digit_to_word(ones)
            else:
                return tens_word
        
        # Handle 11-19 (teens)
        if number >= 12:
            return self._digit_to_word(number - 10) + ' belas'
        elif number == 11:
            return 'sebelas'
        elif number == 10:
            return 'sepuluh'
        
        # Handle 1-9
        if number >= 1:
            return self._digit_to_word(number)
        
        return 'nol'
    
    def _digit_to_word(self, digit):
        """Convert single digit to word - ENSURE PROPER HANDLING OF ZERO"""
        digit_words = {
            0: 'nol', 1: 'satu', 2: 'dua', 3: 'tiga', 4: 'empat', 
            5: 'lima', 6: 'enam', 7: 'tujuh', 8: 'delapan', 9: 'sembilan'
        }
        
        try:
            # Ensure we handle both int and string inputs properly
            if isinstance(digit, str):
                if digit.isdigit():
                    digit_int = int(digit)
                else:
                    return digit
            else:
                digit_int = int(digit)
            
            # Return the word for the digit
            result = digit_words.get(digit_int, str(digit_int))
            
            if self.config.enable_debug_logging:
                logger.debug(f"Digit to word: {digit} → {result}")
            
            return result
            
        except (ValueError, TypeError) as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Digit conversion error for '{digit}': {e}")
            return str(digit)
    
    def _convert_words_to_digits(self, match):
        """Convert Indonesian number words to digits"""
        word = match.group(0).lower()
        
        if word in self.number_words:
            return str(self.number_words[word])
        elif word in self.special_numbers:
            return str(self.special_numbers[word])
        
        return word
    
    def _convert_compound_number(self, match):
        """Convert compound numbers like 'dua puluh lima' to digits or words based on config"""
        multiplier_word = match.group(1).lower()
        scale_word = match.group(2).lower()
        remainder_word = match.group(3).lower() if match.group(3) else ''
        
        try:
            # Get multiplier value
            multiplier = self.number_words.get(multiplier_word, 1)
            if multiplier_word in self.special_numbers:
                multiplier = self.special_numbers[multiplier_word]
            
            # Get scale value
            scale = self.number_words.get(scale_word, 1)
            
            # Calculate base value
            value = multiplier * scale
            
            # Add remainder if present
            if remainder_word and remainder_word in self.number_words:
                value += self.number_words[remainder_word]
            
            # Return words for ASR evaluation
            if self.config.convert_numbers_to_words:
                return self._number_to_words(value)
            else:
                return str(value)
        
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Compound number conversion failed: {e}")
            return match.group(0)
    
    def _convert_teen_number(self, match):
        """Convert teen numbers like 'lima belas' to digits or words"""
        base_word = match.group(1).lower()
        
        try:
            if base_word in self.number_words:
                base_value = self.number_words[base_word]
                teen_value = 10 + base_value
                
                if self.config.convert_numbers_to_words:
                    return self._number_to_words(teen_value)
                else:
                    return str(teen_value)
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Teen number conversion failed: {e}")
        
        return match.group(0)
    
    def convert_numbers(self, text: str) -> str:
        """Main method to convert numbers in text - FIXED TO RESPECT CURRENCY"""
        if not text:
            return ""
        
        result = text
        
        try:
            # CRITICAL: Check if text contains protected currency markers
            # If it does, we need to be very careful about what we process
            has_currency_marker = '\u200B' in text
            
            # If external normalizer is available and enabled
            if self.indo_normalizer and self.config.use_external_normalizer and not has_currency_marker:
                try:
                    # First, save all decimal numbers before normalization
                    decimal_comma_pattern = re.compile(r'\b(\d+(?:\.\d{3})*),(\d+)\b')
                    decimal_positions = []
                    
                    # Record all decimal numbers and their positions
                    for match in decimal_comma_pattern.finditer(text):
                        decimal_positions.append({
                            'start': match.start(),
                            'end': match.end(),
                            'original': match.group(0),
                            'integer_part': match.group(1),
                            'decimal_part': match.group(2)
                        })
                    
                    # Apply external normalizer
                    result = self.indo_normalizer.normalize_text(text)

                    # Fix any decimal numbers that were processed
                    for dec_info in decimal_positions:
                        # Create a pattern to find what the normalizer did with our decimal
                        integer_part = dec_info['integer_part'].replace('.', '')
                        try:
                            integer_value = int(integer_part)
                            integer_words = self._number_to_words(integer_value)
                            
                            # Look for patterns like "integer_words koma ..."
                            pattern = re.escape(integer_words) + r'\s+koma\s+[\w\s]+'
                            
                            # Convert decimal part properly with all digits
                            decimal_words_list = []
                            for digit in dec_info['decimal_part']:
                                if digit.isdigit():
                                    decimal_words_list.append(self._digit_to_word(int(digit)))
                            
                            if decimal_words_list:
                                correct_decimal = f"{integer_words} koma {' '.join(decimal_words_list)}"
                                
                                # Try to find and replace the pattern
                                result = re.sub(pattern, correct_decimal, result, count=1)
                        
                        except Exception as e:
                            if self.config.enable_debug_logging:
                                logger.debug(f"Decimal fix failed: {e}")
                    
                    # Also check if there are any remaining unconverted decimals
                    result = decimal_comma_pattern.sub(lambda m: self._convert_digits_to_words(m), result)
                        
                except Exception as e:
                    logger.warning(f"External normalizer processing failed: {e}")
                    # Fall back to our own conversion
                    result = self._apply_internal_conversion(text)
            else:
                # Use our own conversion patterns or skip if currency markers present
                if has_currency_marker:
                    # Split text by currency markers and only process non-currency parts
                    parts = text.split('\u200B')
                    processed_parts = []
                    
                    for i, part in enumerate(parts):
                        if i == 0:  # First part might have numbers to convert
                            # But check if it ends with "rupiah"
                            if not part.rstrip().endswith('rupiah'):
                                processed_parts.append(self._apply_internal_conversion(part))
                            else:
                                processed_parts.append(part)
                        else:
                            # Parts after markers are protected
                            processed_parts.append(part)
                    
                    result = '\u200B'.join(processed_parts)
                else:
                    result = self._apply_internal_conversion(text)
            
            # Handle ordinal numbers
            for ordinal, replacement in self.ordinal_patterns:
                pattern = r'\b' + re.escape(ordinal) + r'\b'
                old_result = result
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                if self.config.enable_debug_logging and old_result != result:
                    logger.debug(f"Ordinal conversion: '{old_result}' → '{result}'")
            
            return result
            
        except Exception as e:
            logger.warning(f"Number conversion failed: {e}")
            return text
        
    def _apply_internal_conversion(self, text: str) -> str:
        """Apply internal number conversion patterns"""
        result = text
        
        # Apply patterns based on configuration
        if self.config.convert_numbers_to_words:
            # Convert digits to words for ASR
            for pattern, converter in self.patterns:
                if converter == self._convert_digits_to_words:
                    old_result = result
                    result = pattern.sub(converter, result)
                    if self.config.enable_debug_logging and old_result != result:
                        logger.debug(f"Digit conversion: '{old_result}' -> '{result}'")
        
        return result


class IndonesianCurrencyProcessor:
    """Process Indonesian currency formats - FIXED VERSION FOR RUPIAH AND FOREIGN CURRENCIES"""
    
    def __init__(self, config: IndonesianASRConfig):
        self.config = config
        self.number_converter = IndonesianNumberConverter(config)
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile currency patterns - FIXED for proper Rupiah and foreign currency handling"""
        self.patterns = [
            # FOREIGN CURRENCIES - MUST COME FIRST to avoid partial matches
            
            # USD with $ symbol
            (re.compile(r'\$(\d+(?:\.\d{3})*(?:,\d+)?)\b', re.IGNORECASE), 
             self._convert_dollar_symbol),
            
            # USD code format
            (re.compile(r'\b(USD)\s+(\d+(?:\.\d{3})*(?:,\d+)?)\b', re.IGNORECASE),
             self._convert_usd_code),
            
            # EUR with € symbol
            (re.compile(r'€(\d+(?:\.\d{3})*(?:,\d+)?)\b', re.IGNORECASE),
             self._convert_euro_symbol),
            
            # EUR code format
            (re.compile(r'\b(EUR)\s+(\d+(?:\.\d{3})*(?:,\d+)?)\b', re.IGNORECASE),
             self._convert_eur_code),
            
            # GBP with £ symbol
            (re.compile(r'£(\d+(?:\.\d{3})*(?:,\d+)?)\b', re.IGNORECASE),
             self._convert_pound_symbol),
            
            # GBP code format
            (re.compile(r'\b(GBP)\s+(\d+(?:\.\d{3})*(?:,\d+)?)\b', re.IGNORECASE),
             self._convert_gbp_code),
            
            # JPY with ¥ symbol (but not Rp - negative lookbehind)
            (re.compile(r'(?<!R)¥(\d+(?:\.\d{3})*(?:,\d+)?)\b', re.IGNORECASE),
             self._convert_yen_symbol),
            
            # JPY code format
            (re.compile(r'\b(JPY)\s+(\d+(?:\.\d{3})*(?:,\d+)?)\b', re.IGNORECASE),
             self._convert_jpy_code),
            
            # INDONESIAN RUPIAH PATTERNS
            
            # Pattern 1: Rp with amount and trailing ,- or .- or -
            (re.compile(r'(Rp\.?\s*)(\d+(?:\.\d{3})*(?:,\d+)?)[,.-]+\s*$', re.IGNORECASE), 
             self._convert_rupiah_with_trailing_punct),
            
            # Pattern 2: Standard Rp patterns (with or without dot)
            (re.compile(r'(Rp\.?\s*)(\d+(?:\.\d{3})*(?:,\d+)?)\b', re.IGNORECASE), 
             self._convert_rupiah_standard),
            
            # Pattern 3: IDR format
            (re.compile(r'(IDR\s+)(\d+(?:\.\d{3})*(?:,\d+)?)\b', re.IGNORECASE),
             self._convert_idr_format),
            
            # Pattern 4: Written rupiah (number already followed by rupiah)
            (re.compile(r'(\d+(?:\.\d{3})*(?:,\d+)?)\s*rupiah\b', re.IGNORECASE),
             self._convert_rupiah_suffix),
        ]
    
    def _convert_number_to_words_safe(self, amount_str):
        """Safely convert number string to words, handling Indonesian number format"""
        try:
            if self.config.enable_debug_logging:
                logger.debug(f"Converting amount: '{amount_str}'")
            
            # Remove any trailing punctuation first
            amount_str = amount_str.rstrip(',-.')
            
            # Remove thousand separators (dots in Indonesian format)
            clean_amount = amount_str.replace('.', '')
            
            if self.config.enable_debug_logging:
                logger.debug(f"Cleaned amount: '{clean_amount}'")
            
            # Check if there's a decimal comma
            if ',' in clean_amount:
                parts = clean_amount.split(',')
                integer_part = parts[0]
                decimal_part = parts[1]
                
                # Convert integer part
                if integer_part.isdigit():
                    integer_value = int(integer_part)
                    # Use the number converter's method directly
                    integer_words = self.number_converter._number_to_words(integer_value)
                    
                    # Convert decimal part digit by digit
                    decimal_words_list = []
                    for digit in decimal_part:
                        if digit.isdigit():
                            decimal_words_list.append(self.number_converter._digit_to_word(int(digit)))
                    
                    if decimal_words_list:
                        decimal_words = ' '.join(decimal_words_list)
                        return f"{integer_words} koma {decimal_words}"
                    else:
                        return integer_words
                else:
                    return amount_str
            else:
                # No decimal, just convert the integer
                if clean_amount.isdigit():
                    value = int(clean_amount)
                    # Use the number converter's method directly
                    words = self.number_converter._number_to_words(value)
                    if self.config.enable_debug_logging:
                        logger.debug(f"Converted {value} to '{words}'")
                    return words
                else:
                    return amount_str
                    
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Number to words conversion failed: {e}")
            return amount_str
    
    # FOREIGN CURRENCY CONVERTERS
    
    def _convert_dollar_symbol(self, match):
        """Convert $ format to words + dolar"""
        amount_str = match.group(1)
        
        if self.config.enable_debug_logging:
            logger.debug(f"Converting dollar symbol: '${amount_str}'")
        
        try:
            words = self._convert_number_to_words_safe(amount_str)
            result = f"{words} dolar"
            
            if self.config.enable_debug_logging:
                logger.debug(f"Dollar result: '${amount_str}' → '{result}'")
            
            return result
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Dollar conversion failed: {e}")
            return f"{amount_str} dolar"
    
    def _convert_usd_code(self, match):
        """Convert USD format to u s d + words"""
        code = match.group(1)  # USD
        amount_str = match.group(2)
        
        if self.config.enable_debug_logging:
            logger.debug(f"Converting USD code: '{code} {amount_str}'")
        
        try:
            # Convert code to spaced lowercase letters
            spaced_code = ' '.join(code.lower())  # "u s d"
            
            words = self._convert_number_to_words_safe(amount_str)
            result = f"{spaced_code} {words}"
            
            if self.config.enable_debug_logging:
                logger.debug(f"USD code result: '{code} {amount_str}' → '{result}'")
            
            return result
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"USD code conversion failed: {e}")
            return f"{' '.join(code.lower())} {amount_str}"
    
    def _convert_euro_symbol(self, match):
        """Convert € format to words + euro"""
        amount_str = match.group(1)
        
        if self.config.enable_debug_logging:
            logger.debug(f"Converting euro symbol: '€{amount_str}'")
        
        try:
            words = self._convert_number_to_words_safe(amount_str)
            result = f"{words} euro"
            
            if self.config.enable_debug_logging:
                logger.debug(f"Euro result: '€{amount_str}' → '{result}'")
            
            return result
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Euro conversion failed: {e}")
            return f"{amount_str} euro"
    
    def _convert_eur_code(self, match):
        """Convert EUR format to e u r + words"""
        code = match.group(1)  # EUR
        amount_str = match.group(2)
        
        if self.config.enable_debug_logging:
            logger.debug(f"Converting EUR code: '{code} {amount_str}'")
        
        try:
            # Convert code to spaced lowercase letters
            spaced_code = ' '.join(code.lower())  # "e u r"
            
            words = self._convert_number_to_words_safe(amount_str)
            result = f"{spaced_code} {words}"
            
            if self.config.enable_debug_logging:
                logger.debug(f"EUR code result: '{code} {amount_str}' → '{result}'")
            
            return result
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"EUR code conversion failed: {e}")
            return f"{' '.join(code.lower())} {amount_str}"
    
    def _convert_pound_symbol(self, match):
        """Convert £ format to words + pound"""
        amount_str = match.group(1)
        
        if self.config.enable_debug_logging:
            logger.debug(f"Converting pound symbol: '£{amount_str}'")
        
        try:
            words = self._convert_number_to_words_safe(amount_str)
            result = f"{words} pound"
            
            if self.config.enable_debug_logging:
                logger.debug(f"Pound result: '£{amount_str}' → '{result}'")
            
            return result
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Pound conversion failed: {e}")
            return f"{amount_str} pound"
    
    def _convert_gbp_code(self, match):
        """Convert GBP format to g b p + words"""
        code = match.group(1)  # GBP
        amount_str = match.group(2)
        
        if self.config.enable_debug_logging:
            logger.debug(f"Converting GBP code: '{code} {amount_str}'")
        
        try:
            # Convert code to spaced lowercase letters
            spaced_code = ' '.join(code.lower())  # "g b p"
            
            words = self._convert_number_to_words_safe(amount_str)
            result = f"{spaced_code} {words}"
            
            if self.config.enable_debug_logging:
                logger.debug(f"GBP code result: '{code} {amount_str}' → '{result}'")
            
            return result
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"GBP code conversion failed: {e}")
            return f"{' '.join(code.lower())} {amount_str}"
    
    def _convert_yen_symbol(self, match):
        """Convert ¥ format to words + yen"""
        amount_str = match.group(1)
        
        if self.config.enable_debug_logging:
            logger.debug(f"Converting yen symbol: '¥{amount_str}'")
        
        try:
            words = self._convert_number_to_words_safe(amount_str)
            result = f"{words} yen"
            
            if self.config.enable_debug_logging:
                logger.debug(f"Yen result: '¥{amount_str}' → '{result}'")
            
            return result
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Yen conversion failed: {e}")
            return f"{amount_str} yen"
    
    def _convert_jpy_code(self, match):
        """Convert JPY format to j p y + words"""
        code = match.group(1)  # JPY
        amount_str = match.group(2)
        
        if self.config.enable_debug_logging:
            logger.debug(f"Converting JPY code: '{code} {amount_str}'")
        
        try:
            # Convert code to spaced lowercase letters
            spaced_code = ' '.join(code.lower())  # "j p y"
            
            words = self._convert_number_to_words_safe(amount_str)
            result = f"{spaced_code} {words}"
            
            if self.config.enable_debug_logging:
                logger.debug(f"JPY code result: '{code} {amount_str}' → '{result}'")
            
            return result
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"JPY code conversion failed: {e}")
            return f"{' '.join(code.lower())} {amount_str}"
    
    # EXISTING RUPIAH CONVERTERS (unchanged)
    
    def _convert_rupiah_with_trailing_punct(self, match):
        """Convert Rp format with trailing punctuation like Rp 1.000,-"""
        prefix = match.group(1)  # "Rp" or "Rp." with spaces
        amount_str = match.group(2)  # The number part (without trailing punct)
        
        if self.config.enable_debug_logging:
            logger.debug(f"Converting Rupiah with trailing punct: '{match.group(0)}' prefix='{prefix}' amount='{amount_str}'")
        
        try:
            # Convert the amount to words
            words = self._convert_number_to_words_safe(amount_str)
            
            # Return just the words + rupiah (no punctuation)
            result = f"{words} rupiah"
            
            if self.config.enable_debug_logging:
                logger.debug(f"Rupiah with trailing punct result: '{match.group(0)}' → '{result}'")
            
            return result
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Rupiah with trailing punct conversion failed: {e}")
            # Fallback - ensure no trailing punctuation
            return f"{amount_str} rupiah"
    
    def _convert_rupiah_standard(self, match):
        """Convert standard Rp format to words + rupiah"""
        prefix = match.group(1)  # "Rp" or "Rp." with spaces
        amount_str = match.group(2)  # The number part
        
        if self.config.enable_debug_logging:
            logger.debug(f"Converting standard Rupiah: '{match.group(0)}' prefix='{prefix}' amount='{amount_str}'")
        
        try:
            # Check if the text already contains number words
            if re.search(r'[a-zA-Z]', amount_str):
                # Already contains words, just add rupiah
                result = f"{amount_str} rupiah"
            else:
                # Convert the amount to words
                words = self._convert_number_to_words_safe(amount_str)
                
                # Return just the words + rupiah (no prefix)
                result = f"{words} rupiah"
            
            if self.config.enable_debug_logging:
                logger.debug(f"Standard Rupiah result: '{match.group(0)}' → '{result}'")
            
            return result
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Standard Rupiah conversion failed: {e}")
            # Fallback
            return f"{amount_str} rupiah"
    
    def _convert_idr_format(self, match):
        """Convert IDR format to words + rupiah"""
        prefix = match.group(1)  # "IDR" with spaces
        amount_str = match.group(2)  # The number part
        
        if self.config.enable_debug_logging:
            logger.debug(f"Converting IDR format: '{match.group(0)}' prefix='{prefix}' amount='{amount_str}'")
        
        try:
            # Check if already contains words
            if re.search(r'[a-zA-Z]', amount_str):
                result = f"{amount_str} rupiah"
            else:
                words = self._convert_number_to_words_safe(amount_str)
                result = f"{words} rupiah"
            
            if self.config.enable_debug_logging:
                logger.debug(f"IDR format result: '{match.group(0)}' → '{result}'")
            
            return result
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"IDR format conversion failed: {e}")
            return f"{amount_str} rupiah"
    
    def _convert_rupiah_suffix(self, match):
        """Convert number with rupiah suffix - already has rupiah word"""
        amount_str = match.group(1)
        
        if self.config.enable_debug_logging:
            logger.debug(f"Converting rupiah suffix: '{match.group(0)}' with amount '{amount_str}'")
        
        try:
            # Check if already contains words
            if re.search(r'[a-zA-Z]', amount_str):
                result = f"{amount_str} rupiah"
            else:
                words = self._convert_number_to_words_safe(amount_str)
                result = f"{words} rupiah"
            
            if self.config.enable_debug_logging:
                logger.debug(f"Rupiah suffix result: '{match.group(0)}' → '{result}'")
            
            return result
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Rupiah suffix conversion failed: {e}")
            return f"{amount_str} rupiah"
    
    def normalize_currency(self, text: str) -> str:
        """Normalize currency in text - FIXED VERSION FOR RUPIAH AND FOREIGN CURRENCIES"""
        if not self.config.normalize_currency or not text:
            return text or ""
        
        result = text
        
        try:
            # Apply patterns in order
            for pattern, converter in self.patterns:
                old_result = result
                result = pattern.sub(converter, result)
                if self.config.enable_debug_logging and old_result != result:
                    logger.debug(f"Currency pattern applied: '{old_result}' → '{result}'")
            
            # Final cleanup: ensure no double spaces
            result = re.sub(r'\s+', ' ', result).strip()
            
            # Remove any trailing punctuation that might have been missed
            # This handles edge cases where punctuation wasn't at the very end
            result = re.sub(r'\s*[,.-]+\s*$', '', result)
            
            if self.config.enable_debug_logging and text != result:
                logger.debug(f"Final currency normalization: '{text}' → '{result}'")
            
        except Exception as e:
            logger.warning(f"Currency normalization failed: {e}")
            return text
        
        return result


class IndonesianPhoneProcessor:
    """Process Indonesian phone number formats - FIXED FOR DECIMAL NUMBERS"""
    
    def __init__(self, config: IndonesianASRConfig):
        self.config = config
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile phone number patterns - MORE RESTRICTIVE TO AVOID FALSE MATCHES"""
        # Indonesian phone patterns - ORDER MATTERS (most specific first)
        self.patterns = [
            # International format with +62
            (re.compile(r'\+62\s*[\-]?\s*(\d+(?:[\s\-]\d+)*)'), 
             self._convert_international_plus),
            
            # Format with 62 (no +) - must have space or dash after 62 AND be followed by valid phone digits
            (re.compile(r'\b62\s+([789]\d{2}[\s\-]?\d{3,4}[\s\-]?\d{3,5})\b'),
             self._convert_international_no_plus),
            
            # Local format starting with 08 or 02-09 (valid Indonesian phone prefixes)
            (re.compile(r'\b0([2-9]\d{2,3}[\s\-]?\d{3,4}[\s\-]?\d{3,5})\b'),
             self._convert_local),
            
            # Mobile numbers without 0 prefix (8xx) - must be followed by enough digits
            (re.compile(r'\b(8\d{2}[\s\-]\d{3,4}[\s\-]\d{3,5})\b'),
             self._convert_mobile),
            
            # Mobile numbers continuous (8xx) - exactly 10-11 digits total
            (re.compile(r'\b(8\d{9,10})\b(?!\d)'),
             self._convert_mobile),
        ]
    
    def _digits_to_words(self, digits: str) -> str:
        """Convert phone digits to Indonesian words - EACH DIGIT SEPARATELY"""
        digit_words = {
            '0': 'nol', '1': 'satu', '2': 'dua', '3': 'tiga', '4': 'empat',
            '5': 'lima', '6': 'enam', '7': 'tujuh', '8': 'delapan', '9': 'sembilan'
        }
        
        # Clean the input - remove ALL non-digit characters
        clean_digits = re.sub(r'[^\d]', '', digits)
        
        result = []
        for char in clean_digits:
            if char.isdigit():
                result.append(digit_words.get(char, char))
        
        return ' '.join(result)
    
    def _convert_international_plus(self, match):
        """Convert +62 format"""
        # Get all the remaining digits after +62
        remaining = match.group(1)
        
        # Clean and convert all digits
        all_digits = '62' + re.sub(r'[^\d]', '', remaining)
        digit_words = self._digits_to_words(all_digits)
        
        # Add 'plus' at the beginning
        return 'plus ' + digit_words
    
    def _convert_international_no_plus(self, match):
        """Convert 62 format (no plus)"""
        # Get all the remaining digits after 62
        remaining = match.group(1)
        
        # Clean and convert all digits
        all_digits = '62' + re.sub(r'[^\d]', '', remaining)
        return self._digits_to_words(all_digits)
    
    def _convert_local(self, match):
        """Convert local format starting with 0"""
        # Get all the remaining digits after 0
        remaining = match.group(1)
        
        # Clean and convert all digits including the 0
        all_digits = '0' + re.sub(r'[^\d]', '', remaining)
        return self._digits_to_words(all_digits)
    
    def _convert_mobile(self, match):
        """Convert mobile format"""
        # Get the full match
        full_number = match.group(0)
        
        # Clean and convert all digits
        all_digits = re.sub(r'[^\d]', '', full_number)
        return self._digits_to_words(all_digits)
    
    def _is_likely_phone_number(self, text: str) -> bool:
        """Check if text is likely a phone number to avoid false matches"""
        # Remove non-digits for checking
        digits_only = re.sub(r'[^\d]', '', text)
        
        # Check length
        if len(digits_only) < 7 or len(digits_only) > 15:
            return False
        
        # Check Indonesian phone patterns
        if text.startswith('+62'):
            return True
        if re.match(r'^62\s', text):
            return True
        if re.match(r'^0[2-9]', digits_only):
            return True
        if re.match(r'^8[0-9]{9,10}$', digits_only):
            return True
        
        return False
    
    def protect_phone_numbers(self, text: str) -> str:
        """Add protection markers to phone numbers before general number conversion"""
        if not self.config.normalize_phone_numbers or not text:
            return text or ""
        
        # Check if external normalizer is being used
        if hasattr(self.config, 'use_external_normalizer') and self.config.use_external_normalizer:
            # If using external normalizer, we need stronger protection
            # Convert phone numbers immediately instead of protecting
            return self.normalize_phones(text)
        
        # Otherwise use protection markers
        protected_text = text
        
        # Apply all patterns to mark phone numbers
        for pattern, _ in self.patterns:
            # Use a lambda to wrap matches with markers
            protected_text = pattern.sub(lambda m: f'\u200BPHONE\u200B{m.group(0)}\u200BPHONE\u200B', protected_text)
        
        return protected_text
    
    def normalize_phones(self, text: str) -> str:
        """Normalize phone numbers in text - WITH VALIDATION"""
        if not self.config.normalize_phone_numbers or not text:
            return text or ""
        
        result = text
        
        try:
            # Check if text has protection markers
            if '\u200BPHONE\u200B' in result:
                # Process protected phone numbers
                parts = result.split('\u200BPHONE\u200B')
                processed_parts = []
                
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        # Not a phone number
                        processed_parts.append(part)
                    else:
                        # This is a phone number
                        converted = part
                        # Additional validation before converting
                        if self._is_likely_phone_number(part):
                            for pattern, converter in self.patterns:
                                if pattern.match(part):
                                    converted = converter(pattern.match(part))
                                    if self.config.enable_debug_logging:
                                        logger.debug(f"Protected phone conversion: '{part}' → '{converted}'")
                                    break
                        processed_parts.append(converted)
                
                result = ''.join(processed_parts)
            else:
                # No protection markers, process directly
                for pattern, converter in self.patterns:
                    old_result = result
                    result = pattern.sub(converter, result)
                    if self.config.enable_debug_logging and old_result != result:
                        logger.debug(f"Direct phone conversion: '{old_result}' → '{result}'")
            
        except Exception as e:
            logger.warning(f"Phone normalization failed: {e}")
            return text
        
        return result


class IndonesianTimeProcessor:
    """Process Indonesian time and date formats - FIXED VERSION"""
    
    def __init__(self, config: IndonesianASRConfig):
        self.config = config
        self.number_converter = IndonesianNumberConverter(config)
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile time and date patterns - FIXED ORDER AND LOGIC"""
        self.patterns = [
            # Time with timezone MUST come before regular time to avoid partial matches
            (re.compile(r'\b(\d{1,2}):(\d{2})\s*(WIB|WITA|WIT)\b', re.IGNORECASE),
             self._convert_time_with_zone),
            
            # Pukul time format (before regular time)
            (re.compile(r'\bpukul\s+(\d{1,2}):(\d{2})\b', re.IGNORECASE),
             self._convert_pukul_time),
            
            # Jam time format (before regular time)
            (re.compile(r'\bjam\s+(\d{1,2}):(\d{2})\b', re.IGNORECASE),
             self._convert_jam_time),
            
            # Regular time formats (HH:MM)
            (re.compile(r'\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b'),
             self._convert_time),
            
            # Date formats DD/MM/YYYY or DD-MM-YYYY
            (re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b'),
             self._convert_date),
            
            # Written months
            (re.compile(r'\b(\d{1,2})\s+(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+(\d{4})\b', re.IGNORECASE),
             self._convert_written_date),
            
            # Written months without year
            (re.compile(r'\b(\d{1,2})\s+(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\b', re.IGNORECASE),
             self._convert_written_date_no_year),
        ]
        
        # Month names
        self.month_names = {
            1: 'januari', 2: 'februari', 3: 'maret', 4: 'april',
            5: 'mei', 6: 'juni', 7: 'juli', 8: 'agustus',
            9: 'september', 10: 'oktober', 11: 'november', 12: 'desember'
        }
    
    def _convert_time(self, match):
        """Convert time format to Indonesian words - COMPLETELY FIXED"""
        try:
            hour = int(match.group(1))
            minute = int(match.group(2))
            second = int(match.group(3)) if match.group(3) else None
            
            if self.config.enable_debug_logging:
                logger.debug(f"Converting time: hour={hour}, minute={minute}")
            
            # Special case for half hours (setengah)
            if minute == 30:
                # Calculate next hour for "setengah" format
                next_hour = hour + 1
                if next_hour == 24:
                    next_hour = 0
                
                # Convert to 12-hour format for display
                # CRITICAL FIX: For setengah, we use the NEXT hour
                display_hour = next_hour
                if display_hour == 0:
                    display_hour = 12
                elif display_hour > 12:
                    display_hour = display_hour - 12
                
                hour_words = self.number_converter._number_to_words(display_hour)
                result = f"setengah {hour_words}"
                
                if self.config.enable_debug_logging:
                    logger.debug(f"Half hour conversion: {hour}:30 → {result}")
                
                return result
            
            # Special case for :15 - use "seperempat"
            if minute == 15:
                # Convert hour to 12-hour format
                display_hour = hour
                if hour == 0:
                    display_hour = 12
                elif hour > 12:
                    display_hour = hour - 12
                
                hour_words = self.number_converter._number_to_words(display_hour)
                return f"{hour_words} seperempat"
            
            # Special case for :45 - use "kurang seperempat" from next hour
            if minute == 45:
                # Calculate next hour
                next_hour = hour + 1
                if next_hour == 24:
                    next_hour = 0
                
                # Convert to 12-hour
                display_hour = next_hour
                if display_hour == 0:
                    display_hour = 12
                elif display_hour > 12:
                    display_hour = display_hour - 12
                
                hour_words = self.number_converter._number_to_words(display_hour)
                return f"{hour_words} kurang seperempat"
            
            # Special case for exact hours
            if minute == 0:
                if hour == 0:
                    return "dua belas malam"
                elif hour == 12:
                    return "dua belas siang"
                elif hour == 24:
                    return "dua belas malam"
                else:
                    # Convert to 12-hour format for display
                    display_hour = hour
                    if hour > 12:
                        display_hour = hour - 12
                    
                    hour_words = self.number_converter._number_to_words(display_hour)
                    
                    # Add time period
                    if hour < 5:
                        return f"jam {hour_words} dini hari"
                    elif hour < 10:
                        return f"jam {hour_words} pagi"
                    elif hour < 15:
                        return f"jam {hour_words} siang"
                    elif hour < 18:
                        return f"jam {hour_words} sore"
                    else:
                        return f"jam {hour_words} malam"
            
            # For other minutes - use "lewat" format
            # Convert to 12-hour format
            display_hour = hour
            if hour == 0:
                display_hour = 12
            elif hour > 12:
                display_hour = hour - 12
            
            hour_words = self.number_converter._number_to_words(display_hour)
            minute_words = self.number_converter._number_to_words(minute)
            
            # Use "lewat" format
            time_str = f"{hour_words} lewat {minute_words} menit"
            
            # Add seconds if present
            if second is not None:
                second_words = self.number_converter._number_to_words(second)
                time_str += f" {second_words} detik"
            
            return time_str
                    
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Time conversion failed: {e}")
            return match.group(0)
    
    def _convert_time_with_zone(self, match):
        """Convert time with timezone - FIXED"""
        try:
            hour = int(match.group(1))
            minute = int(match.group(2))
            zone = match.group(3).upper()
            
            # Create a mock match object to reuse _convert_time logic
            time_match = type('MockMatch', (), {
                'group': lambda self, x: match.group(x) if x <= 3 else None
            })()
            time_str = self._convert_time(time_match)
            
            # Map timezone - use full names as expected by tests
            zone_names = {
                'WIB': 'waktu indonesia barat',
                'WITA': 'waktu indonesia tengah', 
                'WIT': 'waktu indonesia timur'
            }
            
            zone_text = zone_names.get(zone, zone.lower())
            
            # Combine time and zone
            return f"{time_str} {zone_text}"
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Time with zone conversion failed: {e}")
            return match.group(0)
    
    def _convert_date(self, match):
        """Convert date format to words - FIXED TO CONVERT MONTH NUMBERS TO NAMES"""
        try:
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3))
            
            if self.config.enable_debug_logging:
                logger.debug(f"Converting date: day={day}, month={month}, year={year}")
            
            # Convert to words
            day_words = self.number_converter._number_to_words(day)
            month_name = self.month_names.get(month, str(month))
            year_words = self.number_converter._number_to_words(year)
            
            result = f"{day_words} {month_name} {year_words}"
            
            if self.config.enable_debug_logging:
                logger.debug(f"Date conversion result: {result}")
            
            # Return in the expected format
            return result
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Date conversion failed: {e}")
            return match.group(0)
    
    def _convert_written_date(self, match):
        """Convert written date format"""
        try:
            day = int(match.group(1))
            month_name = match.group(2).lower()
            year = int(match.group(3))
            
            day_words = self.number_converter._number_to_words(day)
            year_words = self.number_converter._number_to_words(year)
            
            # Keep existing format
            return f"{day_words} {month_name} {year_words}"
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Written date conversion failed: {e}")
            return match.group(0)
    
    def _convert_written_date_no_year(self, match):
        """Convert written date format without year"""
        try:
            day = int(match.group(1))
            month_name = match.group(2).lower()
            
            day_words = self.number_converter._number_to_words(day)
            
            # Keep existing format
            return f"{day_words} {month_name}"
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Written date (no year) conversion failed: {e}")
            return match.group(0)
    
    def _convert_pukul_time(self, match):
        """Convert 'pukul HH:MM' format - FIXED"""
        try:
            hour = int(match.group(1))
            minute = int(match.group(2))
            
            if self.config.enable_debug_logging:
                logger.debug(f"Converting pukul time: hour={hour}, minute={minute}")
            
            # For pukul format with half hours
            if minute == 30:
                # Calculate next hour for "setengah" format
                next_hour = hour + 1
                if next_hour == 24:
                    next_hour = 0
                
                # Convert to 12-hour for display
                display_hour = next_hour
                if display_hour == 0:
                    display_hour = 12
                elif display_hour > 12:
                    display_hour = display_hour - 12
                
                hour_words = self.number_converter._number_to_words(display_hour)
                return f"pukul setengah {hour_words}"
            
            # For other cases, use the regular time conversion
            time_match = type('MockMatch', (), {
                'group': lambda self, x: match.group(x) if x <= 2 else None
            })()
            time_result = self._convert_time(time_match)
            
            # Remove "jam" prefix if present
            if time_result.startswith("jam "):
                time_result = time_result[4:]
            
            return f"pukul {time_result}"
                
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Pukul time conversion failed: {e}")
            return match.group(0)
    
    def _convert_jam_time(self, match):
        """Convert 'jam HH:MM' format - FIXED"""
        try:
            hour = int(match.group(1))
            minute = int(match.group(2))
            
            if self.config.enable_debug_logging:
                logger.debug(f"Converting jam time: hour={hour}, minute={minute}")
            
            # For jam format with half hours
            if minute == 30:
                # Calculate next hour for "setengah" format
                next_hour = hour + 1
                if next_hour == 24:
                    next_hour = 0
                
                # Convert to 12-hour for display
                display_hour = next_hour
                if display_hour == 0:
                    display_hour = 12
                elif display_hour > 12:
                    display_hour = display_hour - 12
                
                hour_words = self.number_converter._number_to_words(display_hour)
                return f"jam setengah {hour_words}"
            
            # For other cases, use the regular time conversion
            time_match = type('MockMatch', (), {
                'group': lambda self, x: match.group(x) if x <= 2 else None
            })()
            time_result = self._convert_time(time_match)
            
            # Ensure "jam" prefix is present
            if not time_result.startswith("jam "):
                return f"jam {time_result}"
            else:
                return time_result
                    
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Jam time conversion failed: {e}")
            return match.group(0)
    
    def normalize_datetime(self, text: str) -> str:
        """Normalize date and time in text"""
        if not (self.config.normalize_time_format or self.config.normalize_date_format) or not text:
            return text or ""
        
        result = text
        
        try:
            # Apply patterns in the order they were defined
            for pattern, converter in self.patterns:
                old_result = result
                result = pattern.sub(converter, result)
                if self.config.enable_debug_logging and old_result != result:
                    logger.debug(f"DateTime conversion: '{old_result}' -> '{result}'")
        except Exception as e:
            logger.warning(f"DateTime normalization failed: {e}")
            return text
        
        return result


class IndonesianFillerProcessor:
    """Process Indonesian filler words and discourse markers"""
    
    def __init__(self, config: IndonesianASRConfig):
        self.config = config
        self._setup_filler_lists()
    
    def _setup_filler_lists(self):
        """Setup Indonesian filler words - ENHANCED VERSION"""
        # Common filler words - more comprehensive list
        self.filler_words = [
            # Basic fillers
            'anu',
            
            # Particles and discourse markers
            'kan', 'sih', 'dong', 'deh', 'kok', 'lho', 'lah', 'tuh', 'wah', 'aduh', 'ah', 'eh', 'oh', 'uh',
            
            # Hesitation sounds
            'hmm', 'emm', 'eee', 'mmm', 'uhh', 'ehh',
            
            # Complex fillers 
            'kan ya'
        ]
        
        # Discourse markers that might be kept based on context
        self.discourse_markers = [
            'jadi', 'namun', 'tetapi', 'tapi', 'atau', 'serta', 'dan', 
            'dengan', 'untuk', 'pada', 'dari', 'ke', 'di'
        ]
        
        # Repeated patterns - more comprehensive
        self.repeated_patterns = [
            (r'\b(apa)\s+\1+\b', r'\1'),  # apa apa apa -> apa
            (r'\b(ya)\s+\1+\b', r'\1'),    # ya ya ya -> ya
            (r'\b(itu)\s+\1+\b', r'\1'),  # itu itu -> itu
            (r'\b(ini)\s+\1+\b', r'\1'),  # ini ini -> ini
            (r'\b(kan)\s+\1+\b', r'\1'),  # kan kan -> kan
            (r'\b(gitu)\s+\1+\b', r'\1'), # gitu gitu -> gitu
            (r'\b(terus)\s+\1+\b', r'\1'), # terus terus -> terus
            (r'\b(anu)\s+\1+\b', r'\1'),  # anu anu -> anu
            
            # Common combinations
            (r'\bkan\s+ya\s+kan\b', 'kan ya'), # kan ya kan -> kan ya
            (r'\bgitu\s+lho\s+gitu\b', 'gitu lho'), # gitu lho gitu -> gitu lho
        ]
    
    def remove_fillers(self, text: str) -> str:
        """Remove filler words from text - ENHANCED VERSION"""
        if not self.config.remove_filler_words or not text:
            return text or ""
        
        if not isinstance(text, str):
            text = str(text)
        
        result = text
        original = text
        
        try:
            # Multiple passes to ensure thorough removal
            for iteration in range(5):
                old_text = result
                
                # Pass 1: Remove repeated patterns first
                for pattern, replacement in self.repeated_patterns:
                    result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                
                # Pass 2: Remove filler words with enhanced method
                for filler in self.filler_words:
                    result = self._enhanced_remove_filler(result, filler)
                
                # Pass 3: Clean up artifacts
                result = self._clean_artifacts(result)
                
                # If no changes, we're done
                if result == old_text:
                    break
            
            # Final cleanup
            result = re.sub(r'\s+', ' ', result).strip()
            
            if self.config.enable_debug_logging and original != result:
                logger.debug(f"Filler removal ({iteration+1} iterations): '{original}' → '{result}'")
            
            return result
            
        except Exception as e:
            logger.warning(f"Filler removal failed: {e}")
            return text
    
    def _enhanced_remove_filler(self, text: str, filler: str) -> str:
        """Enhanced filler removal with proper word boundaries - FIXED VERSION"""
        if not filler or filler not in text:
            return text
        
        original_text = text
        
        try:
            
            # Method 1: Word boundary replacement (this was already correct)
            filler_pattern = r'\b' + re.escape(filler) + r'\b'
            text = re.sub(filler_pattern, ' ', text, flags=re.IGNORECASE)
            
            # Method 2: Remove at start/end of text (with word boundaries)
            start_pattern = r'^\s*\b' + re.escape(filler) + r'\b\s*'
            text = re.sub(start_pattern, '', text, flags=re.IGNORECASE)
            
            end_pattern = r'\s*\b' + re.escape(filler) + r'\b\s*$'
            text = re.sub(end_pattern, '', text, flags=re.IGNORECASE)
            
            # Method 3: FIXED - Remove with punctuation (using word boundaries)
            punct_patterns = [
                (r'\b' + re.escape(filler) + r'\b\s*[,.]', ''),  # FIXED: Added \b
                (r'[,.]?\s*\b' + re.escape(filler) + r'\b\s*[,.]?', ' '),  # FIXED: Added \b
                (r'\b' + re.escape(filler) + r'\b\s+', ' '),  # FIXED: Added \b
                (r'\s+\b' + re.escape(filler) + r'\b', ' '),  # FIXED: Added \b
            ]
            
            for pattern, replacement in punct_patterns:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
            # Method 4: Handle standalone filler (exact match only)
            if text.strip().lower() == filler.lower():
                text = ''
            
            # Clean up multiple spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            if self.config.enable_debug_logging and original_text != text:
                logger.debug(f"Enhanced filler removal '{filler}': '{original_text}' → '{text}'")
            
            return text
            
        except Exception as e:
            logger.warning(f"Enhanced filler removal failed for '{filler}': {e}")
            return original_text

    def _clean_artifacts(self, text: str) -> str:
        """Clean up artifacts from filler removal"""
        try:
            # Remove multiple punctuation
            text = re.sub(r'[,.!?]+', lambda m: m.group(0)[0], text)
            
            # Clean up spaces around punctuation
            text = re.sub(r'\s*([,.!?])\s*', r'\1 ', text)
            
            # Remove punctuation at start
            text = re.sub(r'^[,.!?\s]+', '', text)
            
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Artifact cleaning failed: {e}")
            return text


class IndonesianInformalProcessor:
    """Handle informal Indonesian text (bahasa gaul)"""
    
    def __init__(self, config: IndonesianASRConfig):
        self.config = config
        self._setup_mappings()
    
    def _setup_mappings(self):
        """Setup informal to formal mappings"""
        # Common informal variations
        self.informal_mappings = {
            # Shortened words
            'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak', 'g': 'tidak',
            'udah': 'sudah', 'uda': 'sudah', 'dah': 'sudah',
            'aja': 'saja', 'aj': 'saja', 'ajah': 'saja',
            'doang': 'saja', 'doank': 'saja',
            'gimana': 'bagaimana', 'gmn': 'bagaimana', 'gmana': 'bagaimana',
            'bgt': 'banget', 'bngt': 'banget',
            'lg': 'lagi', 'lgi': 'lagi',
            'yg': 'yang', 'yng': 'yang',
            'dgn': 'dengan', 'dg': 'dengan',
            'utk': 'untuk', 'u/': 'untuk', 'dri': 'dari',
            'sm': 'sama', 'sma': 'sama',
            'jg': 'juga', 'jga': 'juga',
            'krn': 'karena', 'krna': 'karena',
            'blm': 'belum', 'blom': 'belum', 'belom': 'belum',
            'hrs': 'harus', 'hrus': 'harus',
            'bs': 'bisa', 'bsa': 'bisa',
            'kl': 'kalau', 'klo': 'kalau', 'kalo': 'kalau',
            'org': 'orang', 'orng': 'orang',
            'bkn': 'bukan', 'bukn': 'bukan',
            'jgn': 'jangan', 'jngn': 'jangan',
            'sdh': 'sudah', 'sudh': 'sudah',
            'tdk': 'tidak', 'tidk': 'tidak',
            'msh': 'masih', 'msih': 'masih',
            'spt': 'seperti', 'sprti': 'seperti',
            'pd': 'pada', 'pda': 'pada',
            'kpd': 'kepada', 'kpda': 'kepada',
            'sy': 'saya', 'sya': 'saya',
            'mk': 'maka', 'mka': 'maka',
            'tp': 'tapi', 'tpi': 'tapi',
            'ap': 'apa', 'apa?': 'apa',
            'knp': 'kenapa', 'knapa': 'kenapa', 'napa': 'kenapa',
            'dmn': 'dimana', 'dmna': 'dimana', 'dimna': 'dimana',
            'gmn': 'gimana', 'gmana': 'gimana',
            'kt': 'kita', 'kta': 'kita',
            'mrk': 'mereka', 'mrka': 'mereka',
            
            # Slang variations
            'gaes': 'guys',
            'plis': 'please',
            'thx': 'terima kasih', 'tq': 'terima kasih', 'thanks': 'terima kasih',
            'sori': 'sorry',
            'oke': 'ok', 'okey': 'ok', 'okay': 'ok',
            
            # Common typos and variations
            'trima': 'terima', 'trma': 'terima',
            'kasih': 'kasih', 'ksh': 'kasih',
            'makasih': 'terima kasih', 'mksh': 'terima kasih', 'mkasih': 'terima kasih',
            'tau': 'tahu', 'tw': 'tahu',
            'gatau': 'tidak tahu', 'gtw': 'tidak tahu', 'gtau': 'tidak tahu',
            'gapapa': 'tidak apa-apa', 'gpp': 'tidak apa-apa',
            'emang': 'memang', 'emg': 'memang',
            'bener': 'benar', 'bnr': 'benar',
            'malem': 'malam', 'mlm': 'malam', 'pg': 'pagi',
            'sre': 'sore', 'gitu':'begitu', 'gtu': 'begitu', 'tlp': 'telepon', 'gini': 'begini',
        }
        
        # Prokem/slang transformations
        self.slang_patterns = [
            # Baku (reversed syllables)
            (r'\bbokap\b', 'bapak'),
            (r'\bnyokap\b', 'ibu'),
            (r'\bcowok\b', 'pria'),
            (r'\bcewek\b', 'wanita'),
        ]
    
    def normalize_informal(self, text: str) -> str:
        """Normalize informal Indonesian text - ALTERNATIVE FIX"""
        if not self.config.handle_informal_text or not text:
            return text or ""
        
        result = text.lower()  # Work with lowercase for matching
        
        try:
            # Apply informal mappings
            for informal, formal in self.informal_mappings.items():
                pattern = r'\b' + re.escape(informal) + r'\b'
                result = re.sub(pattern, formal, result, flags=re.IGNORECASE)
            
            # Apply slang patterns
            for pattern, replacement in self.slang_patterns:
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            
            # Handle repeated characters (e.g., "bagussss" -> "bagus")
            if self.config.handle_repeated_chars:
                # Split into words and only apply to non-numeric words
                words = result.split()
                processed_words = []
                for word in words:
                    if word.isdigit() or any(c.isdigit() for c in word):
                        # Don't process words containing digits
                        processed_words.append(word)
                    else:
                        # Apply repeated character removal to non-numeric words
                        processed_words.append(re.sub(r'(.)\1{2,}', r'\1', word))
                result = ' '.join(processed_words)
                
        except Exception as e:
            logger.warning(f"Informal text normalization failed: {e}")
            return text
        
        return result


class IndonesianASRNormalizer:
    
    def __init__(self, config: Optional[IndonesianASRConfig] = None):
        self.config = config or IndonesianASRConfig()
        
        # Initialize processors
        self.number_converter = IndonesianNumberConverter(self.config)
        self.currency_processor = IndonesianCurrencyProcessor(self.config)
        self.phone_processor = IndonesianPhoneProcessor(self.config)
        self.time_processor = IndonesianTimeProcessor(self.config)
        self.filler_processor = IndonesianFillerProcessor(self.config)
        self.informal_processor = IndonesianInformalProcessor(self.config)
        
        # Initialize Sastrawi stemmer if available
        if SASTRAWI_AVAILABLE and self.config.enable_stemming:
            try:
                factory = StemmerFactory()
                self.stemmer = factory.create_stemmer()
            except Exception as e:
                logger.warning(f"Failed to initialize Sastrawi stemmer: {e}")
                self.stemmer = None
        else:
            self.stemmer = None
        
        if self.config.enable_debug_logging:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.debug("Indonesian ASR Normalizer initialized")
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        if not self.config.normalize_unicode or not text:
            return text or ""
        
        try:
            # Normalize to NFC form
            text = unicodedata.normalize('NFC', text)
            return text
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}")
            return text
    
    def _normalize_old_spelling(self, text: str) -> str:
        """Convert old Indonesian spelling (ejaan lama) to modern EYD"""
        if not self.config.normalize_old_spelling or not text:
            return text or ""
        
        try:
            # Common old spelling conversions
            old_spelling_map = {
                'dj': 'j', 'tj': 'c', 'nj': 'ny', 'sj': 'sy',
                'ch': 'kh', 'oe': 'u'
            }
            
            result = text
            for old, new in old_spelling_map.items():
                result = result.replace(old, new)
            
            return result
        except Exception as e:
            logger.warning(f"Old spelling normalization failed: {e}")
            return text
    
    def _handle_code_mixing(self, text: str) -> str:
        """Handle Indonesian-English code mixing"""
        if not self.config.handle_code_mixing or not text:
            return text or ""
        
        try:
            # Common English words in Indonesian context that should be kept
            common_english = {
                'online', 'offline', 'download', 'upload', 'update', 'upgrade',
                'email', 'website', 'software', 'hardware', 'smartphone',
                'meeting', 'training', 'workshop', 'deadline', 'project'
            }
            
            # For now, just lowercase English words
            # More sophisticated handling would require language identification
            words = text.split()
            result = []
            
            for word in words:
                # Check if word contains only ASCII characters (likely English)
                if word.isascii() and word.lower() in common_english:
                    result.append(word.lower())
                else:
                    result.append(word)
            
            return ' '.join(result)
        except Exception as e:
            logger.warning(f"Code mixing handling failed: {e}")
            return text
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation - ENHANCED VERSION with smart decimal/thousand separator preservation"""
        if not self.config.remove_punctuation or not text:
            return text or ""
        
        try:
            # First, protect decimal numbers and thousand separators
            protected_text = text
            protection_counter = 0
            protected_patterns = {}
            
            if self.config.preserve_decimal_points:
                # Pattern 1: Indonesian decimal numbers (comma as decimal separator)
                # e.g., "3,5" "100,50" "1234,567"
                decimal_pattern = re.compile(r'\b\d+,\d+\b')
                for match in decimal_pattern.finditer(text):
                    protection_key = f'__DECIMAL_{protection_counter}__'
                    protected_patterns[protection_key] = match.group(0)
                    protected_text = protected_text.replace(match.group(0), protection_key, 1)
                    protection_counter += 1
                
                # Pattern 2: Numbers with thousand separators (dots)
                # e.g., "1.000" "1.500.000" "25.750.000"
                thousand_pattern = re.compile(r'\b\d{1,3}(?:\.\d{3})+\b')
                for match in thousand_pattern.finditer(text):
                    protection_key = f'__THOUSAND_{protection_counter}__'
                    protected_patterns[protection_key] = match.group(0)
                    protected_text = protected_text.replace(match.group(0), protection_key, 1)
                    protection_counter += 1
                
                # Pattern 3: Combined pattern - thousand separators with decimal
                # e.g., "1.500.000,50" "25.750,75"
                combined_pattern = re.compile(r'\b\d{1,3}(?:\.\d{3})+,\d+\b')
                for match in combined_pattern.finditer(text):
                    protection_key = f'__COMBINED_{protection_counter}__'
                    protected_patterns[protection_key] = match.group(0)
                    protected_text = protected_text.replace(match.group(0), protection_key, 1)
                    protection_counter += 1
            
            # Protect percentages if configured
            if self.config.preserve_percentages:
                percentage_pattern = re.compile(r'\d+(?:[.,]\d+)?%')
                for match in percentage_pattern.finditer(protected_text):
                    protection_key = f'__PERCENT_{protection_counter}__'
                    protected_patterns[protection_key] = match.group(0)
                    protected_text = protected_text.replace(match.group(0), protection_key, 1)
                    protection_counter += 1
            
            # Now remove ALL punctuation from the protected text
            # This includes all dots, commas, colons, semicolons, quotes, etc.
            
            # Method 1: Use string.punctuation for ASCII punctuation
            translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
            result = protected_text.translate(translator)
            
            # Method 2: Also remove any Unicode punctuation that might have been missed
            # This handles Indonesian-specific punctuation if any
            result = re.sub(r'[^\w\s_]', ' ', result, flags=re.UNICODE)
            
            # Special handling for specific punctuation patterns
            # Remove ellipsis patterns that might have become spaces
            result = re.sub(r'\s{2,}', ' ', result)  # Multiple spaces to single space
            
            # Restore protected patterns
            for protection_key, original_value in protected_patterns.items():
                result = result.replace(protection_key, original_value)
            
            # Final cleanup
            # Remove extra spaces
            result = re.sub(r'\s+', ' ', result)
            # Remove spaces before protected numbers
            result = re.sub(r'\s+(\d+(?:[.,]\d+)?)', r' \1', result)
            # Strip leading/trailing spaces
            result = result.strip()
            
            # Additional cleanup for edge cases
            # Remove any remaining punctuation at the very end of the text
            result = re.sub(r'[^\w\s\d,.]$', '', result)
            
            if self.config.enable_debug_logging:
                logger.debug(f"Punctuation removal: '{text}' → '{result}'")
            
            return result
            
        except Exception as e:
            logger.warning(f"Punctuation removal failed: {e}")
            return text
    
    def _final_cleanup(self, text: str) -> str:
        """Final text cleanup - FIXED to prevent word truncation"""
        try:
            # Normalize whitespace FIRST
            if self.config.normalize_whitespace:
                text = re.sub(r'\s+', ' ', text).strip()
            
            # Convert to lowercase
            if self.config.lowercase_output:
                text = text.lower()
            
            # Apply stemming if enabled (but be very careful)
            if self.config.enable_stemming and self.stemmer:
                try:
                    words = text.split()
                    stemmed_words = []
                    for word in words:
                        # Don't stem number words or short words
                        if (len(word) <= 4 or 
                            word in ['satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh', 'puluh', 'ratus', 'ribu', 'juta', 'milyar', 'miliar']):
                            stemmed_words.append(word)
                        else:
                            stemmed_words.append(self.stemmer.stem(word))
                    text = ' '.join(stemmed_words)
                except Exception as e:
                    if self.config.enable_debug_logging:
                        logger.debug(f"Stemming failed: {e}")
                    # Don't apply stemming if it fails
            
            # Final whitespace cleanup
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.warning(f"Final cleanup failed: {e}")
            return str(text).lower().strip() if text else ""
    
    def _validate_output(self, original: str, normalized: str) -> bool:
        """Validate normalization output - IMPROVED"""
        if not self.config.enable_validation:
            return True
        
        try:
            issues = []
            
            # Check for excessive length changes (more permissive)
            if len(normalized) > len(original) * 5:  # Increased from 3 to 5
                issues.append("Excessive length increase")
            
            # Check for complete text loss
            if len(normalized.strip()) == 0 and len(original.strip()) > 0:
                issues.append("Complete text loss")
            
            # More lenient validation for digits
            if self.config.convert_numbers_to_words and re.search(r'\d', normalized):
                remaining_digits = re.findall(r'\d+', normalized)
                # Allow more digits to remain unconverted
                if len(remaining_digits) > len(original.split()) * 0.3:  # Increased threshold
                    issues.append(f"Many digits remain unconverted: {remaining_digits}")
            
            if issues:
                logger.warning(f"Validation issues (non-critical): {issues}")
                # Only fail in strict mode for critical issues
                if self.config.strict_mode and "Complete text loss" in issues:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return True  # Don't fail validation on exceptions
    
    def _remove_protection_markers(self, text: str) -> str:
        """Remove any protection markers added during processing"""
        if not text:
            return text
        
        # Remove zero-width spaces
        text = text.replace('\u200B', '')
        
        # Remove any phone markers that might remain
        text = text.replace('PHONESTART', '')
        text = text.replace('PHONEEND', '')
        text = text.replace('phone start', '')
        text = text.replace('phone end', '')
        
        # Clean up any remaining artifacts
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

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
            logger.debug(f"=== INDONESIAN ASR NORMALIZATION START ===")
            logger.debug(f"Input: '{text}'")
        
        try:
            original = text
            
            # Step 1: Unicode normalization
            text = self._normalize_unicode(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After Unicode normalization: '{text}'")
            
            # Step 2: Normalize old spelling
            text = self._normalize_old_spelling(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After old spelling normalization: '{text}'")
            
            # Step 3: Handle informal text
            text = self.informal_processor.normalize_informal(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After informal normalization: '{text}'")
            
            # Step 4: Normalize currency BEFORE number conversion
            text = self.currency_processor.normalize_currency(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After currency normalization: '{text}'")
            
            # Step 5: PROTECT phone numbers before general number conversion
            text = self.phone_processor.protect_phone_numbers(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After phone protection: '{text}'")
            
            # CRITICAL FIX: Move datetime normalization BEFORE number conversion
            # Step 6: Normalize time and date BEFORE numbers
            text = self.time_processor.normalize_datetime(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After datetime normalization: '{text}'")
            
            # Step 7: Convert numbers (now time expressions are already processed)
            text = self.number_converter.convert_numbers(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After number conversion: '{text}'")
            
            # Step 8: Process protected phone numbers
            text = self.phone_processor.normalize_phones(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After phone normalization: '{text}'")
            
            # Step 9: Handle code mixing
            text = self._handle_code_mixing(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After code mixing handling: '{text}'")
            
            # Step 10: Remove filler words
            text = self.filler_processor.remove_fillers(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After filler removal: '{text}'")
            
            # Step 11: Remove punctuation
            text = self._remove_punctuation(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After punctuation removal: '{text}'")
            
            # Step 12: Final cleanup
            text = self._final_cleanup(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After final cleanup: '{text}'")

            # Step 13: Remove protection markers
            text = self._remove_protection_markers(text)

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
            
class IndonesianASRNormaliser:
    """AudioBench-compatible Indonesian normaliser wrapper"""
    
    def __init__(self, enable_disfluency_removal: bool = True, **kwargs):
        """Initialize the Indonesian normalizer with AudioBench parameters"""
        self.enable_disfluency_removal = enable_disfluency_removal
        
        # Initialize your Indonesian normalizer config
        config = IndonesianASRConfig(
            remove_filler_words=enable_disfluency_removal,
            handle_informal_text=True,
            normalize_currency=True,
            convert_numbers_to_words=True,
            lowercase_output=True,
            normalize_whitespace=True,
            handle_code_mixing=True,
            use_external_normalizer=True,
            enable_validation=True
        )
        
        # Initialize the main normalizer
        self.normalizer = IndonesianASRNormalizer(config)
    
    def normalize(self, text: str) -> str:
        """Compatibility method for the new normalizer system"""
        return self.normalize_for_asr_eval(text)
    
    def normalize_for_asr_eval(self, text: str,
                               use_preprocessing: bool = True,
                               apply_asr_standardization: bool = True) -> str:
        """Main method called by AudioBench for text normalization"""
        if not text:
            return ""
        
        # Use your Indonesian normalizer's main normalize method
        return self.normalizer.normalize(text)