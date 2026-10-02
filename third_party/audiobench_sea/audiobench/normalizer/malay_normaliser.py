#!/usr/bin/env python3
"""
Malay ASR Normalizer for Fair WER Calculation
Based on Malaya library and optimized for ASR evaluation tasks
"""

import re
import logging
import string
import unicodedata
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import traceback

# External libraries for Malay processing
try:
    import malaya
    MALAYA_AVAILABLE = True
except ImportError:
    MALAYA_AVAILABLE = False
    print("Warning: malaya not available - install with: pip install malaya")

try:
    from num2words import num2words
    NUM2WORDS_AVAILABLE = True
except ImportError:
    NUM2WORDS_AVAILABLE = False
    print("Warning: num2words not available - install with: pip install num2words")

try:
    import phonenumbers
    PHONENUMBERS_AVAILABLE = True
except ImportError:
    PHONENUMBERS_AVAILABLE = False
    print("Warning: phonenumbers not available - install with: pip install phonenumbers")

try:
    import babel.dates
    import babel.numbers
    BABEL_AVAILABLE = True
except ImportError:
    BABEL_AVAILABLE = False
    print("Warning: babel not available - install with: pip install babel")

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
    module_name = "audiobench.normalizer.malay_normaliser"

logger = logging.getLogger(module_name)

@dataclass
class MalayASRConfig:
    """Configuration for Malay ASR normalization"""
    
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
    
    # Malay-specific features
    handle_informal_text: bool = True
    remove_filler_words: bool = True
    handle_code_mixing: bool = True
    handle_repeated_chars: bool = True
    handle_jawi_script: bool = False  # Rumi is default for ASR
    
    # Advanced options
    use_malaya_normalizer: bool = True
    enable_spelling_correction: bool = False  # Usually not needed for ASR
    enable_validation: bool = True
    enable_debug_logging: bool = False
    strict_mode: bool = False
    max_text_length: int = 500000


class MalayNumberConverter:
    """Convert Malay numbers between text and digits using Malaya"""
    
    def __init__(self, config: MalayASRConfig):
        self.config = config
        
        if not MALAYA_AVAILABLE:
            raise ImportError("Malaya is required for number conversion")
        
        # Initialize Malaya normalizer
        try:
            self.malaya_normalizer = malaya.normalizer.rules.load()
            logger.debug("Malaya normalizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Malaya normalizer: {e}")
            raise
        
        self._compile_patterns()
        self._setup_digit_mappings()
    
    def _setup_digit_mappings(self):
        """Setup digit to word mappings for decimal parts"""
        self.digit_to_word = {
            '0': 'kosong', '1': 'satu', '2': 'dua', '3': 'tiga', '4': 'empat',
            '5': 'lima', '6': 'enam', '7': 'tujuh', '8': 'lapan', '9': 'sembilan'
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for number detection"""
        # Pattern 1: Decimal numbers with decimal point
        # Matches: 1.5, 10.50, 0.001, 1,234.56
        self.decimal_pattern = re.compile(
            r'\b\d{1,3}(?:,\d{3})*\.\d+\b|\b\d+\.\d+\b'
        )
        
        # Pattern 2: Numbers with comma thousand separators (no decimal)
        # Matches: 1,234 or 1,234,567
        self.comma_thousand_pattern = re.compile(
            r'\b\d{1,3}(?:,\d{3})+\b'
        )
        
        # Pattern 3: Simple integers (no formatting)
        # Matches: 12, 1234, etc.
        self.simple_number_pattern = re.compile(
            r'\b\d+\b'
        )
    
    def _normalize_number_format(self, number_str: str) -> Tuple[str, bool]:
        """
        Normalize number format and detect if it has decimals
        Malaysian/English format only:
        - Comma (,) = thousand separator
        - Dot (.) = decimal separator
        Returns: (normalized_number, has_decimal)
        """
        # Remove any spaces
        number_str = number_str.strip()
        
        # If it has a dot, it's a decimal number
        if '.' in number_str:
            if ',' in number_str:
                # Format: 1,234.56 - has both thousand separator and decimal
                parts = number_str.split('.')
                if len(parts) == 2:
                    integer_part = parts[0].replace(',', '')  # Remove thousand separators
                    decimal_part = parts[1]
                    return f"{integer_part}.{decimal_part}", True
            else:
                # Format: 123.45 - just decimal, no thousand separator
                return number_str, True
        
        # If it only has commas, they are thousand separators
        elif ',' in number_str:
            # Format: 1,234 or 1,234,567 - remove thousand separators
            clean_number = number_str.replace(',', '')
            return clean_number, False
        
        # No special formatting - just a plain number
        else:
            return number_str, False
    
    def _convert_integer_with_malaya(self, integer_str: str) -> str:
        """Convert integer to words using Malaya"""
        try:
            if not integer_str or not integer_str.isdigit():
                return integer_str
            
            # Convert to integer
            num = int(integer_str)
            
            # Use Malaya's num2word directly
            result = malaya.num2word.to_cardinal(num)
            
            if self.config.enable_debug_logging:
                logger.debug(f"Malaya num2word: {num} → '{result}'")
            
            return result
            
        except Exception as e:
            logger.warning(f"Malaya num2word failed for '{integer_str}': {e}")
            # Return original if conversion fails
            return integer_str
    
    def _convert_decimal_digits(self, decimal_str: str) -> str:
        """Convert decimal digits to individual words"""
        if not decimal_str:
            return ""
        
        words = []
        for digit in decimal_str:
            if digit in self.digit_to_word:
                words.append(self.digit_to_word[digit])
            elif digit.isdigit():
                # Fallback to Malaya for unknown digits
                try:
                    word = malaya.num2word.to_cardinal(int(digit))
                    words.append(word)
                except:
                    words.append(digit)
        
        return ' '.join(words)
    
    def convert_number_to_words(self, number_str: str) -> str:
        """
        Convert a number string to Malay words
        """
        if not number_str:
            return ""
        
        try:
            # For whole numbers, try using Malaya directly first
            if '.' not in number_str and ',' not in number_str:
                try:
                    # Try to parse as integer
                    num = int(number_str)
                    result = malaya.num2word.to_cardinal(num)
                    if self.config.enable_debug_logging:
                        logger.debug(f"Direct Malaya conversion: '{number_str}' → '{result}'")
                    return result
                except:
                    pass
            
            # For more complex formats, normalize first
            normalized, has_decimal = self._normalize_number_format(number_str)
            
            if self.config.enable_debug_logging:
                logger.debug(f"Normalized '{number_str}' → '{normalized}' (decimal: {has_decimal})")
            
            if has_decimal:
                # Split into integer and decimal parts
                parts = normalized.split('.')
                integer_part = parts[0] if parts[0] else "0"
                decimal_part = parts[1] if len(parts) > 1 else ""
                
                # Convert integer part using Malaya
                integer_words = self._convert_integer_with_malaya(integer_part)
                
                # Convert decimal part digit by digit
                if decimal_part:
                    decimal_words = self._convert_decimal_digits(decimal_part)
                    return f"{integer_words} perpuluhan {decimal_words}"
                else:
                    return integer_words
            else:
                # No decimal, just convert the whole number
                return self._convert_integer_with_malaya(normalized)
                
        except Exception as e:
            logger.error(f"Number conversion failed for '{number_str}': {e}")
            return number_str
    
    def convert_numbers(self, text: str) -> str:
        """
        Convert all numbers in text to words
        """
        if not text or not self.config.convert_numbers_to_words:
            return text
        
        result = text
        
        try:
            # Process patterns in specific order
            # 1. First, find all decimal numbers
            decimal_matches = list(self.decimal_pattern.finditer(result))
            
            # 2. Then find comma-formatted numbers
            comma_matches = list(self.comma_thousand_pattern.finditer(result))
            
            # 3. Finally, find simple numbers
            simple_matches = list(self.simple_number_pattern.finditer(result))
            
            # Combine all matches and remove duplicates based on position
            all_matches = []
            processed_positions = set()
            
            # Process in order of specificity
            for matches in [decimal_matches, comma_matches, simple_matches]:
                for match in matches:
                    start, end = match.span()
                    # Check if this position was already processed
                    if not any(start >= p[0] and end <= p[1] for p in processed_positions):
                        all_matches.append(match)
                        processed_positions.add((start, end))
            
            # Sort by position (reverse order to maintain string positions)
            all_matches.sort(key=lambda m: m.start(), reverse=True)
            
            # Process each match
            for match in all_matches:
                number_str = match.group(0)
                start, end = match.span()
                
                # Skip if it's part of a percentage and we want to preserve those
                if self.config.preserve_percentages and end < len(result) and result[end] == '%':
                    continue
                
                # Skip if it's part of a currency prefix (RM, USD, etc.)
                if start >= 2 and result[start-2:start].upper() in ['RM', 'S$', 'R$']:
                    continue
                if start >= 3 and result[start-3:start].upper() in ['USD', 'EUR', 'GBP', 'SGD', 'MYR', 'IDR', 'JPY']:
                    continue
                
                # Convert the number
                words = self.convert_number_to_words(number_str)
                
                # Only replace if we got a valid word conversion
                if words and words != number_str:
                    result = result[:start] + words + result[end:]
                    
                    if self.config.enable_debug_logging:
                        logger.debug(f"Replaced '{number_str}' with '{words}'")
            
            return result
            
        except Exception as e:
            logger.error(f"Number conversion in text failed: {e}")
            return text
    
    def _use_malaya_normalizer(self, text: str) -> str:
        """
        Alternative method using Malaya normalizer directly
        Note: This returns a dict, so we need to extract the normalized text
        """
        try:
            result = self.malaya_normalizer.normalize(text)
            
            # Extract normalized text from the dict
            if isinstance(result, dict):
                normalized_text = result.get('normalize', text)
                return normalized_text
            elif isinstance(result, str):
                return result
            else:
                return str(result)
                
        except Exception as e:
            logger.warning(f"Malaya normalizer failed: {e}")
            return text


class MalayCurrencyProcessor:
    """Process Malaysian currency formats"""
    
    def __init__(self, config: MalayASRConfig):
        self.config = config
        self.number_converter = MalayNumberConverter(config)
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile currency patterns"""
        self.patterns = [
            # Malaysian Ringgit patterns
            # RM with amount (handle both RM1,234.56 and RM1,234 . 56 formats)
            (re.compile(r'RM\s*(\d+(?:,\d{3})*)\s*\.\s*(\d+)\b', re.IGNORECASE), 
             self._convert_ringgit_decimal_split),
            (re.compile(r'RM\s*(\d+(?:,\d{3})*(?:\.\d+)?)\b', re.IGNORECASE), 
             self._convert_ringgit),
            
            # MYR format
            (re.compile(r'MYR\s*(\d+(?:,\d{3})*)\s*\.\s*(\d+)\b', re.IGNORECASE),
             self._convert_myr_decimal_split),
            (re.compile(r'MYR\s*(\d+(?:,\d{3})*(?:\.\d+)?)\b', re.IGNORECASE),
             self._convert_myr),
            
            # Ringgit suffix
            (re.compile(r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*ringgit\b', re.IGNORECASE),
             self._convert_ringgit_suffix),
            
            # Sen (cents)
            (re.compile(r'(\d+)\s*sen\b', re.IGNORECASE),
             self._convert_sen),
            
            # Foreign currencies
            # SGD - Handle S$ properly
            (re.compile(r'[Ss]\$\s*(\d+(?:,\d{3})*)\s*\.\s*(\d+)\b'),
             self._convert_sgd_decimal_split),
            (re.compile(r'[Ss]\$(\d+(?:,\d{3})*(?:\.\d+)?)\b'),
             self._convert_sgd),
            (re.compile(r'SGD\s*(\d+(?:,\d{3})*)\s*\.\s*(\d+)\b', re.IGNORECASE),
             self._convert_sgd_code_decimal_split),
            (re.compile(r'SGD\s*(\d+(?:,\d{3})*(?:\.\d+)?)\b', re.IGNORECASE),
             self._convert_sgd_code),
            
            # USD - Use negative lookbehind to avoid matching S$
            (re.compile(r'(?<![Ss])\$\s*(\d+(?:,\d{3})*)\s*\.\s*(\d+)\b'), 
             self._convert_dollar_decimal_split),
            (re.compile(r'(?<![Ss])\$(\d+(?:,\d{3})*(?:\.\d+)?)\b'), 
             self._convert_dollar),
            (re.compile(r'USD\s*(\d+(?:,\d{3})*)\s*\.\s*(\d+)\b', re.IGNORECASE),
             self._convert_usd_decimal_split),
            (re.compile(r'USD\s*(\d+(?:,\d{3})*(?:\.\d+)?)\b', re.IGNORECASE),
             self._convert_usd),
            
            # EUR
            (re.compile(r'€\s*(\d+(?:,\d{3})*)\s*\.\s*(\d+)\b'),
             self._convert_euro_decimal_split),
            (re.compile(r'€(\d+(?:,\d{3})*(?:\.\d+)?)\b'),
             self._convert_euro),
            (re.compile(r'EUR\s*(\d+(?:,\d{3})*)\s*\.\s*(\d+)\b', re.IGNORECASE),
             self._convert_eur_decimal_split),
            (re.compile(r'EUR\s*(\d+(?:,\d{3})*(?:\.\d+)?)\b', re.IGNORECASE),
             self._convert_eur),
            
            # GBP
            (re.compile(r'£\s*(\d+(?:,\d{3})*)\s*\.\s*(\d+)\b'),
             self._convert_pound_decimal_split),
            (re.compile(r'£(\d+(?:,\d{3})*(?:\.\d+)?)\b'),
             self._convert_pound),
            (re.compile(r'GBP\s*(\d+(?:,\d{3})*)\s*\.\s*(\d+)\b', re.IGNORECASE),
             self._convert_gbp_decimal_split),
            (re.compile(r'GBP\s*(\d+(?:,\d{3})*(?:\.\d+)?)\b', re.IGNORECASE),
             self._convert_gbp),
        ]
    
    def _convert_amount_to_words(self, amount_str):
        """Convert currency amount to words"""
        try:
            # Remove thousand separators
            clean_amount = amount_str.replace(',', '')
            
            # Check if there's a decimal point
            if '.' in clean_amount:
                parts = clean_amount.split('.')
                ringgit_part = parts[0]
                sen_part = parts[1]
                
                # Convert ringgit part
                if ringgit_part.isdigit():
                    ringgit_value = int(ringgit_part)
                    if MALAYA_AVAILABLE:
                        try:
                            ringgit_words = malaya.num2word.to_cardinal(ringgit_value)
                        except:
                            ringgit_words = self.number_converter._number_to_words(ringgit_value)
                    else:
                        ringgit_words = self.number_converter._number_to_words(ringgit_value)
                    
                    # Convert sen part
                    if sen_part and int(sen_part) > 0:
                        # Pad to 2 digits if needed
                        if len(sen_part) == 1:
                            sen_part += '0'
                        sen_value = int(sen_part[:2])
                        
                        if MALAYA_AVAILABLE:
                            try:
                                sen_words = malaya.num2word.to_cardinal(sen_value)
                            except:
                                sen_words = self.number_converter._number_to_words(sen_value)
                        else:
                            sen_words = self.number_converter._number_to_words(sen_value)
                        
                        return ringgit_words, sen_words
                    else:
                        return ringgit_words, None
                else:
                    return amount_str, None
            else:
                # No decimal, just convert the integer
                if clean_amount.isdigit():
                    value = int(clean_amount)
                    if MALAYA_AVAILABLE:
                        try:
                            words = malaya.num2word.to_cardinal(value)
                        except:
                            words = self.number_converter._number_to_words(value)
                    else:
                        words = self.number_converter._number_to_words(value)
                    return words, None
                else:
                    return amount_str, None
                    
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Amount to words conversion failed: {e}")
            return amount_str, None
    
    def _convert_split_amount_to_words(self, ringgit_str, sen_str):
        """Convert split currency amount (when decimal is separated) to words"""
        try:
            # Convert ringgit part
            clean_ringgit = ringgit_str.replace(',', '')
            if clean_ringgit.isdigit():
                ringgit_value = int(clean_ringgit)
                if MALAYA_AVAILABLE:
                    try:
                        ringgit_words = malaya.num2word.to_cardinal(ringgit_value)
                    except:
                        ringgit_words = self.number_converter._number_to_words(ringgit_value)
                else:
                    ringgit_words = self.number_converter._number_to_words(ringgit_value)
            else:
                ringgit_words = ringgit_str
            
            # Convert sen part
            if sen_str and sen_str.isdigit() and int(sen_str) > 0:
                # Pad to 2 digits if needed
                if len(sen_str) == 1:
                    sen_str += '0'
                sen_value = int(sen_str[:2])
                
                if MALAYA_AVAILABLE:
                    try:
                        sen_words = malaya.num2word.to_cardinal(sen_value)
                    except:
                        sen_words = self.number_converter._number_to_words(sen_value)
                else:
                    sen_words = self.number_converter._number_to_words(sen_value)
                
                return ringgit_words, sen_words
            else:
                return ringgit_words, None
                
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Split amount conversion failed: {e}")
            return ringgit_str, None
    
    # Ringgit converters
    def _convert_ringgit_decimal_split(self, match):
        """Convert RM format with split decimal (RM1,234 . 56)"""
        ringgit_str = match.group(1)
        sen_str = match.group(2)
        
        ringgit_words, sen_words = self._convert_split_amount_to_words(ringgit_str, sen_str)
        
        if ringgit_words == 'kosong' and sen_words:
            return f"{sen_words} sen"
        elif sen_words:
            return f"{ringgit_words} ringgit {sen_words} sen"
        else:
            return f"{ringgit_words} ringgit"
    
    def _convert_ringgit(self, match):
        """Convert RM format"""
        amount_str = match.group(1)
        
        ringgit_words, sen_words = self._convert_amount_to_words(amount_str)
        
        # Special case: if ringgit is zero and there are sen
        if ringgit_words == 'kosong' and sen_words:
            return f"{sen_words} sen"
        elif sen_words:
            return f"{ringgit_words} ringgit {sen_words} sen"
        else:
            return f"{ringgit_words} ringgit"
    
    def _convert_myr_decimal_split(self, match):
        """Convert MYR format with split decimal"""
        ringgit_str = match.group(1)
        sen_str = match.group(2)
        
        ringgit_words, sen_words = self._convert_split_amount_to_words(ringgit_str, sen_str)
        
        if ringgit_words == 'kosong' and sen_words:
            return f"{sen_words} sen"
        elif sen_words:
            return f"{ringgit_words} ringgit {sen_words} sen"
        else:
            return f"{ringgit_words} ringgit"
    
    def _convert_myr(self, match):
        """Convert MYR format"""
        amount_str = match.group(1)
        
        ringgit_words, sen_words = self._convert_amount_to_words(amount_str)
        
        # Special case: if ringgit is zero and there are sen
        if ringgit_words == 'kosong' and sen_words:
            return f"{sen_words} sen"
        elif sen_words:
            return f"{ringgit_words} ringgit {sen_words} sen"
        else:
            return f"{ringgit_words} ringgit"
    
    def _convert_ringgit_suffix(self, match):
        """Convert ringgit suffix format"""
        amount_str = match.group(1)
        
        ringgit_words, sen_words = self._convert_amount_to_words(amount_str)
        
        # Special case: if ringgit is zero and there are sen
        if ringgit_words == 'kosong' and sen_words:
            return f"{sen_words} sen"
        elif sen_words:
            return f"{ringgit_words} ringgit {sen_words} sen"
        else:
            return f"{ringgit_words} ringgit"
    
    def _convert_sen(self, match):
        """Convert sen (cents)"""
        amount_str = match.group(1)
        
        if amount_str.isdigit():
            value = int(amount_str)
            if MALAYA_AVAILABLE:
                try:
                    words = malaya.num2word.to_cardinal(value)
                except:
                    words = self.number_converter._number_to_words(value)
            else:
                words = self.number_converter._number_to_words(value)
            return f"{words} sen"
        
        return match.group(0)
    
    # USD converters
    def _convert_dollar_decimal_split(self, match):
        """Convert $ format with split decimal"""
        dollar_str = match.group(1)
        cent_str = match.group(2)
        
        dollar_words, cent_words = self._convert_split_amount_to_words(dollar_str, cent_str)
        
        if cent_words:
            return f"{dollar_words} dolar {cent_words} sen"
        else:
            return f"{dollar_words} dolar"
    
    def _convert_dollar(self, match):
        """Convert $ format"""
        amount_str = match.group(1)
        
        dollar_words, cent_words = self._convert_amount_to_words(amount_str)
        
        if cent_words:
            return f"{dollar_words} dolar {cent_words} sen"
        else:
            return f"{dollar_words} dolar"
    
    def _convert_usd_decimal_split(self, match):
        """Convert USD format with split decimal"""
        dollar_str = match.group(1)
        cent_str = match.group(2)
        
        dollar_words, cent_words = self._convert_split_amount_to_words(dollar_str, cent_str)
        
        if cent_words:
            return f"{dollar_words} dolar {cent_words} sen amerika"
        else:
            return f"{dollar_words} dolar amerika"
    
    def _convert_usd(self, match):
        """Convert USD format"""
        amount_str = match.group(1)
        
        dollar_words, cent_words = self._convert_amount_to_words(amount_str)
        
        if cent_words:
            return f"{dollar_words} dolar {cent_words} sen amerika"
        else:
            return f"{dollar_words} dolar amerika"
    
    # SGD converters
    def _convert_sgd_decimal_split(self, match):
        """Convert S$ format with split decimal"""
        dollar_str = match.group(1)
        cent_str = match.group(2)
        
        dollar_words, cent_words = self._convert_split_amount_to_words(dollar_str, cent_str)
        
        if cent_words:
            return f"{dollar_words} dolar {cent_words} sen singapura"
        else:
            return f"{dollar_words} dolar singapura"
    
    def _convert_sgd(self, match):
        """Convert S$ format"""
        amount_str = match.group(1)
        
        dollar_words, cent_words = self._convert_amount_to_words(amount_str)
        
        if cent_words:
            return f"{dollar_words} dolar {cent_words} sen singapura"
        else:
            return f"{dollar_words} dolar singapura"
    
    def _convert_sgd_code_decimal_split(self, match):
        """Convert SGD format with split decimal"""
        dollar_str = match.group(1)
        cent_str = match.group(2)
        
        dollar_words, cent_words = self._convert_split_amount_to_words(dollar_str, cent_str)
        
        if cent_words:
            return f"{dollar_words} dolar {cent_words} sen singapura"
        else:
            return f"{dollar_words} dolar singapura"
    
    def _convert_sgd_code(self, match):
        """Convert SGD format"""
        amount_str = match.group(1)
        
        dollar_words, cent_words = self._convert_amount_to_words(amount_str)
        
        if cent_words:
            return f"{dollar_words} dolar {cent_words} sen singapura"
        else:
            return f"{dollar_words} dolar singapura"
    
    # EUR converters
    def _convert_euro_decimal_split(self, match):
        """Convert € format with split decimal"""
        euro_str = match.group(1)
        cent_str = match.group(2)
        
        euro_words, cent_words = self._convert_split_amount_to_words(euro_str, cent_str)
        
        if cent_words:
            return f"{euro_words} euro {cent_words} sen"
        else:
            return f"{euro_words} euro"
    
    def _convert_euro(self, match):
        """Convert € format"""
        amount_str = match.group(1)
        
        euro_words, cent_words = self._convert_amount_to_words(amount_str)
        
        if cent_words:
            return f"{euro_words} euro {cent_words} sen"
        else:
            return f"{euro_words} euro"
    
    def _convert_eur_decimal_split(self, match):
        """Convert EUR format with split decimal"""
        euro_str = match.group(1)
        cent_str = match.group(2)
        
        euro_words, cent_words = self._convert_split_amount_to_words(euro_str, cent_str)
        
        if cent_words:
            return f"{euro_words} euro {cent_words} sen"
        else:
            return f"{euro_words} euro"
    
    def _convert_eur(self, match):
        """Convert EUR format"""
        amount_str = match.group(1)
        
        euro_words, cent_words = self._convert_amount_to_words(amount_str)
        
        if cent_words:
            return f"{euro_words} euro {cent_words} sen"
        else:
            return f"{euro_words} euro"
    
    # GBP converters
    def _convert_pound_decimal_split(self, match):
        """Convert £ format with split decimal"""
        pound_str = match.group(1)
        pence_str = match.group(2)
        
        pound_words, pence_words = self._convert_split_amount_to_words(pound_str, pence_str)
        
        if pence_words:
            return f"{pound_words} paun {pence_words} pens"
        else:
            return f"{pound_words} paun"
    
    def _convert_pound(self, match):
        """Convert £ format - just 'paun' without 'british'"""
        amount_str = match.group(1)
        
        pound_words, pence_words = self._convert_amount_to_words(amount_str)
        
        if pence_words:
            return f"{pound_words} paun {pence_words} pens"
        else:
            return f"{pound_words} paun"
    
    def _convert_gbp_decimal_split(self, match):
        """Convert GBP format with split decimal"""
        pound_str = match.group(1)
        pence_str = match.group(2)
        
        pound_words, pence_words = self._convert_split_amount_to_words(pound_str, pence_str)
        
        if pence_words:
            return f"{pound_words} paun british {pence_words} pens"
        else:
            return f"{pound_words} paun british"
    
    def _convert_gbp(self, match):
        """Convert GBP format"""
        amount_str = match.group(1)
        
        pound_words, pence_words = self._convert_amount_to_words(amount_str)
        
        if pence_words:
            return f"{pound_words} paun british {pence_words} pens"
        else:
            return f"{pound_words} paun british"
    
    def normalize_currency(self, text: str) -> str:
        """Normalize currency in text"""
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
            
            # Final cleanup
            result = re.sub(r'\s+', ' ', result).strip()
            
        except Exception as e:
            logger.warning(f"Currency normalization failed: {e}")
            return text
        
        return result


# Replace the entire MalayPhoneProcessor class with this fixed version:

class MalayPhoneProcessor:
    """Process Malaysian phone number formats"""
    
    def __init__(self, config: MalayASRConfig):
        self.config = config
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile phone number patterns - FIXED to handle Malaya tokenization"""
        # Malaysian phone patterns - ORDER MATTERS (most specific first)
        self.patterns = [
            # FIXED: Handle Malaya's tokenized format for + numbers (with space after +)
            (re.compile(r'\+\s*60\s*([1-9]\d{8,9})\b'), 
            self._convert_international_continuous),
            (re.compile(r'\+\s*60\s*([\d\s\-]+)'), 
            self._convert_international_spaced),
            
            # Original patterns (kept for non-tokenized input)
            (re.compile(r'\+60([1-9]\d{8,9})\b'), 
            self._convert_international_continuous),
            (re.compile(r'\+60\s*[\-]?\s*([1-9][\d\s\-]+)'), 
            self._convert_international_spaced),
            
            # Format with 60 (no +) - continuous
            (re.compile(r'\b60([1-9]\d{8,9})\b'),
            self._convert_international_no_plus_continuous),
            
            # Format with 60 (no +) - with spaces
            (re.compile(r'\b60\s+([1-9][\d\s\-]+)\b'),
            self._convert_international_no_plus_spaced),
            
            # Mobile numbers starting with 01 (continuous)
            (re.compile(r'\b(01[0-9]\d{7,8})\b'),
            self._convert_mobile_continuous),
            
            # Mobile numbers starting with 01 (with spaces/dashes)
            (re.compile(r'\b(01[0-9][\s\-]?[\d\s\-]+)\b'),
            self._convert_mobile_spaced),
            
            # Landline numbers starting with 03-09 (continuous)
            (re.compile(r'\b(0[3-9]\d{7,8})\b'),
            self._convert_landline_continuous),
            
            # Landline numbers starting with 03-09 (with spaces/dashes)
            (re.compile(r'\b(0[3-9][\s\-]?[\d\s\-]+)\b'),
            self._convert_landline_spaced),
            
            # FIXED: Toll-free and special numbers - handle tokenized format
            # These patterns now handle both '1-300-88-1234' and '1 - 300 - 88 - 1234'
            (re.compile(r'\b(1\s*-?\s*300\s*-?\s*\d{2}\s*-?\s*\d{4})\b'),
            self._convert_toll_free),
            (re.compile(r'\b(1\s*-?\s*800\s*-?\s*\d{2}\s*-?\s*\d{4})\b'),
            self._convert_toll_free),
            
            # Original toll-free patterns (kept for backward compatibility)
            (re.compile(r'\b(1[\s\-]?300[\s\-]?\d{2}[\s\-]?\d{4})\b'),
            self._convert_toll_free),
            (re.compile(r'\b(1[\s\-]?800[\s\-]?\d{2}[\s\-]?\d{4})\b'),
            self._convert_toll_free),
        ]
    
    def _digits_to_words(self, digits: str) -> str:
        """Convert phone digits to Malay words"""
        digit_words = {
            '0': 'kosong', '1': 'satu', '2': 'dua', '3': 'tiga', '4': 'empat',
            '5': 'lima', '6': 'enam', '7': 'tujuh', '8': 'lapan', '9': 'sembilan'
        }
        
        # Clean the input - remove non-digit characters
        clean_digits = re.sub(r'[^\d]', '', digits)
        
        result = []
        for char in clean_digits:
            if char.isdigit():
                result.append(digit_words.get(char, char))
        
        return ' '.join(result)
    
    def _convert_international_continuous(self, match):
        """Convert +60xxxxxxxxx format - FIXED to use 'plus'"""
        remaining = match.group(1)
        all_digits = '60' + remaining
        digit_words = self._digits_to_words(all_digits)
        return 'plus ' + digit_words  # FIXED: Changed from 'tambah' to 'plus'
    
    def _convert_international_spaced(self, match):
        """Convert +60 xx xxx xxxx format - FIXED to use 'plus'"""
        remaining = match.group(1)
        all_digits = '60' + re.sub(r'[^\d]', '', remaining)
        digit_words = self._digits_to_words(all_digits)
        return 'plus ' + digit_words  # FIXED: Changed from 'tambah' to 'plus'
    
    def _convert_international_no_plus_continuous(self, match):
        """Convert 60xxxxxxxxx format (no plus) - FIXED to use 'plus'"""
        remaining = match.group(1)
        all_digits = '60' + remaining
        digit_words = self._digits_to_words(all_digits)
        return 'plus ' + digit_words  # FIXED: Changed from 'tambah' to 'plus'
    
    def _convert_international_no_plus_spaced(self, match):
        """Convert 60 xx xxx xxxx format (no plus) - FIXED to use 'plus'"""
        remaining = match.group(1)
        all_digits = '60' + re.sub(r'[^\d]', '', remaining)
        digit_words = self._digits_to_words(all_digits)
        return 'plus ' + digit_words  # FIXED: Changed from 'tambah' to 'plus'
    
    def _convert_mobile_continuous(self, match):
        """Convert 01xxxxxxxxx format"""
        full_number = match.group(1)
        return self._digits_to_words(full_number)
    
    def _convert_mobile_spaced(self, match):
        """Convert 01x-xxx-xxxx format"""
        full_number = match.group(1)
        return self._digits_to_words(full_number)
    
    def _convert_landline_continuous(self, match):
        """Convert 0xxxxxxxxxx format"""
        full_number = match.group(1)
        return self._digits_to_words(full_number)
    
    def _convert_landline_spaced(self, match):
        """Convert 0x-xxxxxxxx format"""
        full_number = match.group(1)
        return self._digits_to_words(full_number)
    
    def _convert_toll_free(self, match):
        """Convert toll-free numbers"""
        full_number = match.group(1)
        return self._digits_to_words(full_number)
    
    def protect_phones(self, text: str) -> str:
        """Add protection markers to phone numbers before number conversion - FIXED"""
        if not self.config.normalize_phone_numbers or not text:
            return text or "", {}
        
        protected_text = text
        
        # Apply all patterns to mark phone numbers with placeholders
        phone_placeholders = {}
        placeholder_counter = 0
        
        # FIXED: Sort patterns by length of match to process longer matches first
        all_matches = []
        for pattern, converter in self.patterns:
            for match in pattern.finditer(protected_text):
                all_matches.append((match, converter))
        
        # Sort by start position (reverse) to maintain string positions
        all_matches.sort(key=lambda x: x[0].start(), reverse=True)
        
        # Process matches from end to beginning
        for match, converter in all_matches:
            placeholder = f"__PHONE_{placeholder_counter}__"
            
            # CRITICAL FIX: Check if the match includes trailing whitespace
            matched_text = match.group(0)
            start, end = match.span()
            
            # Count trailing spaces in the matched text
            trailing_spaces = len(matched_text) - len(matched_text.rstrip())
            
            # Store the match without trailing spaces
            phone_placeholders[placeholder] = {
                'original': matched_text.rstrip(),  # Store without trailing spaces
                'converter': converter,
                'match': match,
                'trailing_spaces': trailing_spaces  # Store count of trailing spaces
            }
            
            # Replace with placeholder plus the trailing spaces
            replacement = placeholder + ' ' * trailing_spaces
            protected_text = protected_text[:start] + replacement + protected_text[end:]
            placeholder_counter += 1
        
        return protected_text, phone_placeholders
    
    def normalize_phones(self, text: str, phone_placeholders: dict = None) -> str:
        """Normalize phone numbers in text"""
        if not self.config.normalize_phone_numbers or not text:
            return text or ""
        
        result = text
        
        try:
            if phone_placeholders:
                # Replace placeholders with converted phone numbers
                for placeholder, info in phone_placeholders.items():
                    # Get the converted phone number
                    converted = info['converter'](info['match'])
                    
                    # Add back any trailing spaces that were in the original
                    trailing_spaces = info.get('trailing_spaces', 0)
                    converted_with_spaces = converted + ' ' * trailing_spaces
                    
                    # Replace the placeholder (which already has the spaces)
                    result = result.replace(placeholder + ' ' * trailing_spaces, converted_with_spaces)
                    
                    if self.config.enable_debug_logging:
                        logger.debug(f"Phone conversion: '{info['original']}' → '{converted}'")
            else:
                # Direct conversion (fallback)
                for pattern, converter in self.patterns:
                    old_result = result
                    result = pattern.sub(converter, result)
                    if self.config.enable_debug_logging and old_result != result:
                        logger.debug(f"Phone conversion: '{old_result}' → '{result}'")
            
        except Exception as e:
            logger.warning(f"Phone normalization failed: {e}")
            return text
        
        return result


class MalayDateTimeProcessor:
    """Process Malay date and time formats"""
    
    def __init__(self, config: MalayASRConfig):
        self.config = config
        self.number_converter = MalayNumberConverter(config)
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile date and time patterns - FIXED to handle Malaya tokenization"""
        self.patterns = [
            # Time patterns - ENHANCED to handle tokenized format
            # Time with pukul (handle both "pukul 9:00" and "pukul 9 : 00")
            (re.compile(r'\bpukul\s+(\d{1,2})\s*:\s*(\d{2})\b', re.IGNORECASE),
            self._convert_pukul_time),
            
            # Time with jam (handle both "jam 9:00" and "jam 9 : 00")
            (re.compile(r'\bjam\s+(\d{1,2})\s*:\s*(\d{2})\b', re.IGNORECASE),
            self._convert_jam_time),
            
            # Regular time formats (HH:MM) - handle spaces around colon
            (re.compile(r'\b(\d{1,2})\s*:\s*(\d{2})(?:\s*:\s*(\d{2}))?\b'),
            self._convert_time),
            
            # Date patterns
            # Date formats DD/MM/YYYY or DD-MM-YYYY
            (re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b'),
            self._convert_date),
            
            # Written months
            (re.compile(r'\b(\d{1,2})\s+(Januari|Februari|Mac|April|Mei|Jun|Julai|Ogos|September|Oktober|November|Disember)\s+(\d{4})\b', re.IGNORECASE),
            self._convert_written_date),
            
            # Written months without year
            (re.compile(r'\b(\d{1,2})\s+(Januari|Februari|Mac|April|Mei|Jun|Julai|Ogos|September|Oktober|November|Disember)\b', re.IGNORECASE),
            self._convert_written_date_no_year),
        ]
        
        # Month names
        self.month_names = {
            1: 'januari', 2: 'februari', 3: 'mac', 4: 'april',
            5: 'mei', 6: 'jun', 7: 'julai', 8: 'ogos',
            9: 'september', 10: 'oktober', 11: 'november', 12: 'disember'
        }
        
        # Day names
        self.day_names = {
            'monday': 'isnin', 'tuesday': 'selasa', 'wednesday': 'rabu',
            'thursday': 'khamis', 'friday': 'jumaat', 'saturday': 'sabtu',
            'sunday': 'ahad'
        }
    
    def _convert_time(self, match):
        """Convert time format to Malay words - FIXED to handle special cases"""
        try:
            hour = int(match.group(1))
            minute = int(match.group(2))
            second = int(match.group(3)) if match.group(3) else None
            
            if self.config.enable_debug_logging:
                logger.debug(f"Converting time: hour={hour}, minute={minute}")
            
            # Convert hour to words
            if MALAYA_AVAILABLE:
                try:
                    hour_words = malaya.num2word.to_cardinal(hour)
                except:
                    hour_words = self.number_converter._number_to_words(hour)
            else:
                hour_words = self.number_converter._number_to_words(hour)
            
            # Special case for midnight hours (00:XX)
            if hour == 0:
                # Always use "dua belas" for midnight hour
                hour_words = "dua belas"
                
                if minute == 0:
                    return "dua belas tengah malam"
                elif minute == 30:
                    return "dua belas setengah tengah malam"
                else:
                    # For any other minutes after midnight
                    if MALAYA_AVAILABLE:
                        try:
                            minute_words = malaya.num2word.to_cardinal(minute)
                        except:
                            minute_words = self.number_converter._number_to_words(minute)
                    else:
                        minute_words = self.number_converter._number_to_words(minute)
                    
                    return f"dua belas {minute_words} tengah malam"
            
            # Special case for noon (12:00)
            if hour == 12 and minute == 0:
                return "dua belas tengah hari"
            
            # Special case for exact hours - just return the hour number
            if minute == 0:
                return hour_words
            
            # Special case for half hours (setengah)
            if minute == 30:
                # For 12-hour format or times < 13:00, use "setengah"
                if hour < 13:
                    return f"{hour_words} setengah"
                else:
                    # For 24-hour format times >= 13:00, use numeric format
                    if MALAYA_AVAILABLE:
                        try:
                            minute_words = malaya.num2word.to_cardinal(minute)
                        except:
                            minute_words = self.number_converter._number_to_words(minute)
                    else:
                        minute_words = self.number_converter._number_to_words(minute)
                    
                    return f"{hour_words} {minute_words}"
            
            # For all other minutes - just return "hour minute" without "minit" suffix
            if MALAYA_AVAILABLE:
                try:
                    minute_words = malaya.num2word.to_cardinal(minute)
                except:
                    minute_words = self.number_converter._number_to_words(minute)
            else:
                minute_words = self.number_converter._number_to_words(minute)
            
            # Just return hour and minute words, no "minit" suffix
            time_str = f"{hour_words} {minute_words}"
            
            # Add seconds if present (keeping this part as is)
            if second is not None:
                if MALAYA_AVAILABLE:
                    try:
                        second_words = malaya.num2word.to_cardinal(second)
                    except:
                        second_words = self.number_converter._number_to_words(second)
                else:
                    second_words = self.number_converter._number_to_words(second)
                time_str += f" {second_words} saat"
            
            return time_str
                    
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Time conversion failed: {e}")
            return match.group(0)
    
    def _convert_pukul_time(self, match):
        """Convert 'pukul HH:MM' format - FIXED to handle special cases"""
        try:
            hour = int(match.group(1))
            minute = int(match.group(2))
            
            if self.config.enable_debug_logging:
                logger.debug(f"Converting pukul time: hour={hour}, minute={minute}")
            
            # Special handling for exact hours with pukul
            if minute == 0:
                if hour == 0:
                    return "pukul dua belas tengah malam"
                elif hour == 12:
                    return "pukul dua belas tengah hari"
                else:
                    # For other exact hours, just use "pukul [hour]"
                    if MALAYA_AVAILABLE:
                        try:
                            hour_words = malaya.num2word.to_cardinal(hour)
                        except:
                            hour_words = self.number_converter._number_to_words(hour)
                    else:
                        hour_words = self.number_converter._number_to_words(hour)
                    
                    return f"pukul {hour_words}"
            
            # Special handling for half hours with pukul
            if minute == 30:
                if hour == 0:
                    return "pukul dua belas setengah tengah malam"
                elif hour < 13:
                    # Based on test case: "pukul 9:30" should be "pukul sembilan setengah"
                    if MALAYA_AVAILABLE:
                        try:
                            hour_words = malaya.num2word.to_cardinal(hour)
                        except:
                            hour_words = self.number_converter._number_to_words(hour)
                    else:
                        hour_words = self.number_converter._number_to_words(hour)
                    
                    return f"pukul {hour_words} setengah"
                else:
                    # For 24-hour times, use numeric format
                    if MALAYA_AVAILABLE:
                        try:
                            hour_words = malaya.num2word.to_cardinal(hour)
                            minute_words = malaya.num2word.to_cardinal(minute)
                        except:
                            hour_words = self.number_converter._number_to_words(hour)
                            minute_words = self.number_converter._number_to_words(minute)
                    else:
                        hour_words = self.number_converter._number_to_words(hour)
                        minute_words = self.number_converter._number_to_words(minute)
                    
                    return f"pukul {hour_words} {minute_words}"
            
            # For other minutes, use regular conversion
            # Create a mock match object to reuse _convert_time logic
            time_match = type('MockMatch', (), {
                'group': lambda self, x: match.group(x) if x <= 2 else None
            })()
            time_result = self._convert_time(time_match)
            
            # Don't duplicate "pukul" if it's already in the result
            if time_result.startswith("pukul "):
                return time_result
            else:
                return f"pukul {time_result}"
                
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Pukul time conversion failed: {e}")
            return match.group(0)
    
    def _convert_jam_time(self, match):
        """Convert 'jam HH:MM' format"""
        try:
            hour = int(match.group(1))
            minute = int(match.group(2))
            
            # Create a mock match object to reuse _convert_time logic
            time_match = type('MockMatch', (), {
                'group': lambda self, x: match.group(x) if x <= 2 else None
            })()
            time_result = self._convert_time(time_match)
            
            return f"jam {time_result}"
                    
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Jam time conversion failed: {e}")
            return match.group(0)
    
    def _convert_date(self, match):
        """Convert date format to words"""
        try:
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3))
            
            # Convert to words
            if MALAYA_AVAILABLE:
                try:
                    day_words = malaya.num2word.to_cardinal(day)
                    year_words = malaya.num2word.to_cardinal(year)
                except:
                    day_words = self.number_converter._number_to_words(day)
                    year_words = self.number_converter._number_to_words(year)
            else:
                day_words = self.number_converter._number_to_words(day)
                year_words = self.number_converter._number_to_words(year)
            
            month_name = self.month_names.get(month, str(month))
            
            return f"{day_words} {month_name} {year_words}"
            
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
            
            if MALAYA_AVAILABLE:
                try:
                    day_words = malaya.num2word.to_cardinal(day)
                    year_words = malaya.num2word.to_cardinal(year)
                except:
                    day_words = self.number_converter._number_to_words(day)
                    year_words = self.number_converter._number_to_words(year)
            else:
                day_words = self.number_converter._number_to_words(day)
                year_words = self.number_converter._number_to_words(year)
            
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
            
            if MALAYA_AVAILABLE:
                try:
                    day_words = malaya.num2word.to_cardinal(day)
                except:
                    day_words = self.number_converter._number_to_words(day)
            else:
                day_words = self.number_converter._number_to_words(day)
            
            return f"{day_words} {month_name}"
            
        except Exception as e:
            if self.config.enable_debug_logging:
                logger.debug(f"Written date (no year) conversion failed: {e}")
            return match.group(0)
    
    def normalize_datetime(self, text: str) -> str:
        """Normalize date and time in text - ENHANCED to handle pre-processing"""
        if not (self.config.normalize_time_format or self.config.normalize_date_format) or not text:
            return text or ""
        
        result = text
        
        try:
            # CRITICAL: First, fix any tokenized time formats before applying patterns
            # This handles cases where Malaya has split "9:00" into "9 : 00"
            result = re.sub(r'(\d{1,2})\s+:\s+(\d{2})', r'\1:\2', result)
            
            # Apply patterns
            for pattern, converter in self.patterns:
                old_result = result
                result = pattern.sub(converter, result)
                if self.config.enable_debug_logging and old_result != result:
                    logger.debug(f"DateTime conversion: '{old_result}' -> '{result}'")
        except Exception as e:
            logger.warning(f"DateTime normalization failed: {e}")
            return text
        
        return result


class MalayFillerProcessor:
    """Process Malay filler words and discourse markers"""
    
    def __init__(self, config: MalayASRConfig):
        self.config = config
        self._setup_filler_lists()
    
    def _setup_filler_lists(self):
        """Setup Malay filler words"""
        # Common filler words and discourse markers
        self.filler_words = [
            # Discourse markers
            'kan', 'lah', 'loh', 'mah', 'deh', 'sih', 'dong', 'kok',
            'aduh', 'ah', 'eh', 'oh', 'uh', 'weh', 'woi', 'ke', 'je', 'keje',
            
            # Hesitation sounds
            'hmm', 'emm', 'eee', 'mmm', 'uhh', 'ehh', 'err', 'erm',
        ]
        
        # Repeated patterns
        self.repeated_patterns = []
    
    def remove_fillers(self, text: str) -> str:
        """Remove filler words from text"""
        if not self.config.remove_filler_words or not text:
            return text or ""
        
        result = text
        
        try:
            # Multiple passes
            for iteration in range(5):
                old_text = result
                
                # Remove repeated patterns
                for pattern, replacement in self.repeated_patterns:
                    result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                
                # Remove filler words
                for filler in self.filler_words:
                    # Use word boundaries to avoid partial matches
                    pattern = r'\b' + re.escape(filler) + r'\b'
                    result = re.sub(pattern, ' ', result, flags=re.IGNORECASE)
                
                # Clean up multiple spaces
                result = re.sub(r'\s+', ' ', result).strip()
                
                if result == old_text:
                    break
            
            return result
            
        except Exception as e:
            logger.warning(f"Filler removal failed: {e}")
            return text


class MalayInformalProcessor:
    """Handle informal Malay text (bahasa pasar)"""
    
    def __init__(self, config: MalayASRConfig):
        self.config = config
        self._setup_mappings()
    
    def _setup_mappings(self):
        """Setup informal to formal mappings"""
        # Common informal variations
        self.informal_mappings = {
            # Shortened words
            'tak': 'tidak', 'x': 'tidak', 'xde': 'tidak ada', 'xda': 'tidak ada',
            'dah': 'sudah', 'udah': 'sudah', 'da': 'sudah',
            'jer': 'sahaja', 'aje': 'sahaja', 'aje': 'sahaja',
            'ni': 'ini', 'tu': 'itu',
            'mcm': 'macam', 'camne': 'macam mana', 'cmne': 'macam mana',
            'nak': 'hendak', 'nk': 'hendak',
            'dgn': 'dengan', 'dg': 'dengan', 'ngan': 'dengan',
            'utk': 'untuk', 'u/': 'untuk',
            'dr': 'dari', 'dri': 'dari', 'drpd': 'daripada',
            'yg': 'yang', 'yng': 'yang',
            'jgn': 'jangan', 'jgn': 'jangan',
            'blh': 'boleh', 'bleh': 'boleh', 'bole': 'boleh',
            'sbb': 'sebab', 'psl': 'pasal', 'pasal': 'kerana',
            'tp': 'tetapi', 'tpi': 'tetapi', 'tapi': 'tetapi',
            'bg': 'bagi', 'bgi': 'bagi',
            'kt': 'kita', 'kte': 'kita', 'kite': 'kita',
            'org': 'orang', 'org2': 'orang-orang',
            'sape': 'siapa', 'sapa': 'siapa',
            'ape': 'apa', 'aper': 'apa', 'apekah': 'apakah',
            'bile': 'bila', 'biler': 'bila',
            'mane': 'mana', 'maner': 'mana', 'mn': 'mana',
            'knp': 'kenapa', 'knape': 'kenapa', 'nape': 'kenapa', 'naper': 'kenapa',
            'camne': 'macam mana', 'mcmane': 'macammana', 'bgmn': 'bagaimana',
            'gile': 'gila', 'giler': 'gila',
            'best': 'bagus', 'terbaik': 'terbaik',
            'ok': 'okey', 'okay': 'okey', 'tau':'tahu', 'brg': 'barang',
            
            # Pronouns
            'ku': 'aku','ko': 'engkau', 'kau': 'engkau',
            'korang': 'kamu semua', 'korg': 'kamu semua',
            'diorang': 'mereka', 'diorg': 'mereka', 'depa': 'mereka',
            'kitorang': 'kami', 'kitorg': 'kami',
            
            # Common typos and SMS language
            'mkn': 'makan', 'mknn': 'makanan',
            'mnm': 'minum', 'air': 'air',
            'tgk': 'tengok', 'tgok': 'tengok',
            'dgr': 'dengar', 'dgaq': 'dengar',
            'pgi': 'pergi', 'pg': 'pergi',
            'dtg': 'datang', 'dtng': 'datang',
            'blk': 'balik', 'balek': 'balik',
            'jwb': 'jawab', 'jwpn': 'jawapan',
            'ckp': 'cakap', 'ckap': 'cakap',
            'tmpt': 'tempat', 'tmpat': 'tempat',
            'smlm': 'semalam', 'mlm': 'malam', 'pg': 'pagi',
            'ptg': 'petang', 'ptng': 'petang',
            'hri': 'hari', 'ari': 'hari',
            'bln': 'bulan', 'thn': 'tahun',
            'skrg': 'sekarang', 'skang': 'sekarang',
            'td': 'tadi', 'tdi': 'tadi',
            'nnt': 'nanti', 'nnti': 'nanti',
            'lps': 'lepas', 'lpas': 'lepas', 'pas': 'lepas',
            'dpn': 'depan', 'dpan': 'depan',
            'blkg': 'belakang', 'blakang': 'belakang',
            'ats': 'atas', 'bwh': 'bawah',
            'dlm': 'dalam', 'luar': 'luar',
        }
        
        # Code-mixing patterns (Manglish)
        self.code_mixing_patterns = []
    
    def normalize_informal(self, text: str) -> str:
        """Normalize informal Malay text"""
        if not self.config.handle_informal_text or not text:
            return text or ""
        
        result = text.lower()  # Work with lowercase for matching
        
        try:
            # Apply informal mappings
            for informal, formal in self.informal_mappings.items():
                pattern = r'\b' + re.escape(informal) + r'\b'
                result = re.sub(pattern, formal, result, flags=re.IGNORECASE)
            
            # Apply code-mixing patterns if configured
            if self.config.handle_code_mixing:
                for pattern, replacement in self.code_mixing_patterns:
                    result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            
            # Handle repeated characters (e.g., "bestttt" -> "best")
            if self.config.handle_repeated_chars:
                # Only apply to non-numeric words
                words = result.split()
                processed_words = []
                for word in words:
                    if not any(c.isdigit() for c in word):
                        processed_words.append(re.sub(r'(.)\1{2,}', r'\1', word))
                    else:
                        processed_words.append(word)
                result = ' '.join(processed_words)
                
        except Exception as e:
            logger.warning(f"Informal text normalization failed: {e}")
            return text
        
        return result


class MalayASRNormalizer:
    """Main Malay ASR Normalizer class"""
    def __init__(self, config: Optional[MalayASRConfig] = None):
        self.config = config or MalayASRConfig()
        
        # Initialize processors FIRST (before Malaya) to ensure they always exist
        self.number_converter = MalayNumberConverter(self.config)
        self.currency_processor = MalayCurrencyProcessor(self.config)
        self.phone_processor = MalayPhoneProcessor(self.config)
        self.datetime_processor = MalayDateTimeProcessor(self.config)
        self.filler_processor = MalayFillerProcessor(self.config)
        self.informal_processor = MalayInformalProcessor(self.config)
        
        # Initialize Malaya components if available (AFTER processors)
        self.malaya_normalizer = None
        self.malaya_preprocessing = None
        
        if MALAYA_AVAILABLE and self.config.use_malaya_normalizer:
            try:
                # Load Malaya normalizer with safer configuration
                try:
                    # Try with normalize_entity parameter first
                    if self.config.enable_spelling_correction:
                        self.spell_corrector = malaya.spelling_correction.probability.load()
                        self.malaya_normalizer = malaya.normalizer.rules.load(
                            speller=self.spell_corrector,
                            normalize_entity=False
                        )
                    else:
                        self.malaya_normalizer = malaya.normalizer.rules.load(
                            normalize_entity=False
                        )
                except TypeError:
                    # If normalize_entity parameter not supported, use default
                    logger.warning("normalize_entity parameter not supported, using default normalizer")
                    if self.config.enable_spelling_correction:
                        self.spell_corrector = malaya.spelling_correction.probability.load()
                        self.malaya_normalizer = malaya.normalizer.rules.load(speller=self.spell_corrector)
                    else:
                        self.malaya_normalizer = malaya.normalizer.rules.load()
                
                # Setup preprocessing with safer parameters
                normalize_options = ['url', 'email']  # Only normalize URLs and emails
                
                try:
                    # Try with all parameters first
                    self.malaya_preprocessing = malaya.preprocessing.preprocessing(
                        normalize=normalize_options,
                        annotate=['elongated', 'repeated'],
                        lowercase=False,
                        expand_english_contractions=False,
                        translate_english_to_bm=False,
                        segmenter=None,
                        demoji=False
                    )
                except TypeError as e:
                    # If some parameters not supported, use minimal configuration
                    logger.warning(f"Some preprocessing parameters not supported: {e}")
                    self.malaya_preprocessing = malaya.preprocessing.preprocessing(
                        normalize=normalize_options,
                        annotate=['elongated', 'repeated'],
                        lowercase=False
                    )
                
                logger.info("Malaya components initialized successfully")
                
            except Exception as e:
                logger.warning(f"Failed to initialize Malaya components: {e}")
                self.malaya_normalizer = None
                self.malaya_preprocessing = None
        
        if self.config.enable_debug_logging:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.debug("Malay ASR Normalizer initialized")
    
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
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation while preserving important elements"""
        if not self.config.remove_punctuation or not text:
            return text or ""
        
        try:
            # First, handle percentage conversion if needed
            if self.config.preserve_percentages:
                # Convert percentages to words before removing punctuation
                percentage_pattern = re.compile(r'(\d+(?:[.,]\d+)?)\s*%')
                
                def convert_percentage(match):
                    number_str = match.group(1)
                    # Convert the number to words
                    words = self.number_converter.convert_number_to_words(number_str)
                    return f"{words} peratus"
                
                text = percentage_pattern.sub(convert_percentage, text)
            
            # Handle special notation that Malaya might have added
            # Remove "repeated" markers that Malaya preprocessing might add
            text = re.sub(r'\brepeated\b', '', text, flags=re.IGNORECASE)
            
            # Now handle punctuation removal
            # Create a set of characters to preserve
            preserved_chars = set()
            if self.config.preserve_decimal_points:
                # Only preserve decimal points that are part of numbers
                # We'll handle this by protecting decimal numbers first
                decimal_pattern = re.compile(r'\b\d+\.\d+\b')
                decimal_placeholders = {}
                placeholder_counter = 0
                
                for match in decimal_pattern.finditer(text):
                    placeholder = f"__DECIMAL_{placeholder_counter}__"
                    decimal_placeholders[placeholder] = match.group(0)
                    text = text.replace(match.group(0), placeholder, 1)
                    placeholder_counter += 1
            
            # Remove all punctuation including Unicode punctuation
            # Method 1: Remove ASCII punctuation
            translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
            text = text.translate(translator)
            
            # Method 2: Remove Unicode punctuation categories
            # This handles things like degree symbols (°), quotes, etc.
            chars = []
            for char in text:
                # Check Unicode category
                cat = unicodedata.category(char)
                if cat.startswith('P'):  # Punctuation
                    chars.append(' ')
                elif cat.startswith('S'):  # Symbols
                    chars.append(' ')
                else:
                    chars.append(char)
            text = ''.join(chars)
            
            # Restore protected decimal numbers if any
            if self.config.preserve_decimal_points and 'decimal_placeholders' in locals():
                for placeholder, original in decimal_placeholders.items():
                    text = text.replace(placeholder, original)
            
            # Clean up multiple spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Final cleanup: remove any trailing punctuation that might have been missed
            # This includes periods, commas, etc. at the very end
            text = re.sub(r'[\s\W]+$', '', text)
            
            # Remove leading punctuation as well
            text = re.sub(r'^[\s\W]+', '', text)
            
            # One more pass to clean up spaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.warning(f"Punctuation removal failed: {e}")
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
            
            # Check for excessive length changes
            if len(normalized) > len(original) * 5:
                issues.append("Excessive length increase")
            
            # Check for complete text loss
            if len(normalized.strip()) == 0 and len(original.strip()) > 0:
                issues.append("Complete text loss")
            
            # Check for remaining digits when they should be converted
            if self.config.convert_numbers_to_words and re.search(r'\d', normalized):
                remaining_digits = re.findall(r'\d+', normalized)
                # Allow some digits to remain (like in mixed alphanumeric)
                if len(remaining_digits) > len(original.split()) * 0.3:
                    issues.append(f"Many digits remain unconverted: {remaining_digits}")
            
            if issues:
                logger.warning(f"Validation issues: {issues}")
                if self.config.strict_mode:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return True
    
    def _protect_large_numbers(self, text: str) -> str:
        """
        Protect large numbers with commas from being split by preprocessing
        Replace them with placeholders temporarily
        """
        protected_text = text
        placeholders = {}
        placeholder_counter = 0
        
        # Find all comma-separated numbers (like 1,000,000 or 1,234,567)
        # This pattern matches numbers with multiple comma groups
        large_number_pattern = re.compile(r'\b\d{1,3}(?:,\d{3}){2,}\b')
        
        for match in large_number_pattern.finditer(text):
            number_str = match.group(0)
            placeholder = f"__LARGENUM_{placeholder_counter}__"
            placeholders[placeholder] = number_str
            protected_text = protected_text.replace(number_str, placeholder, 1)
            placeholder_counter += 1
        
        return protected_text, placeholders

    def _restore_large_numbers(self, text: str, placeholders: dict) -> str:
        """
        Restore the protected large numbers
        """
        for placeholder, original in placeholders.items():
            text = text.replace(placeholder, original)
        return text

    def _reconstruct_from_tokens(self, tokens: list) -> str:
        """
        Reconstruct text from Malaya tokens, handling currency symbols properly
        """
        if not tokens:
            return ""
        
        # First pass: merge currency symbols that were split
        merged_tokens = []
        i = 0
        while i < len(tokens):
            current = str(tokens[i])
            
            # Check if this is a split currency symbol
            if i + 1 < len(tokens):
                next_token = str(tokens[i + 1])
                
                # Handle S$ case
                if current.upper() == 'S' and next_token.startswith('$'):
                    merged_tokens.append(current + next_token)
                    i += 2
                    continue
                
                # Handle RM, EUR, GBP followed by number
                if current.upper() in ['RM', 'EUR', 'GBP', 'USD', 'MYR', 'SGD'] and next_token[0].isdigit():
                    # Check if there's a decimal point coming
                    full_amount = next_token
                    j = i + 2
                    
                    # Look for decimal pattern (. followed by digits)
                    if j + 1 < len(tokens) and tokens[j] == '.' and j + 2 < len(tokens) and tokens[j + 1].isdigit():
                        full_amount = next_token + '.' + tokens[j + 1]
                        i = j + 2
                    else:
                        i += 2
                    
                    merged_tokens.append(current + full_amount)
                    continue
            
            # Regular token
            merged_tokens.append(current)
            i += 1
        
        # Join the tokens with spaces
        return ' '.join(merged_tokens)

    def normalize(self, text: str) -> str:
        """Main normalization method - FIXED TO HANDLE TOKENIZED OUTPUT"""
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
            logger.debug(f"=== MALAY ASR NORMALIZATION START ===")
            logger.debug(f"Input: '{text}'")
        
        try:
            original = text
            
            # Step 1: Unicode normalization
            text = self._normalize_unicode(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After Unicode normalization: '{text}'")
            
            # CRITICAL FIX: Protect large numbers before preprocessing
            protected_text, number_placeholders = self._protect_large_numbers(text)
            
            # Step 2: Use Malaya preprocessing if available
            if self.malaya_preprocessing:
                try:
                    preprocessed = self.malaya_preprocessing.process(protected_text)
                    if isinstance(preprocessed, dict):
                        text = preprocessed.get('normalize', protected_text)
                    elif isinstance(preprocessed, list):
                        # CRITICAL FIX: Properly reconstruct text from tokenized list
                        # Join tokens but preserve the original spacing/structure for currency
                        text = self._reconstruct_from_tokens(preprocessed)
                    elif isinstance(preprocessed, str):
                        text = preprocessed
                    else:
                        text = str(preprocessed)
                    
                    if self.config.enable_debug_logging:
                        logger.debug(f"After Malaya preprocessing: '{text}'")
                except Exception as e:
                    logger.warning(f"Malaya preprocessing failed: {e}")
                    text = protected_text
            else:
                text = protected_text
            
            # Restore the protected numbers
            text = self._restore_large_numbers(text, number_placeholders)
            if self.config.enable_debug_logging:
                logger.debug(f"After restoring large numbers: '{text}'")
            
            # Skip Step 3 (Old spelling) - as you requested
            
            # Step 4: Handle informal text
            text = self.informal_processor.normalize_informal(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After informal normalization: '{text}'")
            
            # Step 5: Normalize currency (before numbers to handle currency amounts properly)
            text = self.currency_processor.normalize_currency(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After currency normalization: '{text}'")
            
            # CRITICAL: Step 5.5 - Protect phone numbers BEFORE general number conversion
            text, phone_placeholders = self.phone_processor.protect_phones(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After phone protection: '{text}'")
            
            # Step 6: Normalize date and time (moved before number conversion)
            text = self.datetime_processor.normalize_datetime(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After datetime normalization: '{text}'")
            
            # Step 7: Convert numbers (now phone numbers are protected)
            text = self.number_converter.convert_numbers(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After number conversion: '{text}'")
            
            # Step 8: Normalize protected phone numbers
            text = self.phone_processor.normalize_phones(text, phone_placeholders)
            if self.config.enable_debug_logging:
                logger.debug(f"After phone normalization: '{text}'")
            
            # Step 9: Remove filler words
            text = self.filler_processor.remove_fillers(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After filler removal: '{text}'")
            
            # Step 10: Remove punctuation
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
            
class MalayASRNormaliser:
    """AudioBench-compatible Malay normaliser"""
    
    def __init__(self, enable_disfluency_removal: bool = True, **kwargs):
        # Use your existing MalayASRConfig and MalayASRNormalizer
        config = MalayASRConfig(
            remove_filler_words=enable_disfluency_removal,
            handle_informal_text=True,
            normalize_currency=True,
            convert_numbers_to_words=True,
            lowercase_output=True,
            normalize_whitespace=True
        )
        self.normalizer = MalayASRNormalizer(config)
    
    def normalize(self, text: str) -> str:
        """Compatibility method for the new normalizer system"""
        return self.normalize_for_asr_eval(text)
    
    def normalize_for_asr_eval(self, text: str, 
                               use_preprocessing: bool = True,
                               apply_asr_standardization: bool = True) -> str:
        """Main method called by AudioBench"""
        if not text:
            return ""
        
        try:
            # Use the main MalayASRNormalizer
            return self.normalizer.normalize(text)
        except Exception as e:
            logger.error(f"Malay normalization failed: {e}")
            # Fallback: basic text cleaning
            import re
            text = re.sub(r'\s+', ' ', text.strip())
            return text