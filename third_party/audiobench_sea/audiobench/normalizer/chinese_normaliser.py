"""
Chinese ASR Normalizer
"""
import re
import logging
import string
import unicodedata
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import traceback

try:
    import cn2an
    import warnings
    # Suppress cn2an warnings for invalid format data
    warnings.filterwarnings("ignore", message="不符合格式的数据", category=UserWarning, module="cn2an")
    CN2AN_AVAILABLE = True
except ImportError:
    CN2AN_AVAILABLE = False
    print("Warning: cn2an not available - install with: pip install cn2an")

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
    module_name = "audiobench.normalizer.chinese_normaliser"

logger = logging.getLogger(module_name)

@dataclass
class ASRConfig:
    
    # Core features
    preserve_decimal_points: bool = True
    preserve_percentages: bool = True
    standardize_phone_format: bool = True
    use_telephony_format: bool = True
    normalize_currency: bool = True
    normalize_datetime: bool = True
    normalize_units: bool = True
    
    # Text cleanup
    remove_filler_words: bool = True
    remove_punctuation: bool = True
    handle_chinese_punctuation: bool = True
    normalize_whitespace: bool = True
    lowercase_output: bool = True
    
    enable_validation: bool = True
    enable_debug_logging: bool = False
    strict_mode: bool = False
    max_text_length: int = 500000


class ChineseNumberConverter:
    def __init__(self, config: ASRConfig):
        self.config = config
        self._setup_mappings()
        self._compile_patterns()
    
    def _setup_mappings(self):
        self.digit_map = {
            '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
            '壹': '1', '贰': '2', '叁': '3', '肆': '4', '伍': '5',
            '陆': '6', '柒': '7', '捌': '8', '玖': '9',
            '两': '2', '俩': '2'
        }
    
    def _compile_patterns(self):
        self.patterns = [
            # Decimal numbers
            (r'([一二三四五六七八九十百千万亿两零]+)点([一二三四五六七八九十零]+)(?![刻分时半])',
             self._convert_decimal),

            # Fractions
            (r'([一二三四五六七八九十百千万]+)分之([一二三四五六七八九十百千万]+)',
             self._convert_fraction),

            # X万两千Y → correct  ¥
            (r'([一二三四五六七八九十百千万亿]+)万两千([一二三四五六七八九百十零]+)',
             lambda m: str(
                 (cn2an.cn2an(m.group(1), "normal") if CN2AN_AVAILABLE
                  else int(''.join(self.digit_map[d] for d in m.group(1))))
                 * 10000
                 + 2000
                 + (cn2an.cn2an(m.group(2), "normal") if CN2AN_AVAILABLE
                    else int(''.join(self.digit_map[d] for d in m.group(2))))
             )),

            # Generic 万/千/百 patterns—allow multi‐char high part
            (r'([一二三四五六七八九十百千万亿]+)万([一二三四五六七八九千百十两零]*)',
             self._convert_complex_number),
            (r'([一二三四五六七八九十百千万亿]+)千([一二三四五六七八九十百两零]*)',
             self._convert_complex_number),
            (r'([一二三四五六七八九十百千万亿]+)百([一二三四五六七八九十两零]*)',
             self._convert_complex_number),

            # Tens and teens
            (r'([二三四五六七八九])十([一二三四五六七八九])(?![万千百十点分时])',
             self._convert_ten),
            (r'十([一二三四五六七八九])(?![万千百十点分时])',
             self._convert_teen),
            (r'([二三四五六七八九])十(?![一二三四五六七八九万千百十点分时])',
             self._convert_decades),

            # Special standalone cases
            (r'\b十\b(?![万千百点分时])', '10'),
            (r'\b两\b(?![万千百十点分时])', '2'),
            (r'\b([一二三四五六七八九零壹贰叁肆伍陆柒捌玖])\b(?![万千百十点分时刻])',
             self._convert_single),
        ]

        # Compile regexes
        self.compiled_patterns = []
        for pat, conv in self.patterns:
            if callable(conv):
                self.compiled_patterns.append((re.compile(pat), conv))
            else:
                self.compiled_patterns.append((re.compile(pat), lambda m, r=conv: r))
    
    def _try_cn2an(self, text):
        if not CN2AN_AVAILABLE or not text:
            return text
        
        # Filter out invalid inputs that cause cn2an warnings
        # Single unit characters like 百, 千, 万 are not valid standalone numbers
        if text in ['百', '千', '万', '亿', '十']:
            return text
            
        # Check if text contains only Chinese number characters
        chinese_digits = '一二三四五六七八九十百千万亿零'
        if not all(c in chinese_digits for c in text):
            return text
            
        try:
            result = cn2an.cn2an(text, "normal")
            return str(result)
        except Exception as e:
            # Suppress warnings for invalid formats
            if self.config.enable_debug_logging:
                logger.debug(f"cn2an conversion failed for '{text}': {e}")
            return text
    
    def _convert_single(self, match):
        return self.digit_map.get(match.group(1), match.group(1))
    
    def _convert_teen(self, match):
        digit = match.group(1)
        return f"1{self.digit_map.get(digit, digit)}"
    
    def _convert_ten(self, match):
        ten_digit = match.group(1)
        unit_digit = match.group(2)
        return f"{self.digit_map.get(ten_digit, ten_digit)}{self.digit_map.get(unit_digit, unit_digit)}"
    
    def _convert_decades(self, match):
        ten_digit = match.group(1)
        return f"{self.digit_map.get(ten_digit, ten_digit)}0"
    
    def _convert_complex_number(self, match):
        text = match.group(0)

        if CN2AN_AVAILABLE and '两千' in text:
            try:
                # Use the safe wrapper function
                result = self._try_cn2an(text)
                if result != text:
                    return result
            except:
                pass

        if CN2AN_AVAILABLE:
            try:
                # Use the safe wrapper function
                result = self._try_cn2an(text)
                if result != text:
                    return result
            except:
                pass

        # Fallback
        m = re.match(r'([一二三四五六七八九十]+)(万|千|百)(.+)', text)
        if m:
            high, unit_char, low = m.groups()
            unit_map = {"万":10000, "千":1000, "百":100}
            try:
                high_val = self._try_cn2an(high) if CN2AN_AVAILABLE else int(self.digit_map[high])
                low_val = self._try_cn2an(low) if CN2AN_AVAILABLE else int(self._try_cn2an(low))
                return str(int(high_val) * unit_map[unit_char] + int(low_val))
            except:
                return text

        return text
    
    def _convert_decimal(self, match):
        integer_part = match.group(1)
        decimal_part = match.group(2)
        
        integer_val = self._try_cn2an(integer_part)
        if not integer_val.isdigit():
            return match.group(0)
        
        decimal_digits = ''.join(self.digit_map.get(d, d) for d in decimal_part)
        
        return f"{integer_val}.{decimal_digits}"
    
    def _convert_fraction(self, match):
        denominator = match.group(1)
        numerator = match.group(2)
        
        denom_val = self._try_cn2an(denominator)
        numer_val = self._try_cn2an(numerator)
        
        if denom_val.isdigit() and numer_val.isdigit():
            return f"{numer_val}/{denom_val}"
        
        return match.group(0)
    
    def convert_numbers(self, text: str) -> str:
        if not text:
            return ""

        original = text
        try:
            for pattern, converter in self.compiled_patterns:
                text = pattern.sub(converter, text)

            if CN2AN_AVAILABLE:
                try:
                    cn2an_result = cn2an.transform(text, "cn2an")
                    if cn2an_result and cn2an_result != text:
                        if self.config.enable_debug_logging:
                            logger.debug(f"cn2an fallback: '{text}' → '{cn2an_result}'")
                        text = cn2an_result
                except Exception as e:
                    if self.config.enable_debug_logging:
                        logger.debug(f"cn2an fallback error: {e}")

            if self.config.enable_debug_logging and original != text:
                logger.debug(f"Number conversion: '{original}' → '{text}'")

            return text

        except Exception as e:
            logger.warning(f"Number conversion failed: {e}")
            return original


class UnitProcessor:
    
    def __init__(self, config: ASRConfig):
        self.config = config
        self._setup_unit_mappings()
        self._compile_patterns()
    
    def _setup_unit_mappings(self):
        self.unit_mappings = {
            # Weight
            'kg': '公斤', 'g': '克', 't': '吨', 'mg': '毫克', 'lb': '磅', 'oz': '盎司',
            
            # Distance/Length
            'km': '公里', 'm': '米', 'cm': '厘米', 'mm': '毫米', 
            'ft': '英尺', 'inch': '英寸', 'in': '英寸', 'mile': '英里', 'yard': '码',
            
            # Volume
            'l': '升', 'ml': '毫升', 'L': '升', 'gal': '加仑', 'qt': '夸脱',
            
            # Temperature
            '°C': '摄氏度', '℃': '摄氏度', '°F': '华氏度', '℉': '华氏度',
            
            # Time
            'ms': '毫秒', 's': '秒', 'sec': '秒', 'min': '分钟', 'h': '小时', 'hr': '小时',
            
            # Technology - Storage units
            'GB': 'gb', 'MB': 'mb', 'KB': 'kb', 'TB': 'tb', 'PB': 'pb',
            'gb': 'gb', 'mb': 'mb', 'kb': 'kb', 'tb': 'tb', 'pb': 'pb',
            'Gb': 'gb', 'Mb': 'mb', 'Kb': 'kb', 'Tb': 'tb', 'Pb': 'pb',
            
            # Frequency
            'Hz': '赫兹', 'hz': '赫兹', 'HZ': '赫兹',
            'MHz': '兆赫', 'mhz': '兆赫', 'Mhz': '兆赫', 'MHZ': '兆赫',
            'GHz': '吉赫', 'ghz': '吉赫', 'Ghz': '吉赫', 'GHZ': '吉赫',
            'KHz': '千赫', 'khz': '千赫', 'kHz': '千赫', 'Khz': '千赫', 'KHZ': '千赫',
            
            # Electrical
            'V': '伏特', 'v': '伏特', 'volt': '伏特', 'volts': '伏特',
            'A': '安培', 'a': '安培', 'amp': '安培', 'amps': '安培',
            'W': '瓦特', 'w': '瓦特', 'watt': '瓦特', 'watts': '瓦特',
            'kW': '千瓦', 'kw': '千瓦', 'KW': '千瓦',
            'mW': '毫瓦', 'mw': '毫瓦', 'MW': '毫瓦',
            
            # Network speeds
            'Mbps': 'mbps', 'mbps': 'mbps', 'MBPS': 'mbps',
            'Gbps': 'gbps', 'gbps': 'gbps', 'GBPS': 'gbps', 
            'Kbps': 'kbps', 'kbps': 'kbps', 'KBPS': 'kbps',
            'MB/s': 'mb每秒', 'mb/s': 'mb每秒', 'GB/s': 'gb每秒', 'gb/s': 'gb每秒',
            
            # Composite units
            'km/h': '公里每小时', 'mph': '英里每小时', 'm/s': '米每秒',
        }
    
    def _compile_patterns(self):
        
        self.pattern_sets = []
        
        # Pattern Set 1: Composite units and special characters
        composite_patterns = [
            ('km/h', '公里每小时'), ('mph', '英里每小时'), ('m/s', '米每秒'),
            ('MB/s', 'mb每秒'), ('mb/s', 'mb每秒'), ('GB/s', 'gb每秒'), ('gb/s', 'gb每秒'),
            ('°C', '摄氏度'), ('℃', '摄氏度'), ('°F', '华氏度'), ('℉', '华氏度'),
        ]
        
        for unit_symbol, unit_target in composite_patterns:
            pattern = rf'(\d+(?:\.\d+)?)\s*{re.escape(unit_symbol)}'
            self.pattern_sets.append(('composite', re.compile(pattern, 0), unit_target, unit_symbol))
        
        # Pattern Set 2: Frequency units
        frequency_patterns = [
            ('GHz', '吉赫'), ('ghz', '吉赫'), ('Ghz', '吉赫'), ('GHZ', '吉赫'),
            ('MHz', '兆赫'), ('mhz', '兆赫'), ('Mhz', '兆赫'), ('MHZ', '兆赫'),
            ('KHz', '千赫'), ('khz', '千赫'), ('kHz', '千赫'), ('Khz', '千赫'), ('KHZ', '千赫'),
            ('Hz', '赫兹'), ('hz', '赫兹'), ('HZ', '赫兹'),
        ]
        
        for unit_symbol, unit_target in frequency_patterns:
            pattern = rf'(\d+(?:\.\d+)?)\s*{re.escape(unit_symbol)}(?![a-zA-Z])'
            self.pattern_sets.append(('frequency', re.compile(pattern, re.IGNORECASE), unit_target, unit_symbol))
        
        # Pattern Set 3: Network speeds
        network_patterns = [
            ('Mbps', 'mbps'), ('mbps', 'mbps'), ('MBPS', 'mbps'),
            ('Gbps', 'gbps'), ('gbps', 'gbps'), ('GBPS', 'gbps'), 
            ('Kbps', 'kbps'), ('kbps', 'kbps'), ('KBPS', 'kbps'),
        ]
        
        for unit_symbol, unit_target in network_patterns:
            pattern = rf'(\d+(?:\.\d+)?)\s*{re.escape(unit_symbol)}\b'
            self.pattern_sets.append(('network', re.compile(pattern, re.IGNORECASE), unit_target, unit_symbol))
        
        # Pattern Set 4: Storage units
        storage_patterns = [
            ('GB', 'gb'), ('gb', 'gb'), ('Gb', 'gb'),
            ('MB', 'mb'), ('mb', 'mb'), ('Mb', 'mb'),
            ('TB', 'tb'), ('tb', 'tb'), ('Tb', 'tb'),
            ('KB', 'kb'), ('kb', 'kb'), ('Kb', 'kb'),
            ('PB', 'pb'), ('pb', 'pb'), ('Pb', 'pb'),
        ]
        
        for unit_symbol, unit_target in storage_patterns:
            pattern = rf'(\d+(?:\.\d+)?)\s*{re.escape(unit_symbol)}\b'
            self.pattern_sets.append(('storage', re.compile(pattern, 0), unit_target, unit_symbol))
        
        # Pattern Set 5: Electrical units with power modifiers
        electrical_modified_patterns = [
            ('kW', '千瓦'), ('kw', '千瓦'), ('KW', '千瓦'),
            ('mW', '毫瓦'), ('mw', '毫瓦'), ('MW', '毫瓦'),
        ]
        
        for unit_symbol, unit_target in electrical_modified_patterns:
            pattern = rf'(\d+(?:\.\d+)?)\s*{re.escape(unit_symbol)}\b'
            self.pattern_sets.append(('electrical_mod', re.compile(pattern, re.IGNORECASE), unit_target, unit_symbol))
        
        # Pattern Set 6: Single letter electrical units
        electrical_single_patterns = [
            ('W', '瓦特'), ('w', '瓦特'),
            ('V', '伏特'), ('v', '伏特'),
            ('A', '安培'), ('a', '安培'),
        ]
        
        for unit_symbol, unit_target in electrical_single_patterns:
            pattern = rf'(\d+(?:\.\d+)?)\s*{re.escape(unit_symbol)}(?![a-zA-Z])'
            self.pattern_sets.append(('electrical_single', re.compile(pattern, re.IGNORECASE), unit_target, unit_symbol))
        
        # Pattern Set 7: Other units
        other_patterns = [
            ('kg', '公斤'), ('g', '克'), ('mg', '毫克'), ('t', '吨'),
            ('km', '公里'), ('cm', '厘米'), ('mm', '毫米'), ('m', '米'),
            ('ms', '毫秒'), ('s', '秒'), ('h', '小时'),
            ('L', '升'), ('l', '升'), ('ml', '毫升'),
            ('lb', '磅'), ('oz', '盎司'), ('ft', '英尺'), ('in', '英寸'), ('inch', '英寸'),
            ('mile', '英里'), ('yard', '码'), ('gal', '加仑'), ('qt', '夸脱'),
            ('sec', '秒'), ('min', '分钟'), ('hr', '小时'),
            ('volt', '伏特'), ('volts', '伏特'), ('amp', '安培'), ('amps', '安培'),
            ('watt', '瓦特'), ('watts', '瓦特'),
        ]
        
        for unit_symbol, unit_target in other_patterns:
            if len(unit_symbol) == 1:
                pattern = rf'(\d+(?:\.\d+)?)\s*{re.escape(unit_symbol)}(?![a-zA-Z])'
            else:
                pattern = rf'(\d+(?:\.\d+)?)\s*{re.escape(unit_symbol)}\b'
            self.pattern_sets.append(('other', re.compile(pattern, re.IGNORECASE), unit_target, unit_symbol))
    
    def convert_units(self, text: str) -> str:
        if not self.config.normalize_units or not text:
            return text or ""
        
        try:
            original = text
            
            for pattern_type, compiled_pattern, unit_target, original_unit in self.pattern_sets:
                def replace_unit(match):
                    number = match.group(1)
                    return f"{number}{unit_target}"
                
                new_text = compiled_pattern.sub(replace_unit, text)
                if new_text != text:
                    if self.config.enable_debug_logging:
                        logger.debug(f"Unit {pattern_type} '{original_unit}': '{text}' → '{new_text}'")
                    text = new_text
            
            if self.config.enable_debug_logging and original != text:
                logger.debug(f"Final unit conversion: '{original}' → '{text}'")
            
            return text
            
        except Exception as e:
            logger.warning(f"Unit conversion failed: {e}")
            return text

class CurrencyProcessor:
    
    def __init__(self, config: ASRConfig):
        self.config = config
        self._compile_patterns()
    
    def _compile_patterns(self):
        self.patterns = [
            # Chinese Yuan
            (re.compile(r'¥(\d+(?:\.\d+)?)元'), r'\1元'),
            (re.compile(r'¥(\d+(?:\.\d+)?)(?!元)'), r'\1元'),
            (re.compile(r'人民币(\d+(?:\.\d+)?)元?'), r'\1元'),
            (re.compile(r'(\d+(?:\.\d+)?)rmb'), r'\1元', re.IGNORECASE),
            
            # Other currencies
            (re.compile(r'\$(\d+(?:\.\d+)?)'), r'\1美元'),
            (re.compile(r'€(\d+(?:\.\d+)?)'), r'\1欧元'),
            (re.compile(r'£(\d+(?:\.\d+)?)'), r'\1英镑'),
            (re.compile(r'￡(\d+(?:\.\d+)?)'), r'\1英镑'),
            
            # Fix double units
            (re.compile(r'(\d+(?:\.\d+)?)元元'), r'\1元'),
        ]
    
    def normalize_currency(self, text: str) -> str:
        if not self.config.normalize_currency or not text:
            return text or ""
        
        try:
            original = text
            
            for pattern_info in self.patterns:
                if len(pattern_info) == 3:
                    pattern, replacement, flags = pattern_info
                    text = pattern.sub(replacement, text)
                else:
                    pattern, replacement = pattern_info
                    text = pattern.sub(replacement, text)
            
            if self.config.enable_debug_logging and original != text:
                logger.debug(f"Currency normalization: '{original}' → '{text}'")
            
            return text
            
        except Exception as e:
            logger.warning(f"Currency normalization failed: {e}")
            return text


class PhoneProcessor:
    
    def __init__(self, config: ASRConfig):
        self.config = config
        self._setup_digit_mapping()
    
    def _setup_digit_mapping(self):
        if self.config.use_telephony_format:
            self.digit_map = {
                '0': '零', '1': '幺', '2': '二', '3': '三', '4': '四',
                '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
            }
        else:
            self.digit_map = {
                '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
                '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
            }
    
    def _digits_to_chinese(self, digit_string: str) -> str:
        if not digit_string:
            return ""
        
        if not isinstance(digit_string, str):
            digit_string = str(digit_string)
        
        result = ""
        converted_count = 0
        
        for char in digit_string:
            if char.isdigit():
                chinese_digit = self.digit_map.get(char, char)
                result += chinese_digit
                if chinese_digit != char:
                    converted_count += 1
            else:
                result += char
        
        digit_count = sum(1 for c in digit_string if c.isdigit())
        if digit_count > 0 and converted_count == 0:
            logger.warning(f"No digits converted in '{digit_string}' - check digit_map")
        
        if self.config.enable_debug_logging:
            logger.debug(f"Digits to Chinese: '{digit_string}' → '{result}' (converted {converted_count}/{digit_count} digits)")
        
        return result
    

    def normalize_phones(self, text: str) -> str:
        if not self.config.standardize_phone_format or not text:
            return text or ""

        original = text
        
        if self.config.enable_debug_logging:
            logger.debug(f"Phone normalization input: '{text}'")
        
        # +86 international format
        def replace_international_spaced(match):
            digits = '86' + match.group(1) + match.group(2) + match.group(3)
            converted = self._digits_to_chinese(digits)
            return converted
        
        def replace_international_continuous(match):
            digits = '86' + match.group(1)
            converted = self._digits_to_chinese(digits)
            return converted
        
        text = re.sub(r'\+86[\s-]*(\d{3})[\s-]+(\d{4})[\s-]+(\d{4})', 
                    replace_international_spaced, text)
        text = re.sub(r'\+86[\s-]*(\d{11})', 
                    replace_international_continuous, text)
        
        # Context + continuous 11 digits
        def replace_context_continuous(match):
            context = match.group(1)
            digits = match.group(2) 
            if self.config.enable_debug_logging:
                logger.debug(f"Context continuous match: context='{context}', digits='{digits}'")

            if len(digits) == 11 and digits[0] == '1' and digits[1] in '3456789':
                converted = self._digits_to_chinese(digits)
                result = context + converted
                if self.config.enable_debug_logging:
                    logger.debug(f"Context continuous result: '{result}'")
                return result
            return match.group(0)
        
        text = re.sub(r'(手机号码?|电话号码?|联系电话|手机|电话)(\d{11})', 
                    replace_context_continuous, text)
        
        # Context + 3-4-4 spaced format
        def replace_context_spaced_no_space(match):
            context = match.group(1)
            digit1, digit2, digit3 = match.group(2), match.group(3), match.group(4)
            all_digits = digit1 + digit2 + digit3
            if self.config.enable_debug_logging:
                logger.debug(f"Context spaced (no space) match: context='{context}', digits='{all_digits}'")
            converted = self._digits_to_chinese(all_digits)
            result = context + converted
            if self.config.enable_debug_logging:
                logger.debug(f"Context spaced (no space) result: '{result}'")
            return result
        
        text = re.sub(r'(手机号码?|电话号码?|联系电话|手机|电话)(\d{3})[\s]+(\d{4})[\s]+(\d{4})', 
                    replace_context_spaced_no_space, text)
        
        # Context + 3-4-4 spaced format
        def replace_context_spaced_with_space(match):
            context = match.group(1)
            digit1, digit2, digit3 = match.group(2), match.group(3), match.group(4)
            all_digits = digit1 + digit2 + digit3
            if self.config.enable_debug_logging:
                logger.debug(f"Context spaced (with space) match: context='{context}', digits='{all_digits}'")
            converted = self._digits_to_chinese(all_digits)
            result = context + converted
            if self.config.enable_debug_logging:
                logger.debug(f"Context spaced (with space) result: '{result}'")
            return result
        
        text = re.sub(r'(手机号码?|电话号码?|联系电话|手机|电话)[\s：:]+(\d{3})[\s]+(\d{4})[\s]+(\d{4})', 
                    replace_context_spaced_with_space, text)
        
        # Context + 3-4-4 dashed format
        text = re.sub(r'(手机号码?|电话号码?|联系电话|手机|电话)：(\d{3})[-]+(\d{4})[-]+(\d{4})', 
                    lambda m: m.group(1) + self._digits_to_chinese(m.group(2) + m.group(3) + m.group(4)), text)
        
        # Context + 3-4-4 dashed format
        text = re.sub(r'(手机号码?|电话号码?|联系电话|手机|电话)(\d{3})[-]+(\d{4})[-]+(\d{4})', 
                    lambda m: m.group(1) + self._digits_to_chinese(m.group(2) + m.group(3) + m.group(4)), text)
        
        # Handling "是" pattern specifically
        text = re.sub(r'(手机号码?|电话号码?)是[\s]*(\d{3})[\s]+(\d{4})[\s]+(\d{4})', 
                    lambda m: m.group(1) + '是' + self._digits_to_chinese(m.group(2) + m.group(3) + m.group(4)), text)
        
        # Service numbers
        def replace_service_400_800(match):
            digits = match.group(1) + match.group(2) + match.group(3)
            converted = self._digits_to_chinese(digits)
            if self.config.enable_debug_logging:
                logger.debug(f"Service 400-800 conversion: '{match.group(0)}' → '{converted}'")
            return converted
        
        text = re.sub(r'\b(400)[-\s]*(800)[-\s]*(\d{4})\b', 
                    replace_service_400_800, text)
        
        # General 400 service numbers
        def replace_service_400_general(match):
            digits = match.group(1) + match.group(2) + match.group(3)
            converted = self._digits_to_chinese(digits)
            if self.config.enable_debug_logging:
                logger.debug(f"Service 400 general conversion: '{match.group(0)}' → '{converted}'")
            return converted
        
        text = re.sub(r'\b(400)[-\s]*(\d{3})[-\s]*(\d{4})\b', 
                    replace_service_400_general, text)
        
        # Other service numbers (4xx, 8xx, 9xx)
        def replace_service_general(match):
            digits = match.group(1) + match.group(2) + match.group(3)
            converted = self._digits_to_chinese(digits)
            if self.config.enable_debug_logging:
                logger.debug(f"Service general conversion: '{match.group(0)}' → '{converted}'")
            return converted
        
        text = re.sub(r'\b([489]\d{2})[-\s]*(\d{3})[-\s]*(\d{4})\b', 
                    replace_service_general, text)
        
        # 9. Handle standalone mobile numbers
        def replace_standalone(match):
            if len(match.groups()) == 3:  # 3-4-4 format
                digits = match.group(1) + match.group(2) + match.group(3)
            else:  # 11-digit format
                digits = match.group(1)
            
            # Validate mobile number
            if len(digits) == 11 and digits[0] == '1' and digits[1] in '3456789':
                converted = self._digits_to_chinese(digits)
                if self.config.enable_debug_logging:
                    logger.debug(f"Standalone mobile conversion: '{digits}' → '{converted}'")
                return converted
            return match.group(0)
        
        text = re.sub(r'\b(1[3-9]\d)[\s-]+(\d{4})[\s-]+(\d{4})\b', replace_standalone, text)
        text = re.sub(r'\b(1[3-9]\d{9})\b', replace_standalone, text)
        
        if self.config.enable_debug_logging and original != text:
            logger.debug(f"Phone normalization result: '{original}' → '{text}'")
        
        return text

class DateTimeProcessor:
    
    def __init__(self, config: ASRConfig):
        self.config = config
        self._compile_patterns()
    
    def _compile_patterns(self):
        self.patterns = [
            (re.compile(r'([一二三四五六七八九十\d]+)点半(?![0-9])'), self._convert_half_hour),
            (re.compile(r'([一二三四五六七八九十\d]+)点([一二三]刻)'), self._convert_quarter),
            (re.compile(r'([一二三四五六七八九十\d]+)点([一二三四五六七八九十\d]+)分'), self._convert_full_time),
            (re.compile(r'(\d{1,2}):(\d{2})'), self._convert_digital_time),
            
            (re.compile(r'(\d{4})年(\d{1,2})月(\d{1,2})日'), r'\1年\2月\3日'),
            (re.compile(r'(\d{1,2})月(\d{1,2})[号日]'), r'\1月\2日'),
        ]
    
    def _convert_time_number(self, num_str: str) -> str:
        if num_str.isdigit():
            return num_str
        
        if CN2AN_AVAILABLE:
            try:
                result = self._try_cn2an(num_str)
                if result != num_str:
                    return result
            except:
                pass
        
        time_map = {
            '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
            '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
            '十一': '11', '十二': '12', '十三': '13', '十四': '14',
            '十五': '15', '十六': '16', '十七': '17', '十八': '18',
            '十九': '19', '二十': '20', '二十一': '21', '二十二': '22',
            '二十三': '23', '二十四': '24'
        }
        return time_map.get(num_str, num_str)
    
    def _convert_half_hour(self, match):
        hour = match.group(1)
        hour_arabic = self._convert_time_number(hour)
        return f"{hour_arabic}点30分"
    
    def _convert_quarter(self, match):
        hour = match.group(1)
        quarter = match.group(2)
        
        quarter_map = {'一刻': '15', '二刻': '30', '三刻': '45'}
        minute = quarter_map.get(quarter, quarter)
        
        hour_arabic = self._convert_time_number(hour)
        return f"{hour_arabic}点{minute}分"
    
    def _convert_full_time(self, match):
        hour = match.group(1)
        minute = match.group(2)
        
        hour_arabic = self._convert_time_number(hour)
        minute_arabic = self._convert_time_number(minute)
        
        return f"{hour_arabic}点{minute_arabic}分"
    
    def _convert_digital_time(self, match):
        hour, minute = match.groups()
        return f"{hour}点{minute}分"
    
    def normalize_datetime(self, text: str) -> str:
        if not self.config.normalize_datetime or not text:
            return text or ""
        
        try:
            original = text
            
            for pattern, converter in self.patterns:
                if callable(converter):
                    text = pattern.sub(converter, text)
                else:
                    text = pattern.sub(converter, text)
            
            if self.config.enable_debug_logging and original != text:
                logger.debug(f"DateTime normalization: '{original}' → '{text}'")
            
            return text
            
        except Exception as e:
            logger.warning(f"DateTime normalization failed: {e}")
            return text


class FillerProcessor:
    
    def __init__(self, config: ASRConfig):
        self.config = config
        self._setup_filler_patterns()
    
    def _setup_filler_patterns(self):
        self.multi_word_fillers = []
        
        # Annoying patterns
        self.extra_aggressive_patterns = []
        
        # Single-word primary fillers
        self.primary_fillers = []
        
        # Hesitation sounds and interjections
        self.hesitation_sounds = [
            '呃', '嗯', '额', '啊', '呀', '哦', '咧', '啦', '呗', '嘛',
            '唔', '嘨', '诶', '哎', '嘿', '唉'
        ]
        
        # Discourse markers
        self.discourse_markers = []
    
    def remove_fillers(self, text: str) -> str:
        if not self.config.remove_filler_words or not text:
            return text or ""

        if not isinstance(text, str):
            text = str(text)

        original = text
        
        for iteration in range(8):
            old_text = text
            
            # PASS 0: Handle extra aggressive patterns first
            text = self._remove_extra_aggressive_patterns(text)
            
            # PASS 1: Handle explicit repeated patterns
            text = self._remove_repeated_patterns(text)
            
            # PASS 2: Remove multi-word fillers
            text = self._remove_multi_word_fillers(text)
            
            # PASS 3: Remove primary single-word fillers
            text = self._remove_primary_fillers(text)
            
            # PASS 4: Remove hesitation sounds
            text = self._remove_hesitation_sounds(text)
            
            # PASS 5: Remove discourse markers
            text = self._remove_discourse_markers(text)
            
            # PASS 6: Clean up artifacts
            text = self._clean_artifacts(text)
            
            # If no changes, we're done :D
            if text == old_text:
                break
        
        text = self._final_aggressive_cleanup(text)
        
        text = text.replace('___PROTECTED_ZHEGE___', '这个')
        
        if self.config.enable_debug_logging and original != text:
            logger.debug(f"Filler removal ({iteration+1} iterations): '{original}' → '{text}'")
        
        return text
    
    def _remove_extra_aggressive_patterns(self, text: str) -> str:
        for pattern in self.extra_aggressive_patterns:
            if pattern in text:
                text = text.replace(pattern, ' ')

                text = text.replace(f'{pattern}，', ' ')
                text = text.replace(f'{pattern}。', ' ')
                text = text.replace(f' {pattern} ', ' ')
                text = text.replace(f' {pattern}', ' ')
                text = text.replace(f'{pattern} ', ' ')
                
                try:
                    pattern_escaped = re.escape(pattern)
                    text = re.sub(pattern_escaped, ' ', text)
                except:
                    pass
        
        # SPECIAL CASE: Handle "额这个" sequence (remove only "额", keep "这个")
        # Use a persistent placeholder to protect "这个" from later removal
        if '额这个' in text:
            text = text.replace('额这个', '___PROTECTED_ZHEGE___')
        
        # SPECIAL CASE: Handle "呃这个" sequence (remove only "呃", keep "这个")
        if '呃这个' in text:
            text = text.replace('呃这个', '___PROTECTED_ZHEGE___')
            
        # SPECIAL CASE: Handle "嗯这个" sequence (remove only "嗯", keep "这个")
        if '嗯这个' in text:
            text = text.replace('嗯这个', '___PROTECTED_ZHEGE___')
        
        return text
    
    def _remove_repeated_patterns(self, text: str) -> str:
        for pattern, replacement in self.repeated_patterns:
            text = text.replace(pattern, replacement)
        return text
    
    def _remove_multi_word_fillers(self, text: str) -> str:
        for filler in self.multi_word_fillers:
            text = self._aggressive_remove_single_filler(text, filler)
        return text
    
    def _remove_primary_fillers(self, text: str) -> str:
        for filler in self.primary_fillers:
            text = self._aggressive_remove_single_filler(text, filler)
        return text
    
    def _remove_hesitation_sounds(self, text: str) -> str:
        for filler in self.hesitation_sounds:
            text = self._aggressive_remove_single_filler(text, filler)
        return text
    
    def _remove_discourse_markers(self, text: str) -> str:
        for filler in self.discourse_markers:
            text = self._aggressive_remove_single_filler(text, filler)
        return text
    
    def _aggressive_remove_single_filler(self, text: str, filler: str) -> str:
        if not filler or filler not in text:
            return text
        
        # PROTECTION: Don't remove filler if it's part of a compound word
        if filler == '额':
            # Protect compound words containing 额
            protected_words = ['营业额', '金额', '数额', '总额', '余额', '面额', '定额']
            for word in protected_words:
                if word in text:
                    return text
        
        # Method 1: Simple replacement for common cases
        text = text.replace(f' {filler} ', ' ')
        text = text.replace(f'{filler} ', ' ')
        text = text.replace(f' {filler}', ' ')
        
        # Method 2: At start/end of text
        if text.startswith(filler):
            text = text[len(filler):].lstrip()
        if text.endswith(filler):
            text = text[:-len(filler)].rstrip()
        
        # Method 3: With punctuation
        punctuation_patterns = [
            (f' {filler}，', ' '), (f' {filler}。', ' '), (f' {filler}！', ' '), (f' {filler}？', ' '),
            (f'，{filler} ', ' '), (f'。{filler} ', ' '), (f'！{filler} ', ' '), (f'？{filler} ', ' '),
            (f'，{filler}，', '，'), (f'。{filler}。', '。'),
            (f'，{filler}', ' '), (f'。{filler}', ' '), (f'！{filler}', ' '), (f'？{filler}', ' '),
            (f'{filler}，', ' '), (f'{filler}。', ' '), (f'{filler}！', ' '), (f'{filler}？', ' '),
        ]
        
        for pattern, replacement in punctuation_patterns:
            text = text.replace(pattern, replacement)
        
        # Method 4: Regex for CJK boundaries
        try:
            cjk_pattern = f'([\u4e00-\u9fff]){re.escape(filler)}([\u4e00-\u9fff])'
            text = re.sub(cjk_pattern, r'\1\2', text)
            
            word_boundary_pattern = f'\\b{re.escape(filler)}\\b'
            text = re.sub(word_boundary_pattern, ' ', text)
            
            standalone_pattern = f'(?<![\\u4e00-\\u9fff]){re.escape(filler)}(?![\\u4e00-\\u9fff])'
            text = re.sub(standalone_pattern, ' ', text)
            
        except Exception:
            text = text.replace(filler, ' ')
        
        return text
    
    def _clean_artifacts(self, text: str) -> str:
        text = re.sub(r'，+', '，', text)
        text = re.sub(r'。+', '。', text)
        text = re.sub(r'！+', '！', text)
        text = re.sub(r'？+', '？', text)
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*([，。！？；：])', r'\1', text)
        text = re.sub(r'([，。！？；：])\s+', r'\1 ', text)
        
        # Remove punctuation at start
        text = re.sub(r'^[，。！？；：\s]+', '', text)
        
        return text.strip()
    
    def _final_aggressive_cleanup(self, text: str) -> str:
        critical_check_list = ['那个', '就是', '然后', '那么', '呃', '嗯', '啊', '总而言之']
        
        for filler in critical_check_list:
            if filler in text:
                text = text.replace(filler, ' ')

        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = text.strip('，。！？；：')
        text = text.strip()
        
        return text

class ChineseASRNormalizer:
    
    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        
        # Initialize processors
        self.filler_processor = FillerProcessor(self.config)
        self.currency_processor = CurrencyProcessor(self.config)
        self.unit_processor = UnitProcessor(self.config)
        self.phone_processor = PhoneProcessor(self.config)
        self.number_converter = ChineseNumberConverter(self.config)
        self.datetime_processor = DateTimeProcessor(self.config)
        
        if self.config.enable_debug_logging:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.debug("Chinese ASR normalizer initialized (Fixed Arabic digits mode)")
    
    def _final_cleanup(self, text: str) -> str:
        try:
            if self.config.remove_punctuation:
                if self.config.handle_chinese_punctuation:
                    chinese_punct = '，。！？；：""\'\'（）【】《》〈〉「」『』〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‛„‟…‧﹏'
                    for punct in chinese_punct:
                        text = text.replace(punct, ' ')

                preserved_chars = '.'
                if self.config.preserve_percentages:
                    preserved_chars += '%'
                if '/' in text:
                    preserved_chars += '/'
                if '-' in text:
                    preserved_chars += '-'
                
                punct_to_remove = string.punctuation
                for char in preserved_chars:
                    punct_to_remove = punct_to_remove.replace(char, '')
                text = text.translate(str.maketrans('', '', punct_to_remove))

            if self.config.normalize_whitespace:
                text = re.sub(r'\s+', ' ', text).strip()
                text = re.sub(r'\s+([\u4e00-\u9fff])', r'\1', text)

            if self.config.lowercase_output:
                text = text.lower()

            text = re.sub(r'\s+', ' ', text).strip()
            return text

        except Exception as e:
            logger.warning(f"Final cleanup failed: {e}")
            # Fallback: force lowercase
            try:
                return text.lower().strip()
            except:
                return str(text).lower().strip() if text else ""

    def _validate_output(self, original: str, normalized: str) -> bool:
        if not self.config.enable_validation:
            return True
        
        try:
            issues = []
            
            # More specific Chinese number patterns that should be converted
            chinese_numbers = ['三千', '万', '十万', '二十', '三十', '四十', '五十', '一千', '两千']
            has_chinese_nums = any(cn in normalized for cn in chinese_numbers)
            
            # Contexts where Chinese numbers might be legitimate (not requiring conversion)
            is_phone_context = any(kw in normalized for kw in ['手机', '电话', '号码'])
            is_time_context = any(kw in normalized for kw in ['点', '分', '时'])
            is_address_context = any(kw in normalized for kw in ['路', '街', '号', '区', '市', '省'])
            is_name_context = any(kw in normalized for kw in ['先生', '女士', '老师', '医生', '经理'])
            is_measurement_context = any(kw in normalized for kw in ['米', '公里', '斤', '公斤', '元', '块'])
            
            # Only flag if Chinese numbers are present and none of the legitimate contexts apply
            if (has_chinese_nums and not is_phone_context and not is_time_context and 
                not is_address_context and not is_name_context and not is_measurement_context):
                # Additional check: only flag if the Chinese numbers appear to be standalone numbers
                # (not part of compound words or proper nouns)
                standalone_number_pattern = r'\b(三千|万|十万|二十|三十|四十|五十|一千|两千)\b'
                if re.search(standalone_number_pattern, normalized):
                    issues.append("Chinese numbers not converted to Arabic")
            
            if any(symbol in normalized for symbol in ['¥', '$', '€', '￡']):
                issues.append("Currency symbols not removed")
            
            if len(normalized) > len(original) * 3:
                issues.append("Excessive length increase")
            
            if self.config.remove_filler_words:
                critical_fillers = ['那个', '这个', '呃', '嗯']
                remaining_fillers = []
                
                for filler in critical_fillers:
                    if (re.search(rf'^{re.escape(filler)}\s', normalized) or
                        re.search(rf'\s{re.escape(filler)}\s', normalized) or
                        re.search(rf'\s{re.escape(filler)}$', normalized)):
                        remaining_fillers.append(filler)
                
                if remaining_fillers:
                    issues.append(f"Filler words not removed: {remaining_fillers}")
            
            if issues:
                logger.warning(f"Validation issues: {issues}")
                if self.config.strict_mode:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return not self.config.strict_mode
    
    def normalize(self, text: str) -> str:
        """FIXED: Main normalization with phone number protection"""
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
            logger.debug(f"=== FIXED ASR NORMALIZATION START ===")
            logger.debug(f"Input: '{text}'")
        
        try:
            original = text
            
            # Filler removal
            text = self.filler_processor.remove_fillers(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After filler removal: '{text}'")
            
            # Currency normalization
            text = self.currency_processor.normalize_currency(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After currency: '{text}'")
            
            # Unit conversion
            text = self.unit_processor.convert_units(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After units: '{text}'")
            
            text = self._protect_phone_numbers(text)
            text = self.phone_processor.normalize_phones(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After phone: '{text}'")
            
            # Date/time normalization
            text = self.datetime_processor.normalize_datetime(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After datetime: '{text}'")
            
            # Number conversion (after phone/datetime but avoid protected phone areas)
            text = self._number_conversion_with_phone_protection(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After number conversion: '{text}'")
            
            # Remove protection markers
            text = self._remove_phone_protection(text)
            
            # Final cleanup (punctuation, case, whitespace)
            text = self._final_cleanup(text)
            if self.config.enable_debug_logging:
                logger.debug(f"After cleanup: '{text}'")
            
            # Validation
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


    def _protect_phone_numbers(self, text: str) -> str:
        if self.config.enable_debug_logging:
            logger.debug(f"Phone protection input: '{text}'")
        
        protection_patterns = [
            # Context + continuous 11 digits
            r'(手机号码?|电话号码?|联系电话|手机|电话)(\d{11})\b',
            
            # Context + spaced formats
            r'(手机号码?|电话号码?|联系电话|手机|电话)(\d{3})[\s]+(\d{4})[\s]+(\d{4})',
            
            # Context + spaced formats
            r'(手机号码?|电话号码?|联系电话|手机|电话)[\s：:]+(\d{3})[\s]+(\d{4})[\s]+(\d{4})',
            r'(手机号码?|电话号码?)是[\s]*(\d{3})[\s]+(\d{4})[\s]+(\d{4})',
            
            # Context + dashed formats
            r'(手机号码?|电话号码?|联系电话|手机|电话)：(\d{3})[-]+(\d{4})[-]+(\d{4})',
            r'(手机号码?|电话号码?|联系电话|手机|电话)(\d{3})[-]+(\d{4})[-]+(\d{4})',
            
            # International format
            r'\+86[\s-]*(\d{3})[\s-]+(\d{4})[\s-]+(\d{4})',
            r'\+86[\s-]*(\d{11})',
            
            # Service numbers
            r'\b(400)[-\s]*(800)[-\s]*(\d{4})\b',
            r'\b([489]\d{2})[-\s]*(\d{3})[-\s]*(\d{4})\b',
            r'\b(400)[-\s]*(\d{3})[-\s]*(\d{4})\b',
            
            # Standalone mobile
            r'\b(1[3-9]\d)[\s-]+(\d{4})[\s-]+(\d{4})\b',
            r'\b(1[3-9]\d{9})\b',
        ]
        
        # Apply protection markers
        protected_count = 0
        for i, pattern_str in enumerate(protection_patterns):
            pattern = re.compile(pattern_str)
            matches = list(pattern.finditer(text))
            
            for match in reversed(matches):
                start, end = match.span()
                protected_text = f'__PHONE_START__{match.group(0)}__PHONE_END__'
                text = text[:start] + protected_text + text[end:]
                protected_count += 1
                
                if self.config.enable_debug_logging:
                    logger.debug(f"Protected phone pattern {i+1}: '{match.group(0)}'")
        
        if self.config.enable_debug_logging:
            logger.debug(f"Phone protection result: '{text}' (protected {protected_count} phones)")
        
        return text


    def _number_conversion_with_phone_protection(self, text: str) -> str:
        if '__PHONE_START__' not in text:
            return self.number_converter.convert_numbers(text)
        
        if self.config.enable_debug_logging:
            logger.debug(f"Number conversion with protection input: '{text}'")
        
        parts = re.split(r'(__PHONE_START__.*?__PHONE_END__)', text)
        
        result_parts = []
        for i, part in enumerate(parts):
            if part.startswith('__PHONE_START__') and part.endswith('__PHONE_END__'):
                protected_content = part[len('__PHONE_START__'):-len('__PHONE_END__')]
                
                phone_processed = self.phone_processor.normalize_phones(protected_content)
                
                result_parts.append(f'__PHONE_START__{phone_processed}__PHONE_END__')
                
                if self.config.enable_debug_logging:
                    logger.debug(f"Protected phone processing {i}: '{protected_content}' → '{phone_processed}'")
            else:
                if part.strip():
                    converted_part = self.number_converter.convert_numbers(part)
                    result_parts.append(converted_part)
                    if self.config.enable_debug_logging:
                        logger.debug(f"Converting non-protected part {i}: '{part}' → '{converted_part}'")
                else:
                    result_parts.append(part)
        
        result = ''.join(result_parts)
        
        if self.config.enable_debug_logging:
            logger.debug(f"Number conversion with protection result: '{result}'")
        
        return result

    def _remove_phone_protection(self, text: str) -> str:
        text = text.replace('__PHONE_START__', '')
        text = text.replace('__PHONE_END__', '')
        return text
    
class ChineseASRNormaliser:
    """AudioBench-compatible Chinese normaliser"""
    
    def __init__(self, enable_disfluency_removal: bool = True, **kwargs):
        self.enable_disfluency_removal = enable_disfluency_removal
        
        # Initialize the main normalizer with appropriate config
        config = ASRConfig()
        config.remove_filler_words = enable_disfluency_removal
        self.normalizer = ChineseASRNormalizer(config)
    
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
            
            return self.normalizer.normalize(text)
        
        except Exception as e:
            logger.error(f"Chinese normalization failed: {e}")
            # Fallback: basic text cleaning
            import re
            text = re.sub(r'\s+', ' ', text.strip())
            return text