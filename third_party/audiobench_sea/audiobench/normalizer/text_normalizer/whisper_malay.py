import json
import os
import re
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union

from more_itertools import windowed
import string

# import sys
# sys.path.append(os.getcwd())

from .basic import remove_symbols_and_diacritics
from .whisper_english import EnglishTextNormalizer

# Lazy import malaya to avoid circular import issues with torchvision
_malaya = None

def _get_malaya():
    """Lazy import malaya module to avoid circular import issues."""
    global _malaya
    if _malaya is None:
        try:
            import malaya
            _malaya = malaya
        except ImportError as e:
            raise ImportError(
                "malaya is required for MalayCodeSwitchEnglishSpellingNormalizer. "
                "Install it with: pip install malaya"
            ) from e
    return _malaya

EnglishSpellingNormalizer = EnglishTextNormalizer()

class MalayCodeSwitchEnglishSpellingNormalizer:

    # Function to check if a character is punctuation
    def is_punctuation(self, char):
        return char in string.punctuation
    
    def malayNumberNormalizer(self, word):  
        # 1. change number --> digits
        malaya = _get_malaya()
        converted = malaya.word2num.word2num(word)
        
        return converted
    
    def englishSpellingNormalizer(self, text):
        malaya = _get_malaya()
        ignore_pattern = ["(?:&\w+)*"]
        # Regular expression to match words and punctuation
        pattern = f'\w+{"|".join(ignore_pattern)}|[^\w\s]'
        
        tokens = re.findall(pattern, text)
        
        processed_tokens = []
        for token in tokens:
            if not malaya.dictionary.is_malay(token) and not self.is_punctuation(token):
                # Borrow from Whisper
                token = EnglishSpellingNormalizer(token)
                processed_tokens.append(token)
            else:
                try:
                    token = str(self.malayNumberNormalizer(token)) # convert number to digit
                except ValueError:
                    token = token
                
                processed_tokens.append(token)
        
        text = " ".join(tokens)
        # breakpoint()
                
        text = remove_symbols_and_diacritics(text) # remove symbels / marks
        text = re.sub(r'\s+', ' ', text).strip() # remove multiple space
        
        return text
