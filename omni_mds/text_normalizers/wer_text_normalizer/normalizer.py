#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Tuesday, April 30th 2024, 1:56:20 pm
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
#
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###


import re

import jiwer

from . import whisper_english

all_jiwer_process = jiwer.Compose(
    [
        jiwer.RemoveMultipleSpaces(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveKaldiNonWords(),
        jiwer.RemovePunctuation(),
    ]
)

EnglishTextNormalizer = whisper_english.EnglishTextNormalizer()


def normalize_text(text):
    """Normalize text by converting to lowercase and standardizing numbers."""

    # Digit to word conversions
    digits_to_words = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10": "ten",
        "11": "eleven",
        "12": "twelve",
        "13": "thirteen",
        "14": "fourteen",
        "15": "fifteen",
        "16": "sixteen",
        "17": "seventeen",
        "18": "eighteen",
        "19": "nineteen",
        "20": "twenty",
        "30": "thirty",
        "40": "forty",
        "50": "fifty",
        "60": "sixty",
        "70": "seventy",
        "80": "eighty",
        "90": "ninety",
    }
    for digit, word in digits_to_words.items():
        text = re.sub(r"\b" + digit + r"\b", word, text)

    # Expand common contractions
    contractions = {
        "i'm": "i am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are",
        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "doesn't": "does not",
        "don't": "do not",
        "didn't": "did not",
        "that's": "that is",

        # new contractions added 
        # "can't": "can not",
        # "won't": "will not",
        # "wouldn't": "would not",
        # "couldn't": "could not",
        # "shouldn't": "should not",
        # "you'll": "you will",
        # "they'll": "they will",
        # "that'll": "that will",
        # "it'll": "it will",
        # "here's": "here is",
        # "there's": "there is",
        # "what's": "what is",
        # "china's": "china is",
        # "pakistan's": "pakistan is",
        # "london's": "london is"
    }
    for contraction, expanded in contractions.items():
        text = re.sub(r"\b" + contraction + r"\b", expanded, text)

    return text


# def remove_punctuation(text):
#     """ Remove punctuation from text. """
#     text = re.sub(r'[^\w\s]', '', text)  # Remove all except word characters and whitespace
#     return text


def remove_non_speech_elements(text):
    """Remove common non-speech elements like 'uh', 'um', etc."""
    non_speech_patterns = r"\b(uh|umm|um|er|ah)\b"
    # non_speech_patterns = r"\b(uh|umm|um|er|ah|lah|err|eh|lor|mah|meh|loh)\b"
    text = re.sub(non_speech_patterns, "", text)
    return text


def remove_parentheses(text):
    return re.sub(r"(\[|\(|\{|\<)[^\(\)\\n\[\]]*(\]|\)|\}|\>)", "", text)


def normalize_imda(text):
    return " ".join(
        re.sub(
            "\s?(\[|\(|<)[a-zA-Z]*(\]|\)|>)\s?",
            " ",
            re.sub(
                "<(tamil|malay|mandarin)>([^<>:]*):?([^<>:]*)</(tamil|malay|mandarin)>",
                r"\2",
                text,
            ),
        ).split()
    ).strip()


def normalize_gigaspeech(text):
    text = re.sub("<SIL>|<MUSIC>|<NOISE>|<OTHER>", "", text)
    text = re.sub("<COMMA>", ",", text)
    text = re.sub("<PERIOD>", ".", text)
    text = re.sub("<QUESTIONMARK>", "?", text)
    text = re.sub("<EXCLAMATIONPOINT>", "!", text)
    return text


def preprocess_text_asr(text, dataset="default", target_lang="en"):

    if dataset == "gigaspeech":
        text = normalize_gigaspeech(text)
    elif dataset == "imda":
        text = normalize_imda(text)

    # All Adapt to Lower Case
    text = text.lower()

    # Borrow from Whisper
    text = EnglishTextNormalizer(text)

    # Should be handled by EnglishNumberNormalizer and EnglishTextNormalizer
    # Does not hurt to have it here
    text = normalize_text(text)

    # Use regex to remove all things between brackets [] () {} <>
    text = remove_parentheses(text)

    # Add some standard process from jiwer
    text = all_jiwer_process(text)

    # Remove some non-speech elements
    text = remove_non_speech_elements(text).strip()

    # Separate Chinese characters
    # if "zh" in target_lang:
    text = separate_and_space_chinese(text)
    text = separate_and_space_thai(text)

    return text


def separate_and_space_chinese(text):
    # Separate Chinese characters and non-Chinese parts
    parts = re.split(r"([\u4e00-\u9fff]+)", text)

    # Add spaces between Chinese characters and join the parts back together
    processed_parts = []
    for part in parts:
        if re.match(r"[\u4e00-\u9fff]+", part):
            spaced = " ".join(char for char in part)  # Add space between Chinese characters
            processed_parts.append(spaced)
        else:
            processed_parts.append(part)

    processed_str = " ".join(processed_parts)
    return re.sub("\s+", " ", processed_str)


def separate_and_space_thai(text):
    # Separate Chinese characters and non-Chinese parts
    parts = re.split(r"([\u0E00-\u0E7F]+)", text)

    # Add spaces between Chinese characters and join the parts back together
    processed_parts = []
    for part in parts:
        if re.match(r"[\u0E00-\u0E7F]+", part):
            spaced = " ".join(char for char in part)
            processed_parts.append(spaced)
        else:
            processed_parts.append(part)

    processed_str = " ".join(processed_parts)
    return re.sub("\s+", " ", processed_str)


def add_speaker_asr(text):
    if not re.search("<Speaker\d>", text):
        return f"<Speaker1>: {text}"
    return text


def detect_language(text):
    # Define Unicode ranges for each language
    language_patterns = {
        'zh': r'[\u4E00-\u9FFF]',  # Chinese characters
        'th': r'[\u0E00-\u0E7F]',  # Thai characters
        'ta': r'[\u0B80-\u0BFF]',  # Tamil characters
        'vi': r'[\u0102\u0103\u0110\u0111\u0128\u0129\u0168\u0169\u01A0\u01A1\u01AF\u01B0\u1EA0-\u1EF9]',  # Vietnamese-specific characters
    }
    
    langs_found = []
    # Check for each language pattern in the text
    for lang_code, pattern in language_patterns.items():
        if re.search(pattern, text):
            langs_found.append(lang_code)
    return langs_found


def add_language_tag(text, example, semantic=False):
    if semantic:
        language_mapper = {
            'en': 'English',
            'id': 'Indonesian',
            'ja': 'Japanese',
            'jv': 'Javanese',
            'th': 'Thai',
            'iba': 'Iban',
            'hok': 'Hokkien',
            'tl': 'Tagalog',
            'vi': 'Vietnamese',
            'my': 'Burmese',
            'ne': 'Nepali',
            'ms': 'Malay',
            'zh': 'Chinese',
            'codeswitch': 'English,{others}',
            'ko': 'Korean',
            'yue': 'Cantonese',
            'ta': 'Tamil',
            'lo': 'Lao',
            'unk': 'Unknown'
        }
    else:
        language_mapper = {
            'en': '<unused0>',
            'id': '<unused1>',
            'ja': '<unused2>',
            'jv': '<unused3>',
            'th': '<unused4>',
            'iba': '<unused5>',
            'hok': '<unused12><unused6>',
            'tl': '<unused7>',
            'vi': '<unused8>',
            'my': '<unused9>',
            'ne': '<unused10>',
            'ms': '<unused11>',
            'zh': '<unused12>',
            'codeswitch': '<unused13>',
            'ko': '<unused14>',
            'yue': '<unused12><unused15>',
            'ta': '<unused16>',
            'lo': '<unused17>',
            'unk': '<unused18>'
        }
    language_tag = example.get("language", "unk")
    if language_tag not in language_mapper:
        language_tag = "unk"

    lang_token = language_mapper[language_tag]

    if language_tag == "codeswitch":
        if semantic:
            if sub_langs:=detect_language(text):
                sub_token = ",".join([language_mapper[sl] for sl in sub_langs])
            else:
                sub_token = "other"
            lang_token = lang_token.format(others=sub_token)
        else:
            for sub_lang in detect_language(text):
                lang_token += language_mapper[sub_lang]

    if semantic:
        lang_token = f"<speech_language: {lang_token}>"

    return lang_token + text


def standardize_chinese(text):
    pattern = r'(?<=[\u4E00-\u9FFF])\s+(?=[\u4E00-\u9FFF])'
    # Use re.sub() to replace the matched spaces with an empty string
    return re.sub(pattern, '', text).strip()


def standardize_thai(text):
    pattern = r'(?<=[\u0E00-\u0E7F])\s+(?=[\u0E00-\u0E7F])'
    # Use re.sub() to replace the matched spaces with an empty string
    return re.sub(pattern, '', text).strip()


def standardize_apos(text):
    """
    TODO: remove this once the apostrophe restoration is done.
    """
    case_sensitive_contractions = {
        "Im": "I'm",
        "Ive": "I've",
        "Hes": "He's",
        "hes": "he's",
        "Its": "It's",
        "Isnt": "Isn't",
        "isnt": "isn't",
    }

    contractions = {
        "Arent": "Aren't", #
        "Dont": "Don't", # 
        "Doesnt": "Doesn't", #
        "Didnt": "Didn't", #
        "Cant": "Can't",
        "Wont": "Won't",
        "Wasnt": "Wasn't",
        "Werent": "Weren't",
        "Wouldnt": "Wouldn't",
        "Couldnt": "Couldn't",
        "Shouldnt": "Shouldn't",
        "Hasnt": "Hasn't",
        "Hadnt": "Hadn't",
        "Havent": "Haven't",
        "Youll": "You'll",
        "Theyll": "They'll",
        "Thatll": "That'll",
        "Youre": "You're",
        "Theyre": "They're",
        "Youve": "You've",
        "Weve": "We've",
        "Theyve": "They've",
        "Shes": "She's",
        "Heres": "Here's",
        "Theres": "There's",
        "Thats": "That's",
        "Whats": "What's",
        "Lets": "Let's",
        "Chinas": "China's",
        "Pakistans": "Pakistan's",
        "Londons": "London's",
    }

    # for contraction, expanded in case_sensitive_contractions.items():
    #     text = re.sub(r"\b" + contraction + r"\b", expanded, text)

    for contraction, expanded in contractions.items():
        text = re.sub(r"\b" + contraction + r"\b", expanded, text)
        text = re.sub(r"\b" + contraction.lower() + r"\b", expanded.lower(), text)

    return text

# Mapping for variants that should be normalized.
FILLER_MAPPING = {
    "oh": "oh", 
    "ow": "oh", 
    "ohh": "oh", 
    "ohhh": "oh",
    "um": "um", 
    "umm": "um", 
    "em": "um", 
    "urm": "um",
    "hm": "hm", 
    "hmm": "hm", 
    "mmhmm": "hm", 
    "mm": "hm", 
    "mmm": "hm",
    "err": "err", 
    "er": "err", 
    # "eh": "err", 
    "errm": "err",
    "lo": "loh", 
    # "loh": "loh", 
    # "lor": "loh",
    "wa": "wah", 
    "wah": "wah",
    "arrrr": "ah",
    "ar": "ah",
    "arr": "ah",
}

# The complete set of target filler words (after normalization)
TARGET_FILLERS = {"ah", "uh", "eh", "oh", "um", "hm", "err", 
                  "lah", "leh", "hah", "huh", "wow", 
                  "wah", "walao", "siah", "mah", "meh", "aiyah", "loh", "lor"}

def count_leading_trailing_spaces(s):
    leading_spaces = len(s) - len(s.lstrip())
    trailing_spaces = len(s) - len(s.rstrip())
    return leading_spaces, trailing_spaces

def normalize_filler(candidate):
    """
    Strip extra punctuation/whitespace, then split on hyphen.
    For each part, if it is in our mapping or in our target set, 
    we normalize it (using the mapping if available, otherwise lower-case it).
    If any part is not a recognized filler word, return None.
    """
    # Remove common wrapping punctuation and spaces.
    cleaned = candidate.strip(" []()!?,;:").strip()
    if not cleaned:
        return None

    # Split compound filler words connected by hyphens.
    parts = cleaned.split('-')
    normalized_parts = []
    for part in parts:
        lower_part = part.lower()
        if lower_part in FILLER_MAPPING:
            normalized_parts.append(FILLER_MAPPING[lower_part])
        elif lower_part in TARGET_FILLERS:
            normalized_parts.append(lower_part)
        else:
            # If any part is not a filler word candidate, do not change.
            return None
    return "-".join(normalized_parts)

def filler_replacer(match):
    """
    This is the replacement function for re.sub.
    It takes the matched token, cleans and normalizes it.
    If it qualifies as a filler word (or compound filler),
    return the standardized version wrapped in parentheses.
    Otherwise, return the original match.
    """
    token = match.group(0)
    ls, rs = count_leading_trailing_spaces(token)
    norm = normalize_filler(token)
    if norm is not None:
        return " "*ls + f"({norm})" + " "*rs
    else:
        return token

def standardize_fillers(text):
    """
    Process the input text and replace all filler words (even those already
    wrapped or connected with hyphens) with their normalized version 
    wrapped in round brackets.
    
    The regex below uses negative lookbehind and lookahead so that only
    “standalone” sequences (possibly with extra wrapping punctuation) are matched.
    """
    # This pattern matches a sequence that may have leading/trailing punctuation/spaces.
    pattern = r'(?<![a-zA-Z0-9_])([\[\(\s]*[A-Za-z]+(?:-[A-Za-z]+)*[\]\)\?\s]*)(?![a-zA-Z0-9_])'
    text = re.sub(pattern, filler_replacer, text)
    text = re.sub("[\(\[（【]+", "(", text)
    text = re.sub("[\)\]）】]+", ")", text)
    text = re.sub("\s+", " ", text)
    return text.strip()


def cast_lower_case(text):
    text = text.lower()
    text = re.sub("<speaker", "<Speaker", text)
    return text


def remove_no_sound_elements(text):
    text = re.sub("\n", " ", text)
    text = re.sub("[\(<](SPN|ppo|ppl|ppb|ppc|laugh|unk|unknown)[>\)]", "", text, flags=re.IGNORECASE)
    text = re.sub("\s+", " ", text)
    return text.strip()


# Compile once for efficiency
TAG_PATTERN = re.compile(r"#([a-zA-Z,_\-.。，]+)#")
PUNCT_PATTERN = re.compile(r"[,_\-.。，]")


def replace_punct_in_tags(text):
    """
    Replace any -, _, comma, or period inside hashtag-wrapped segments
    with a single space, but leave all other hashtags untouched.
    """
    def _repl(match):
        content = match.group(1)
        # Only transform tags containing the target punctuation
        if PUNCT_PATTERN.search(content):
            # Replace punctuation with space
            new_content = PUNCT_PATTERN.sub(" ", content)
            # Collapse multiple spaces (optional)
            new_content = re.sub(r"\s+", " ", new_content).strip()
            return f"#{new_content}#"
        # Otherwise return the original hashtag
        return match.group(0)

    # Apply across the entire text
    return TAG_PATTERN.sub(_repl, text)


hash_match_corrections = {
    "wah lau": "walao",
    "kena": "kena",
    "kopi": "kopi",
    "pak tor": "paktor",
    "da bao": "dabao",
    "da pao": "dabao",
    "ta pao": "dabao",
    "kay poh": "kaypoh",
    "ang moh": "ang moh",
    "ang mo": "ang moh",
    "ang bao": "ang bao",
    "ah ma": "ah ma",
    "ah mah": "ah ma",
    "ah lian": "ah lian",
    "ah beng": "ah beng",
    "ah gong": "ah gong",
    "ah pek": "ah pek",
    "xiao mei mei": "小妹妹",
    "tom yum": "tom yum",
    "bo pian": "bo bian",
    "bo bian": "bo bian",
    "ma la": "mala",
    "gai gai": "gai gai",
    "pad thai": "pad thai",
    "hor fun": "hor fun",
    "ko song": "kosong",
    "mee soto": "mee soto",
    "mee pok": "mee pok",
    "ban mian": "ban mian",
    "teo heng": "Teo Heng",
    "kang kong": "kang kong",
    "hwa chong": "Hwa Chong",
    "swee choon": "Swee Choon",
    "swee koon": "Swee Koon",
    "tai seng": "Tai Seng",
    "yew tee": "Yew Tee",
    "chong pang": "Chong Pang",
    "and mo kio": "Ang Mo Kio",
    "yuan ching": "Yuan Ching",
    "jun hup": "Jun Hup",
    "hian ching": "Hian Ching",
    "mugen rao": "Mugen Rao",
}


universal_corrections = { 
    "edgeware road": "Edgware Road",
    "ang mo kio": "Ang Mo Kio",
    "yio chu kang": "Yio Chu Kang",
    "chua chu kang": "Chua Chu Kang",
    "choa chu kang": "Chua Chu Kang",
    "lim chu kang": "Lim Chu Kang",
    "phua chu kang": "Phua Chu Kang",
    "tampines": "Tampines",
    "bedok": "Bedok",
    "jurong": "Jurong",
    "changi": "Changi",
    "yishun": "Yishun",
    "bugis": "Bugis",
    "kembagan": "Kembangan",
    "kembangan": "Kembangan",
    "johor bahru": "Johor Bahru",
    "paya lebar": "Paya Lebar",
    "bukit batok": "Bukit Batok",
    "bukit panjang": "Bukit Panjang",
    "bukit timah": "Bukit Timah",
    "bukit merah": "Bukit Merah",
    "bukit gombak": "Bukit Gombak",
    "potong pasir": "Potong Pasir",
    "tanah merah": "Tanah Merah",
    "tanjong pagar": "Tanjong Pagar",
    "tanjong katong": "Tanjong Katong",
    "dhoby ghaut": "Dhoby Ghaut",
    "boon keng": "Boon Keng",
    "bras basah": "Bras Basah",
    "joo chiat": "Joo Chiat",
    "telok ayer": "Telok Ayer",
    "telok blangah": "Telok Blangah",
    "sentosa": "Sentosa",
    "somerset": "Somerset",
    "bangkok": "Bangkok",
    "buangkok": "Buangkok",
    "shanghai": "Shanghai",
    "beijing": "Beijing",
    "sengkang": "Sengkang",
    "punggol": "Punggol",
    "hougang": "Hougang",
    "serongoon": "Serangoon",
    "serangoon": "Serangoon",
    "bishan": "Bishan",
    "geylang": "Geylang",
    "yishun": "Yishun",
    "sembawang": "Sembawang",
    "pasir ris": "Pasir Ris",
    "toa payoh": "Toa Payoh",
    "pulau ubin": "Pulau Ubin",
    "tiong bahru": "Tiong Bahru",
    "jalan kayu": "Jalan Kayu",
    "jalan besar": "Jalan Besar",
    "telok blangah": "Telok Blangah",
    "boon lay": "Boon Lay",
    "kranji": "Kranji",
    "woodlands": "Woodlands",
    "woodland": "Woodlands",
    "ngee ann": "Ngee Ann",
    "ho chi minh": "Ho Chi Minh",
    "dee kosh": "Dee Kosh",
    "lee hsien loong": "Lee Hsien Loong",
    "lee kuan yew": "Lee Kuan Yew",
    "tan tock seng": "Tan Tock Seng",
    "goh chok tong": "Goh Chok Tong",
    "jian hao tan": "Jian Hao Tan",
    "lee min ho": "Lee Min Ho",
    "wee kim wee": "Wee Kim Wee",
    "kim soo hyun": "Kim Soo Hyun",
    "xi jin ping": "Xi Jin Ping",
    "kim jong un": "Kim Jong Un",
    "seo yong hak": "Seo Yong Hak",
    "zhao li ying": "Zhao Li Ying",
    "zheng hui yu": "Zheng Hui Yu",
    "toh jun long": "Toh Jun Long",
    "teo kyan leng": "Teo Kyan Leng",
    "ramya pandian": "Ramya Pandian",
    "hai di lao": "Hai Di Lao",
    "din tai fung": "Din Tai Fung",
    "tim ho wan": "Tim Ho Wan",
    "don Don Donki": "Don Don Donki",
    "lau pa sat": "Lau Pa Sat",
    "thian hock keng temple": "Thian Hock Keng Temple",
    "phoon huat": "Phoon Huat",
    "gong cha": "Gong Cha",
    "sheng siong": "Sheng Siong",
    "thye hua kwan": "Thye Hua Kwan",
    "bayern munich": "Bayern Munich",
    "Sepak Takraw": "Sepak Takraw",
    "yu gi oh": "Yu Gi Oh",
    "pasar malam": "pasar malam",
    "karang guni": "karang guni",
    "kampung": "kampung",
    "kopitiam": "kopitiam",
    "hari raya": "Hari Raya",
    "chee cheong fun": "chee cheong fun",
    "zhup cai png": "zhup cai png",
    "bak chor mee": "bak chor mee",
    "char kway teow": "char kway teow",
    "kway teow": "kway teow",
    "kway chap": "kway chap",
    "chai tow kuay": "chai tow kway",
    "cai tao kway": "chai tow kway",
    "ang ku kueh": "ang ku kueh",
    "huat kueh": "huat kueh",
    "bak kut teh": "bak kut teh",
    "teh tarik": "teh tarik",
    "nasi lemak": "nasi lemak",
    "nasi goreng": "nasi goreng",
    "nasi padang": "nasi padang",
    "nasi briyani": "nasi briyani",
    "mee goreng": "mee goreng",
    "hokkien mee": "hokkien mee",
    "wanton mee": "wanton mee",
    "char siew bao": "char siew bao",
    "xiao long bao": "xiao long bao",
    "char siew": "char siew",
    "siew dai": "siew dai",
    "siew mai": "siew mai",
    "ayam penyet": "ayam penyet",
    "gulab jamun": "gulab jamun",
    "tai wan": "Taiwan",
    "tai pei": "Taipei",
    "teochew": "Teochew",
    "peking": "Peking",
    "jia lat": "jialat",
    "bao ka liao": "bao ka liao"
}


def correct_words(text):
    for source, target in hash_match_corrections.items():
        match_regex = "[\s,.，。\-_]{0,2}".join(source.split(" "))
        match_regex = f"(?<=#){match_regex}(?=#)"
        text = re.sub(match_regex, target, text, flags=re.IGNORECASE)

    for source, target in universal_corrections.items():
        match_regex = "[\s,.，。\-_]{0,2}".join(source.split(" "))
        match_regex = f"(?<![a-zA-Z0-9_]){match_regex}(?![a-zA-Z0-9_])"
        text = re.sub(match_regex, target, text, flags=re.IGNORECASE)

    text = replace_punct_in_tags(text)
    return text


def remove_hash_tag(response_text):
    response_text = response_text.replace("#", " ") # remove hash tag
    response_text = re.sub("\s+", " ", response_text).strip()
    return response_text