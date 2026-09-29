language_mapping = {
    "en": 0,
    "id": 1,
    "ja": 2,
    "jv": 3,
    "th": 4,
    "iba": 5,
    "hok": 6,
    "tl": 7,
    "vi": 8,
    "my": 9,
    "ne": 10,
    "ms": 11,
    "zh": 12,
    "codeswitch": 13,
    "ko": 14,
    "yue": 15,
    "ta": 16,
    "lo": 17,
    "unk": -1,
}


def get_language_id(language):
    if language in language_mapping:
        return language_mapping[language]
    return language_mapping["unk"]
