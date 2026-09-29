from .burmese import *
from .chinese import *
from .english import *
from .indonesian import *
from .japanese import *
from .korean import *
from .lao import *
from .malay import *
from .nepali import *
from .tagalog import *
from .tamil import *
from .thai import *
from .vietnamese import *

default_lang_code = "en"


vars = locals()
instructions_dict = {
    lang_code: {
        "asr": vars[f"asr_instructions_{lang_code}"],
        "ac": vars[f"ac_instructions_{lang_code}"],
        "no_punctuation": vars[f"no_punctuation_instructions_{lang_code}"],
        "lowercase": vars[f"lowercase_instructions_{lang_code}"],
        "uppercase": vars[f"uppercase_instructions_{lang_code}"],
    }
    for lang_code in ["my", "zh", "en", "id", "ja", "ko", "lo", "ms", "ne", "tl", "ta", "th", "vi"]
}


def get_instructions_list(lang_code):
    if lang_code not in instructions_dict:
        return instructions_dict[default_lang_code]
    return instructions_dict[lang_code]
