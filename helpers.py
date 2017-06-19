import re
import string
import unicodedata


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# Turns a unicode string to plain ASCII (http://stackoverflow.com/a/518232/2809427)
def unicode_to_ascii(s):
    chars = [c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn']
    char_list = ''.join(chars)
    return char_list


