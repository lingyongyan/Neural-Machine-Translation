import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import re
import time
import unicodedata


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


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


def validate_language(l):
    p = '../data/{}.txt'.format(l)
    p = os.path.abspath(p)
    print(p)

    if not os.path.exists(p):
        url = 'http://www.manythings.org/anki/'
        print("{}.txt does not exist in the data directory. Please go to '{}' and download the data set.".format(l, url))
        exit(1)


def validate_language_params(l):
    is_missing = (not os.path.exists('../data/attention_params_{}'.format(l))
                  or not os.path.exists('../data/decoder_params_{}'.format(l))
                  or not os.path.exists('../data/encoder_params_{}'.format(l)))

    if is_missing:
        print("Model params for language '{}' do not exist in the data directory. Please train a new model for this language.".format(l))
        exit(1)
