import string
import hanlp
from hanlp.common.trie import Trie

tokenizer_zh = hanlp.load('LARGE_ALBERT_BASE')
# tokenizer_zh = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
tokenizer_en = hanlp.utils.rules.tokenize_english

trie = Trie()
custom_dict = {
    ' ': 0,  # split by space can avoid exceeding the max sequence length of 126
    '。': 0,
    '？': 0,
    '！': 0,
    '.': 0,
    '?': 0,
    '!': 0,
}


def load_custom_dict(infile):
    inf = open(infile)
    for line in inf:
        keyword = line.rstrip()
        custom_dict[keyword] = 1
    inf.close()


load_custom_dict('custom_dict.txt')
trie.update(custom_dict)

#: A string containing Chinese punctuation marks (non-stops).
non_stops = (
    # Fullwidth ASCII variants
    '\uFF02\uFF03\uFF04\uFF05\uFF06\uFF07\uFF08\uFF09\uFF0A\uFF0B\uFF0C\uFF0D'
    '\uFF0F\uFF1A\uFF1B\uFF1C\uFF1D\uFF1E\uFF20\uFF3B\uFF3C\uFF3D\uFF3E\uFF3F'
    '\uFF40\uFF5B\uFF5C\uFF5D\uFF5E\uFF5F\uFF60'

    # Halfwidth CJK punctuation
    '\uFF62\uFF63\uFF64'

    # CJK symbols and punctuation
    '\u3000\u3001\u3003'

    # CJK angle and corner brackets
    '\u3008\u3009\u300A\u300B\u300C\u300D\u300E\u300F\u3010\u3011'

    # CJK brackets and symbols/punctuation
    '\u3014\u3015\u3016\u3017\u3018\u3019\u301A\u301B\u301C\u301D\u301E\u301F'

    # Other CJK symbols
    '\u3030'

    # Special CJK indicators
    '\u303E\u303F'

    # Dashes
    '\u2013\u2014'

    # Quotation marks and apostrophe
    '\u2018\u2019\u201B\u201C\u201D\u201E\u201F'

    # General punctuation
    '\u2026\u2027'

    # Overscores and underscores
    '\uFE4F'

    # Small form variants
    '\uFE51\uFE54'

    # Latin punctuation
    '\u00B7'
)

#: A string of Chinese stops.
stops = (
    '\uFF01'  # Fullwidth exclamation mark
    '\uFF1F'  # Fullwidth question mark
    '\uFF61'  # Halfwidth ideographic full stop
    '\u3002'  # Ideographic full stop
)

#: A string containing all Chinese punctuation.
punctuation = non_stops + stops


def split_sents(text: str, trie: Trie):
    words = trie.parse_longest(text)
    sents = []
    pre_start = 0
    offsets = []
    for word, value, start, end in words:
        if pre_start != start:
            point = pre_start + 126
            while point < start:
                sents.append(text[pre_start: point])
                offsets.append(pre_start)
                pre_start = point
                point += 126
            if pre_start < start:
                sents.append(text[pre_start: start])
                offsets.append(pre_start)
        pre_start = end
    end = len(text)
    if pre_start != end:
        point = pre_start + 126
        while point < end:
            sents.append(text[pre_start: point])
            offsets.append(pre_start)
            pre_start = point
            point += 126
        if pre_start < end:
            sents.append(text[pre_start:])
            offsets.append(pre_start)
    return sents, offsets, words


def merge_parts(parts, offsets, words):
    items = [(i, p) for (i, p) in zip(offsets, parts)]
    items += [(start, [word]) for (word, value, start, end) in words]
    # In case you need the tag, use the following line instead
    # items += [(start, [(word, value)]) for (word, value, start, end) in words]
    return [each for x in sorted(items) for each in x[1]]


tokenizer = hanlp.pipeline().append(split_sents, output_key=('parts', 'offsets', 'words'), trie=trie) \
    .append(tokenizer_zh, input_key='parts', output_key='tokens') \
    .append(merge_parts, input_key=('tokens', 'offsets', 'words'), output_key='merged')


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False


def has_chinese(text):
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp):
            return True
    return False


def load_stop_words(infile):
    inf = open(infile)
    stop_word_set = set()
    for line in inf:
        stop_word_set.add(line.rstrip())
    inf.close()
    return stop_word_set


# stop_words = load_stop_words('stop_words.txt')


def tokenize(text):
    ret = tokenizer(text).merged if has_chinese(text) else tokenizer_en(text)

    def filter_fn(x):
        # filter spaces and punctuation
        return x != ' ' and x not in punctuation and x not in string.punctuation  # and x not in stop_words
    return list(filter(filter_fn, ret))
