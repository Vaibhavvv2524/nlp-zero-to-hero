"""Rule-based punctuation / sentence-boundary baseline.

No GPU and no trained model: boundaries are chosen with a small dynamic program
that combines (a) common sentence-final verb forms, (b) common sentence-initial
words, (c) ``NULL`` markers, and (d) a length prior around 16 words.
"""

import math

MIN_LEN = 4
MAX_LEN = 32
MEAN_LEN = 16.0
SIGMA_LEN = 4.0

W_END = 0.5
W_START = 0.5
W_CONJ = 0.2
W_BAD = 0.3
W_NULL = 1.0

END_WORDS = {
    "है", "हैं", "था", "थी", "थे", "हुआ", "हुई", "हुए", "हुईं",
    "गया", "गई", "गए", "गईं", "किया", "की", "किए", "कीं",
    "दिया", "दी", "दिए", "दीं", "लिया", "ली", "लिए", "लीं",
    "रहा", "रही", "रहे", "रहीं", "सकता", "सकती", "सकते", "सकतीं",
    "चाहिए", "होगा", "होगी", "होंगे", "आया", "आई", "आए", "आईं",
    "बना", "बनी", "बने", "बनीं", "मिला", "मिली", "मिले", "मिलीं",
    "कहा", "पड़ा", "पड़ी", "पड़े", "लगा", "लगी", "लगे",
    "रखा", "रखी", "रखे", "बताया", "बताई", "बताए",
    "जाता", "जाती", "जाते", "आता", "आती", "आते", "देता", "देती", "देते",
    "लेता", "लेती", "लेते", "करता", "करती", "करते", "होता", "होती", "होते",
    "पाता", "पाती", "पाते", "जानता", "जानती", "जानते", "समझता", "समझती",
    "समझते", "दिखता", "दिखती", "दिखते",
}

STARTERS = {
    "बहुत", "कई", "एक", "यह", "वह", "ये", "वे", "मैं", "हम", "तुम", "आप",
    "वो", "जब", "तब", "जहाँ", "वहाँ", "यहाँ", "आज", "कल", "अब", "फिर",
    "बाद", "पहले", "अंत", "अतः", "इसलिए", "इसके", "इसकी", "इसका", "उसके",
    "उसकी", "उसका", "लेकिन", "परंतु", "किंतु", "तथा", "और", "या", "अथवा",
    "केवल", "सिर्फ", "लगभग", "करीब", "प्रायः", "इस", "उस", "जो", "जिस",
    "जिन", "क्या", "कौन", "कब", "कहाँ", "कहां", "क्यों", "कैसे", "कितना",
    "कितनी", "कितने", "किस", "किसने", "किसका", "हाँ", "जी", "नहीं", "अरे",
    "ओह", "वाह", "शाबाश", "क्योंकि", "अगर", "यदि", "जैसे", "वैसे", "भगवान",
}

BAD_STARTERS = {
    "में", "का", "की", "के", "को", "से", "पर", "ने", "भी", "ही", "तो",
    "तक", "लिए", "साथ", "बिना", "द्वारा", "वाला", "वाली", "वाले", "है",
    "हैं", "था", "थी", "थे",
}

CONJUNCTIONS = {"और", "तथा", "लेकिन", "परंतु", "किंतु", "अथवा", "या", "एवं"}

QUESTION_STARTERS = {
    "क्या", "कौन", "कब", "कहाँ", "कहां", "क्यों", "कैसे", "कितना",
    "कितनी", "कितने", "किस", "किसने", "किसका", "किसकी", "किसे",
}

EXCLAMATION_STARTERS = {"वाह", "अरे", "ओह", "हाय", "शाबाश", "अहा"}


def _length_score(length):
    return -0.5 * ((length - MEAN_LEN) / SIGMA_LEN) ** 2


def _gap_score(prev_word, next_word, gap_index, null_positions):
    score = 0.0
    if gap_index in null_positions:
        score += W_NULL
    if prev_word in END_WORDS:
        score += W_END
    if next_word in STARTERS:
        score += W_START
    if next_word in CONJUNCTIONS:
        score += W_CONJ
    if next_word in BAD_STARTERS:
        score -= W_BAD
    if prev_word in CONJUNCTIONS or prev_word in BAD_STARTERS:
        score -= W_BAD
    return score


def segment_tokens(tokens, null_positions=()):
    """Split a token list into sentences using a Viterbi-style dynamic program."""
    n = len(tokens)
    if n == 0:
        return []
    if n <= MIN_LEN:
        return [list(tokens)]

    null_positions = set(null_positions)
    negative = float("-inf")
    best = [negative] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0

    for end in range(1, n + 1):
        shortest = 1 if end == n else MIN_LEN
        for length in range(shortest, MAX_LEN + 1):
            start = end - length
            if start < 0:
                break
            if best[start] == negative:
                continue
            gap = 0.0
            if start > 0:
                gap = _gap_score(
                    tokens[start - 1], tokens[start], start, null_positions
                )
            candidate = best[start] + gap + _length_score(length)
            if candidate > best[end]:
                best[end] = candidate
                back[end] = start

    sentences = []
    end = n
    while end > 0:
        start = back[end]
        sentences.append(tokens[start:end])
        end = start
    sentences.reverse()
    return sentences


def punctuate(sentence_tokens, add_commas=True):
    words = list(sentence_tokens)
    if add_commas:
        for index in range(3, len(words)):
            if words[index] in CONJUNCTIONS and not words[index - 1].endswith(","):
                words[index - 1] = words[index - 1] + ","

    if words and words[0] in QUESTION_STARTERS:
        end_mark = "?"
    elif words and words[0] in EXCLAMATION_STARTERS:
        end_mark = "!"
    else:
        end_mark = "।"
    return " ".join(words) + end_mark
