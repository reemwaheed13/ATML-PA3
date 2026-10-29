"""VQA answer normalization + soft consensus accuracy. (handout p9)

Natural VQAv2 is scored with soft consensus accuracy against all ten human answers:
    s(a_hat) = (1/10) * sum_{i=1..10} min(1, (1/3) * sum_{j != i} 1[a_hat == a_j])
Grounding questions use normalized exact match (presence and counting kept separate).

The answer normalization follows the official VQA evaluation code (see README "Attribution"),
applied consistently throughout training and scoring. The contraction / digit / article maps below
are reproduced from that code so our normalization matches the benchmark's.
"""

from __future__ import annotations

import re

# --- Official VQA (v2) normalization tables (Goyal et al.; VQA eval code) -------------------------

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
    "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's",
    "somethingd": "something'd", "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
    "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
    "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}

_MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

_ARTICLES = {"a", "an", "the"}

_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT = [
    ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@",
    "`", ",", "?", "!",
]


def _process_punctuation(text: str) -> str:
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or (_COMMA_STRIP.search(text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return _PERIOD_STRIP.sub("", out)


def _process_digit_article(text: str) -> str:
    words = []
    for word in text.lower().split():
        word = _MANUAL_MAP.get(word, word)
        if word not in _ARTICLES:
            words.append(word)
    for i, word in enumerate(words):
        if word in _CONTRACTIONS:
            words[i] = _CONTRACTIONS[word]
    return " ".join(words)


def normalize_answer(ans: str) -> str:
    """Apply the official VQA answer normalization (whitespace, punctuation, digits, articles)."""
    if ans is None:
        return ""
    ans = ans.replace("\n", " ").replace("\t", " ").strip()
    ans = _process_punctuation(ans)
    ans = _process_digit_article(ans)
    return ans


def soft_consensus_accuracy(pred: str, answers: list[str]) -> float:
    """Return s(a_hat) over the human answers (ten for VQAv2), min(1, matches/3) averaged leave-one-out."""
    if not answers:
        return 0.0
    p = normalize_answer(pred)
    norm = [normalize_answer(a) for a in answers]
    n = len(norm)
    total = 0.0
    for i in range(n):
        matches = sum(1 for j in range(n) if j != i and norm[j] == p)
        total += min(1.0, matches / 3.0)
    return total / n


def normalized_exact_match(pred: str, target: str) -> float:
    """Return 1.0 if normalized pred == normalized target else 0.0 (grounding questions)."""
    return 1.0 if normalize_answer(pred) == normalize_answer(target) else 0.0
