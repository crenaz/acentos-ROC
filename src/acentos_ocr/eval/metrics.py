from __future__ import annotations

import re
from collections import Counter

# Transcriptions are Markdown; OCR output is plain text. Comparing them raw would
# score formatting rather than recognition.
_HEADING = re.compile(r"^#+\s*", re.M)
_BULLET = re.compile(r"^[.·*-]\s*", re.M)
_WHITESPACE = re.compile(r"\s+")

# A handful of transcriptions label a graphic that carries text, e.g.
# `MAIN LOGO: Crest of the Cayman Islands`. The label is the transcriber's
# annotation and is not printed on the page, so it is dropped; the value after it
# usually is printed, so it is kept. This affects 3 lines across the corpus.
_ANNOTATION = re.compile(r"^(?:[A-Z]+\s+)?LOGO:\s*", re.M | re.I)


def normalise(text: str) -> str:
    """Strip Markdown scaffolding and transcriber annotations, collapse whitespace."""
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _ANNOTATION.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip().lower()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        previous = current
    return previous[-1]


def edit_counts(truth: str, hypothesis: str) -> tuple[int, int]:
    """(edit distance, truth length). Returned as counts so a corpus-wide rate can
    be micro-averaged -- summing errors and lengths separately -- rather than
    averaging per-image rates, which would let a two-line advert outweigh a page."""
    return levenshtein(truth, hypothesis), len(truth)


def word_miss_counts(truth: str, hypothesis: str) -> tuple[int, int]:
    """
    (missed word tokens, total word tokens), compared as multisets so a word
    occurring three times in the truth and once in the output counts as two misses.
    """
    truth_words = Counter(truth.split())
    total = sum(truth_words.values())
    found = sum((truth_words & Counter(hypothesis.split())).values())
    return total - found, total


def cer(truth: str, hypothesis: str) -> float:
    """
    Character error rate: edit distance over the length of the truth.

    Order-sensitive. Text that was read perfectly but emitted in the wrong
    sequence scores as badly as text that was never read at all -- which is a
    real failure mode here, since Tesseract's layout analysis can shred a
    full-width paragraph into column-shaped blocks. Pair this with
    `word_miss_rate` to tell the two apart.
    """
    distance, length = edit_counts(truth, hypothesis)
    if length == 0:
        return 0.0 if not hypothesis else 1.0
    return distance / length


def word_miss_rate(truth: str, hypothesis: str) -> float:
    """
    Fraction of ground-truth word tokens that appear nowhere in the output.

    Order-insensitive, and deliberately scaled to be read against `cer`:

    * both low                -> the page was read correctly.
    * both high               -> a recognition failure; the words are not there.
    * CER high, miss rate low -> the words were recognised but emitted out of
                                 reading order. A segmentation problem, not an
                                 image-quality one.
    """
    missed, total = word_miss_counts(truth, hypothesis)
    return missed / total if total else 0.0
