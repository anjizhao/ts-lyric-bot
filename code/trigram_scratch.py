
from collections import defaultdict
import random
import string
from typing import Dict, List, Tuple

from nltk import trigrams


TRANSLATION_TABLE = str.maketrans(dict.fromkeys(string.punctuation))


def read_tweets() -> List[str]:
    with open('data/2011_tweets.txt') as txtfile:
        text = txtfile.read().strip()
        sentences = text.split('\n\n')
    return sentences


def clean_sentence(text: str) -> List[str]:
    ''' lowercase, remove punctuation, and split on spaces. '''
    text = text.lower().replace('\n', ' ').translate(TRANSLATION_TABLE)
    return text.split(' ')


def build_document(sentences: List[str]) -> List[str]:
    ''' combine all sentences into one giant list of words '''
    document: List[str] = []
    for s in sentences:
        words = clean_sentence(s)
        document.extend(words + ['.'])
    return document


def count_grams(
    document: List[str],
) -> Tuple[Dict[Tuple[str, str], Dict[str, int]], List[str]]:

    counts: Dict[Tuple[str, str], Dict[str, int]]
    counts = defaultdict(lambda: defaultdict(int))
    # 1st level keys of dict are the "grams"; values are dicts
    # 2nd level dict keys are each word following the gram w/ value count

    start_words: List[str] = []  # track sentence "starts"
    start_words.append(document[0])

    for w0, w1, w2 in trigrams(document):
        if w0 == '.':
            start_words.append(w1)
        counts[(w0, w1)][w2] += 1

    return counts, start_words


def counts_to_probabilities(
    counts: Dict[Tuple[str, str], Dict[str, int]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    probabilities: Dict[Tuple[str, str], Dict[str, float]] = {}
    for k, v in counts.items():
        probabilities[k] = {}
        total_appearances = sum(v.values())
        for word, count in v.items():
            probabilities[k][word] = count / total_appearances
    return probabilities


def generate_sentence(
    probabilities: Dict[Tuple[str, str], Dict[str, float]],
    start_words: List[str],
) -> str:
    prev = '.'  # start with a fake "end of last sentence"
    current = random.choice(start_words)
    result = [current]
    while True:
        candidates = probabilities[(prev, current)]
        words, weights = list(zip(*candidates.items()))
        next_word = random.choices(words, weights=weights, k=1)[0]
        if next_word == '.':
            return ' '.join(result)
        result.append(next_word)
        prev = current
        current = next_word


if __name__ == '__main__':

    sentences = read_tweets()
    assert [len(s) > 1 for s in sentences]
    document = build_document(sentences)

    counts, start_words = count_grams(document)
    probabilities = counts_to_probabilities(counts)

    print(generate_sentence(probabilities, start_words))


