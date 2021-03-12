
#

from functools import partial
from typing import List, Optional


from nltk.lm.models import Laplace
from nltk.lm.preprocessing import (
    flatten, pad_both_ends,  # padded_everygrams, padded_everygram_pipeline,
)
from nltk.tokenize import word_tokenize
# from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.util import everygrams  # , ngrams


text = '''There's something 'bout the way
The street looks when it's just rained
There's a glow off the pavement, you walk me to the car
And you know I wanna ask you to dance right there
In the middle of the parking lot, yeah
Oh, yeah
We're driving down the road, I wonder if you know
I'm trying so hard not to get caught up now
But you're just so cool, run your hands through your hair
Absent-mindedly making me want you'''


class MyNGram:
    def __init__(self, n: int, interpolate: bool = True) -> None:

        # params
        self.n = n
        self.interpolate = interpolate

        # the actual model
        self.model = Laplace(n)

        self.make_grams = partial(
            # function to be used to break a list of strs into "grams"
            everygrams,
            min_len=1 if interpolate else n,  # i think??
            max_len=n,
        )

    def tokenize(self, text: str) -> List[str]:
        return list(word_tokenize(text.lower()))

    def _pad(self, tokens: List[str]) -> List[str]:
        return list(pad_both_ends(tokens, self.n))

    def padded_tokenize(self, text: str) -> List[str]:
        return self._pad(self.tokenize(text))

    def train(self, sentences: List[List[str]]) -> None:
        # input sentences should be tokenized and padded already
        training_data = (self.make_grams(s) for s in sentences)
        vocab = flatten(sentences)
        self.model.fit(training_data, vocab)

    def prep_test_text(self, text: str) -> List[str]:
        ''' convert test string into list of grams '''
        return list(self.make_grams(self.padded_tokenize(text)))

    def test_text_entropy(self, text: str) -> float:
        test_grams = self.prep_test_text(text)
        return self.model.entropy(test_grams)

    def test_text_perplexity(self, text: str) -> float:
        test_grams = self.prep_test_text(text)
        return self.model.perplexity(test_grams)

    def test_text_log_score(self, text: str) -> float:
        test_grams = self.prep_test_text(text)
        entropy = self.model.entropy(test_grams)
        # entropy is negative of the average log likelihood
        return (-1) * entropy * len(test_grams)

    def test_text_likelihood(self, text: str) -> float:
        return 2 ** self.test_text_log_score(text)

    def generate(
        self, num_words: int, context: Optional[List[str]] = None,
    ) -> List[str]:
        if context is None:
            context = self.padded_tokenize('')[:self.n - 1]
        return self.model.generate(num_words, context)


if __name__ == '__main__':
    ngram_a = MyNGram(1, interpolate=False)
    ngram_b = MyNGram(2, interpolate=False)
    ngram_c = MyNGram(2, interpolate=True)
    print('ngram_a (1-grams)')
    print('ngram_b (2-grams)')
    print('ngram_c (1-grams & 2-grams)')
    print('')
    for ngram in (ngram_a, ngram_b, ngram_c):
        sentences = [
            ngram.padded_tokenize(s) for s in text.strip().split('\n')
        ]
        ngram.train(sentences)
    test_strings = [
        "The street looks when it's just rained",
        'Something about the way my street looks?',
        'i am Taylor Alison Swift and i love CATS!',
        'asdfasdf jlkdfjk asdf; ',
        "I wanna stay right here, in this passenger's seat",
    ]
    for test_str in test_strings:
        print('test str:', test_str)
        print('ngram_a log_score: {:.4f}'.format(ngram_a.test_text_log_score(test_str)))  # noqa
        print('ngram_b log_score: {:.4f}'.format(ngram_b.test_text_log_score(test_str)))  # noqa
        print('ngram_c log_score: {:.4f}'.format(ngram_c.test_text_log_score(test_str)))  # noqa
        print('')

    print(ngram_a.generate(5))
    print(ngram_b.generate(5))
    print(ngram_c.generate(5))

