
from functools import partial
from statistics import mean
from typing import Iterator, List, Optional

from nltk.lm.api import _random_generator, _weighted_choice
from nltk.lm.models import LanguageModel, Laplace
from nltk.lm.preprocessing import flatten, pad_both_ends
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.util import everygrams
import tqdm


detokenizer = TreebankWordDetokenizer()


class MyNGram:
    def __init__(
        self,
        model: LanguageModel = None,  # an instantiated nltk language model
        include_lower_order_grams: bool = False,
        strip_some_punctuation: bool = True,
        start_token: str = '<s>',
        end_token: str = '</s>',
    ) -> None:

        # the actual nltk model
        if model is None:
            self.model = Laplace(1)
        else:
            self.model = model

        self.order = self.model.order

        self.include_lower_order_grams = include_lower_order_grams
        self.strip_some_punctuation = strip_some_punctuation

        self.start_token = start_token
        self.end_token = end_token

        # function to be used to break a list of strs into "grams"
        self.make_grams = partial(
            everygrams,
            min_len=1 if include_lower_order_grams else self.order,
            max_len=self.order,
        )


    def tokenize(self, text: str) -> List[str]:
        if self.strip_some_punctuation:
            return word_tokenize(
                text.lower().replace('"', '').replace(
                    '(', '').replace(')', '').replace('--', '')
            )
        else:
            return word_tokenize(text.lower())


    def detokenize(self, tokens: List[str]) -> str:
        # filter out start/end tokens before using detokenizer
        words = [t for t in tokens if t != self.start_token and t != self.end_token]
        return detokenizer.detokenize(words)


    def padded_tokenize(self, text: str) -> List[str]:
        return pad_both_ends(
            self.tokenize(text),
            self.model.order,
            left_pad_symbol=self.start_token,
            right_pad_symbol=self.end_token,
        )


    def tokenize_sentences(self, sentences: List[str]) -> Iterator[List[str]]:
        ''' returns a new iterator '''
        tokenized_sentences = (
            self.padded_tokenize(s) for s in sentences
        )
        return tokenized_sentences


    def train(
        self,
        raw_sentences: List[str],
        update_vocab: bool = False,
    ) -> None:
        tokenized_sentences = self.tokenize_sentences(raw_sentences)
        training_data = (self.make_grams(s) for s in tokenized_sentences)
        # create a new iterator bc the last one is done!
        tokenized_sentences = self.tokenize_sentences(raw_sentences)
        vocab = list(flatten(tokenized_sentences))
        if update_vocab and self.model.vocab:
            # in the original nltk code, we can call model.fit() multiple times
            # with different sets of training data, but the vocabulary is only
            # set once, on the first call. subsequent calls skip the vocab
            # update because model.vocab already exists. i want to also add new
            # words to the vocab when i fit additional data.
            self.model.vocab.update(vocab)
            # ^ this is the same line of code from the original to set vocab
        self.model.fit(training_data, vocab)


    def _prep_test_text(self, text: str) -> List[str]:
        ''' convert test string into list of grams '''
        return self.make_grams(self.padded_tokenize(text))


    def test_text_entropy(self, text: str) -> float:
        test_grams = self._prep_test_text(text)
        return self.model.entropy(test_grams)


    def test_text_perplexity(self, text: str) -> float:
        test_grams = self._prep_test_text(text)
        return self.model.perplexity(test_grams)


    def test_texts_avg_entropy(
        self, texts: List[str], show_progressbar: bool = True,
    ) -> float:
        ''' returns average entropy across test texts '''
        entropies = []
        for s in tqdm.tqdm(texts, disable=(not show_progressbar)):
            entropies.append(self.test_text_entropy(s))
        return mean(entropies)


    def generate(self, num_words=1, text_seed: Optional[List[str]] = None, random_seed=None) -> List[str]:
        '''
        generate sentence from model. this is mostly copied from the
        nltk.lm.api.LanguageModel.generate method, but modified to include
        start/stop token logic.

        namely, if no text_seed (context) is provided, we assume
        this is the beginning of a sentence, so we create a list of (order - 1)
        sentence start tokens and use that as the context.

        whenever the next word generated is a stop token, we break out of the
        loop and "stop" trying to find new words. this avoids getting the
        `ValueError("Can't choose from empty population")` exception when a
        sentence ends before `num_words` is reached.
        '''

        if text_seed is None:
            text_seed = []
        else:
            text_seed = list(text_seed)

        while len(text_seed) < self.model.order - 1:
            text_seed = [self.start_token] + text_seed

        random_generator = _random_generator(random_seed)
        if num_words == 1:  # This is the base recursion case.
            context = (
                text_seed[-self.model.order + 1:]
                if len(text_seed) >= self.model.order
                else text_seed
            )
            samples = self.model.context_counts(
                self.model.vocab.lookup(context)
            )
            while context and not samples:
                context = context[1:] if len(context) > 1 else []
                samples = self.model.context_counts(
                    self.model.vocab.lookup(context)
                )
            samples = sorted(samples)
            # Sorting samples achieves two things:
            # - reproducible randomness when sampling
            # - turns Mapping into Sequence which `_weighted_choice` expects
            return _weighted_choice(
                samples,
                tuple(self.model.score(w, context) for w in samples),
                random_generator,
            )

        # We build up text one word at a time using the preceding context.
        generated = []
        for _ in range(num_words):
            word = self.generate(
                num_words=1,
                text_seed=text_seed + generated,
                random_seed=random_generator,
            )
            if word == self.end_token:
                break  # end the sentence early if end_token is encountered
            generated.append(word)

        return text_seed + generated

    def generate_sentence(self, max_words: int = 30, text_seed: Optional[str] = None) -> str:
        if text_seed is not None:
            text_seed = text_seed.split(' ')
        generated = self.generate(max_words, text_seed=text_seed)
        return self.detokenize(generated)



if __name__ == '__main__':

    test_text = "There's something 'bout the way\nThe street looks when it's just rained\nThere's a glow off the pavement, you walk me to the car\nAnd you know I wanna ask you to dance right there\nIn the middle of the parking lot, yeah\nOh, yeah\nWe're driving down the road, I wonder if you know\nI'm trying so hard not to get caught up now\nBut you're just so cool, run your hands through your hair\nAbsent-mindedly making me want you"  # noqa

    ngram_a = MyNGram(Laplace(1))
    ngram_b = MyNGram(Laplace(2))
    ngram_c = MyNGram(Laplace(3))

    for ngram in (ngram_a, ngram_b, ngram_c):
        ngram.train(raw_sentences=test_text.strip().split('\n'))

    print(ngram_a.generate(10))
    print(ngram_b.generate(10))
    print(ngram_c.generate(10))
