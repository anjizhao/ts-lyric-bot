
from nltk.lm.vocabulary import Vocabulary


def __len__(self):
    """
    a more efficient version of nltk's Vocabulary.__len__ to monkey-patch in.

    the original code was making a TON of function calls to check if each
    word's count was >= the unknown word cutoff value in order to include
    that word in the vocabulary "length" (this happens in the `for _ in self`
    line below, which creates an iterator that checks `if item in self` for
    each item in the word list).

    however, those extra calls are unnecessary when cutoff == 1 (which is the
    default) because if a word exists in the vocab list at all, it already has
    a count >= 1. this reduces the number of function calls by len(word_list).
    """

    # ---- start added code block

    if self.counts and (self._cutoff == 1):
        return len(self.counts) + 1  # +1 is for the self.unk_label

    # ---- end added code block

    return sum(1 for _ in self)


Vocabulary.__len__ = __len__
