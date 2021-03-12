
from functools import partial
from typing import Callable, List, Tuple


from nltk.lm.models import Laplace, MLE
from nltk.lm.preprocessing import (
    flatten, pad_both_ends, padded_everygrams, padded_everygram_pipeline,
)
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.util import everygrams, ngrams


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

N = 2

model = MLE(N)
model2 = Laplace(N)

sent_tokenize(text)

sentences = text.split('\n')
tokenized_doc = [word_tokenize(s) for s in sentences]
document = [list(pad_both_ends(s, n=N)) for s in tokenized_doc]

train, vocab = padded_everygram_pipeline(2, tokenized_doc)
# is the same as
train = (padded_everygrams(N, s) for s in tokenized_doc)
vocab = list(flatten(pad_both_ends(s, n=N) for s in tokenized_doc))
model.fit(train, vocab)

detokenize = TreebankWordDetokenizer().detokenize

# maximizing the likelihood is the same as minimizing the cross-entropy.
# entropy is negative of average log likelihood
# "perplexity" = 2^entropy


model_uni = Laplace(1)
model_bi = Laplace(2)

