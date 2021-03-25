# THIS IS A DRAFT / NOTES

import time
from typing import List

from nltk.lm.models import (
    LanguageModel, Lidstone, KneserNeyInterpolated,
)
from nltk.tokenize.treebank import TreebankWordDetokenizer

from code.ngrams import MyNGram
from code.model_selection import LMDef


detokenizer = TreebankWordDetokenizer()

model_def = LMDef(KneserNeyInterpolated, [3], {'discount': 0.42})
# model_def = LMDef(Lidstone, [0.0006, 3])

def get_all_lyrics_from_file(
    filename: str,
) -> List[str]:
    with open(filename, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines if line]
    return lines


def main():
    ts_data = get_all_lyrics_from_file(
        'data/lyrics_dataset_raw_1615587261.txt',
    )
    ts_data_edited = get_all_lyrics_from_file(
        'data/lyrics_dataset_raw_edited.txt',
    )
    adj_data = get_all_lyrics_from_file(
        'data/lyrics_dataset_raw_adjacent_1616638438.txt',
    )

    model = MyNGram(
        model_def.class_(*model_def.args, **model_def.kwargs),
    )
    model.train(
        ts_data + ts_data_edited + adj_data
    )
    generated = model.generate(30)
    print(detokenizer.detokenize(generated))


if __name__ == '__main__':
    main()

