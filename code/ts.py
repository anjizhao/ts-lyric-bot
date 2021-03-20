# THIS IS A DRAFT / NOTES


from typing import List

from nltk.lm.models import (
    LanguageModel, Lidstone, KneserNeyInterpolated,
)

from code.ngrams import MyNGram
from code.model_selection import LMDef


model_def = LMDef(KneserNeyInterpolated, [3], {'discount': 0.42})


def get_all_lyrics() -> List[str]:
    with open('data/lyrics_dataset_raw.txt') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines if line]
    return lines



def main():
    data = get_all_lyrics()
    model = MyNGram(
        model_def.class_(*model_def.args, **model_def.kwargs)
    )
    model.train(data)
    print(model.generate(20))


