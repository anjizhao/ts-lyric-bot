
from typing import Any, List, NamedTuple

from nltk.lm.models import (
    LanguageModel, Lidstone, KneserNeyInterpolated, WittenBellInterpolated,
)
import numpy as np
from sklearn.model_selection import KFold
import tqdm

from code.ngrams import MyNGram


class LMDef(NamedTuple):
    # class & __init__ arguments for each model/hyperparameters we will test
    class_: LanguageModel
    args: List[Any]


test_models = [
    # LMDef(Lidstone, [.01, 1]),
    # LMDef(Lidstone, [.1, 1]),
    LMDef(Lidstone, [1, 1]),
    # LMDef(Lidstone, [.01, 2]),
    # LMDef(Lidstone, [.1, 2]),
    LMDef(Lidstone, [1, 2]),
    # LMDef(Lidstone, [.01, 3]),
    # LMDef(Lidstone, [.1, 3]),
    LMDef(Lidstone, [1, 3]),
    # LMDef(KneserNeyInterpolated, [1]),
    # LMDef(KneserNeyInterpolated, [2]),
    # LMDef(KneserNeyInterpolated, [3]),
    # LMDef(WittenBellInterpolated, [1]),
    # LMDef(WittenBellInterpolated, [2]),
    # LMDef(WittenBellInterpolated, [3]),
]




def get_train_set() -> List[str]:
    with open('data/lyrics_dataset_train.txt') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines if line]
    return lines


# we are gona try models with Laplace, Lidstone, Witten-Bell, and Kneser-Ney
# smoothing bc those are the ones nltk provides :shrug:.



def kfold_validation(
    test_models: List[LMDef], dataset=List[str], n_splits: int = 5,
) -> None:
    dataset = np.array(dataset)
    for model_def in test_models:
        print('testing model', model_def)
        # want same train/test folds for each model
        kf = KFold(n_splits=3, shuffle=True, random_state=2)
        for index, split_indices in enumerate(kf.split(dataset)):
            # need to make a new UNTRAINED model for EACH fold!
            model = MyNGram(model_def.class_(*model_def.args))
            train_indices, test_indices = split_indices
            train_set, test_set = dataset[train_indices], dataset[test_indices]
            print('training', index)
            model.train(raw_sentences=train_set)
            print('calculating entropies', index)
            train_entropy = model.test_texts_avg_entropy(train_set)
            test_entropy = model.test_texts_avg_entropy(test_set)
            print(
                'fold {}: avg entropy train {:.4f}, test {:.4f}'.format(
                    index, train_entropy, test_entropy,
                )
            )
        print()



if __name__ == '__main__':
    training_corpus = get_train_set()
    # my_models = [MyNGram(model) for model in nltk_models]
    # for m in my_models:
    #     sentences = [m.padded_tokenize(s) for s in training_corpus]
    #     m.train(sentences)
    # test_strings = [
    #     "The street looks when it's just rained",
    #     'i am Taylor Alison Swift and i love CATS!',
    #     'asdfasdf jlkdfjk asdf; ',
    # ]
    # for t in test_strings:
    #     print(t)
    #     for m in my_models:
    #         print(m.model, 'entropy: {:.4f}'.format(m.test_text_entropy(t)))

    # for m in my_models:
    #     print(m.model, m.generate(4))

    kfold_validation(test_models, training_corpus, n_splits=3)
