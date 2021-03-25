
from collections import Counter
import csv
import glob
import random
from statistics import mean, stdev
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from typing import Counter as CounterType

import matplotlib.pyplot as plt
from nltk.lm.models import LanguageModel, Lidstone, KneserNeyInterpolated
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import tqdm

from code.model_selection import LMDef
from code.ngrams import MyNGram



'''
$ python code/model_selection.py
(params minimizing test_entropy_mean for each order)
-- ** these models REMOVE these punctuation marks ", (, ), -- !!!! ***

        model      order     alpha  train_entropy_mean  test_entropy_mean   ...
87  Lidstone           2    0.0040            4.799502           6.472088
40  Lidstone           3    0.0006            2.487153           5.936380
29  Lidstone           4    0.0002            1.563236           5.851572

        model      order  discount  train_entropy_mean  test_entropy_mean   ...
9   KneserNeyInt.      2      0.50            4.647399           6.241376
46  KneserNeyInt.      3      0.44            2.411580           5.730923
35  KneserNeyInt.      4      0.28            1.529271           5.702818
'''

# test for... generating sensible sentences? we are gonna just not use the
# unigram models


best_model_defs = [
    LMDef(Lidstone, [0.0040, 2]),
    LMDef(Lidstone, [0.0006, 3]),
    LMDef(Lidstone, [0.0002, 4]),
    LMDef(KneserNeyInterpolated, [2], {'discount': 0.50}),
    LMDef(KneserNeyInterpolated, [3], {'discount': 0.44}),
    LMDef(KneserNeyInterpolated, [4], {'discount': 0.28}),
]


score_filename = 'data/sentence_scores/model_{}_fold_{}.txt'


def _valid_score(user_input: str) -> bool:
    try:
        score = int(user_input)
    except ValueError:
        return False
    return -2 <= score <= 3


score_rubric = '''
-2: literally a lyric
-1: u hav overfit
 0: perfect, beautiful
 1: is good enough
 2: bad
 3: garbagé
'''


def kfold_score_generated_sentences(
    test_models: List[LMDef],
    dataset: List[str],
    n_splits: int = 4,
    strip_some_punctuation: bool = True,
) -> None:
    dataset = np.array(dataset)  # type: ignore
    model_map: Dict[Tuple[int, int], MyNGram] = {
        (model_id, fold_id): MyNGram(
            model_def.class_(*model_def.args, **model_def.kwargs),
            strip_some_punctuation=strip_some_punctuation,
        )
        for model_id, model_def in enumerate(test_models)
        for fold_id in range(n_splits)
    }

    # train each model on its appropriate fold of training data

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)

    print('training models...')
    for fold_id, split_indices in enumerate(kf.split(dataset)):
        train_indices, test_indices = split_indices
        train_set = dataset[train_indices]

        for model_id in range(len(test_models)):
            key = (model_id, fold_id)
            model_map[key].train(train_set)

    print(score_rubric)
    print('(type "quit" to quit)\n')
    model_keys = list(model_map.keys())
    while True:
        key = random.choice(model_keys)
        model = model_map[key]
        print(' '.join(model.generate(20)))
        score = input('rate this sentence (-2 to 3): ')
        if score.lower() == 'quit':
            break
        if _valid_score(score):
            with open(score_filename.format(*key), 'a') as outputfile:
                outputfile.write('{}\n'.format(score))
        else:
            print('invalid input')
        print()


# k model instances per "best model", train each one on a k-fold train set.
# so we have a total of 6 * k models (if k=4 that's 24).
# choose 1 model randomly from the 24, have it generate a sentence,
# score the sentence, and save the score somewhere. repeat this 1000000 times.
# then we can find the average score per model per fold. sure would be nice to
# have a database right about now LOL!
# i guess we can just write (append) to a text file .. for a total of 24 files

# still split by folds so we can make sure score trends are consistent across
# different training data?


def get_sentence_scores() -> Dict[str, List[int]]:
    file_scores: Dict[str, List[int]] = {}
    for filename in glob.glob('data/sentence_scores/*'):
        with open(filename, 'r') as f:
            lines = f.readlines()
            scores = [int(line.strip()) for line in lines]
            file_scores[filename] = scores
    return file_scores


def _filename_to_id_tuple(filename: str) -> Tuple[int, int]:
    str_ids = filename.replace(
        'data/sentence_scores/model_', '',
    ).replace('.txt', '').split('_fold_')
    assert len(str_ids) == 2
    assert [i.isdigit() for i in str_ids]
    return tuple(int(i) for i in str_ids)  # type: ignore


def count_file_scores(
    file_scores: Dict[str, List[int]],
) -> Dict[Tuple[int, int], CounterType[int]]:
    model_map: Dict[Tuple[int, int], CounterType[int]] = {}
    for filename, scores in file_scores.items():
        model_map[_filename_to_id_tuple(filename)] = Counter(scores)
    return model_map


def counts_to_df(
    file_score_counts: Dict[Tuple[int, int], CounterType[int]],
) -> pd.DataFrame:
    dicts: List[Dict[str, Any]] = []
    for key, val in file_score_counts.items():
        d = {
            'model_id': key[0],
            'fold_id': key[1],
        }
        d.update(val)  # type: ignore
        dicts.append(d)
    dicts.sort(key=lambda x: (x['model_id'], x['fold_id']))
    return pd.DataFrame(dicts).fillna(0).astype(int)


def plot_pandas():
    file_score_counts = count_file_scores(get_sentence_scores())
    df = counts_to_df(file_score_counts)
    scores = sorted([c for c in df.columns if type(c) == int], reverse=True)
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 7))
    for model_id in sorted(set(df.model_id)):
        ax = axes.flatten()[model_id]
        model_counts_df = df[df.model_id == model_id].reset_index()[scores]
        n_samples = model_counts_df.sum().sum()
        percentized_df = model_counts_df / n_samples * 100
        percentized_df.T.plot.bar(
            stacked=True,
            ax=ax,
            title='model {} (n={})'.format(model_id, n_samples),
            legend=False,
        )
        ax.set_xlabel('score')
        ax.set_ylabel('% of samples')
    plt.show()
