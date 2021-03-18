
from collections import Counter
import csv
import glob
import random
from statistics import mean, stdev
import time
from typing import Any, Dict, List, NamedTuple, Tuple
from typing import Counter as CounterType

import matplotlib.pyplot as plt
from nltk.lm.models import (
    LanguageModel, Lidstone, KneserNeyInterpolated,
)
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from code.ngrams import MyNGram


class LMDef(NamedTuple):
    # class & __init__ arguments for each model/hyperparameters we will test
    class_: LanguageModel
    args: List[Any] = []
    kwargs: Dict[str, Any] = dict()


test_orders = [1, 2, 3, 4]
# lidstone_test_alphas = [0.001, 0.01, 0.1, 1, 5]
lidstone_test_alphas = np.linspace(0, 1, 11)[1:]  # 0-1 excluding 0
# lidstone_test_alphas = np.linspace(0, 0.001, 11)[1:]  # 0-0.005 excluding 0

# kn_test_discounts = [0.001, 0.01, 0.1, 1]
# kn_test_discounts = np.linspace(0, 1, 11)[1:]  # 0-1 excluding 0
kn_test_discounts = np.linspace(0.1, 0.6, 11)  # 0.1-0.6 inclusive
# kn_test_discounts = np.linspace(0.2, 0.6, 21)  # 0.2-0.6 excl. 0


lidstone_models = [
    LMDef(Lidstone, [alpha, order])
    for alpha in lidstone_test_alphas for order in test_orders
]

kn_models = [
    LMDef(
        KneserNeyInterpolated,
        args=[order],
        kwargs={'discount': discount},
    )
    for discount in kn_test_discounts for order in test_orders
]


def get_train_set() -> List[str]:
    with open('data/lyrics_dataset_train.txt') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines if line]
    return lines


# we are gona try models with Laplace, Lidstone, and Kneser-Ney
# smoothing bc those are the ones nltk provides :shrug:.



def kfold_validation_entropy(
    test_models: List[LMDef], dataset=List[str], n_splits: int = 4,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    dataset = np.array(dataset)
    for model_def in test_models:
        print('testing model', model_def)
        # want same train/test folds for each model
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)
        train_entropies: List[float] = []
        test_entropies: List[float] = []
        for index, split_indices in enumerate(kf.split(dataset)):
            # need to make a new UNTRAINED model for EACH fold!
            model = MyNGram(
                model_def.class_(*model_def.args, **model_def.kwargs)
            )
            train_indices, test_indices = split_indices
            train_set, test_set = dataset[train_indices], dataset[test_indices]
            print('training', index)
            model.train(raw_sentences=train_set)
            print('calculating entropies', index)
            train_entropy = model.test_texts_avg_entropy(train_set)
            test_entropy = model.test_texts_avg_entropy(test_set)
            train_entropies.append(train_entropy)
            test_entropies.append(test_entropy)
            print(
                'fold {}: avg entropy train {:.4f}, test {:.4f}'.format(
                    index, train_entropy, test_entropy,
                )
            )
        results.append({
            'model': model_def.class_.__name__,
            'args': model_def.args,
            'kwargs': model_def.kwargs,
            'train_entropy_mean': mean(train_entropies),
            'test_entropy_mean': mean(test_entropies),
            'train_entropy_stdev': stdev(train_entropies),
            'test_entropy_stdev': stdev(test_entropies),
        })
        print()
    return results


def write_lidstone_results_to_file(results: List[Dict[str, Any]]) -> str:
    filename = 'data/model_selection_lidstone_{}.csv'.format(int(time.time()))
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                'model', 'order', 'alpha',
                'train_entropy_mean', 'test_entropy_mean',
                'train_entropy_stdev', 'test_entropy_stdev',
            ],
            extrasaction='ignore',
        )
        writer.writeheader()
        for r in results:
            r['alpha'], r['order'] = r.pop('args')
            writer.writerow(r)
    return filename


def write_kn_results_to_file(results: List[Dict[str, Any]]) -> str:
    filename = 'data/model_selection_kn_{}.csv'.format(int(time.time()))
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                'model', 'order', 'discount',
                'train_entropy_mean', 'test_entropy_mean',
                'train_entropy_stdev', 'test_entropy_stdev',
            ],
            extrasaction='ignore',
        )
        writer.writeheader()
        for r in results:
            r['order'] = r['args'][0]
            r['discount'] = r['kwargs'].get('discount')
            writer.writerow(r)
    return filename


def read_csv_to_df(filename: str) -> pd.DataFrame:
    with open(filename, 'r') as csvfile:
        df = pd.read_csv(csvfile)
    return df


def plot_lidstone(df):
    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    orders = sorted(set(df.order))
    alphas = sorted(set(df.alpha))  # noqa
    for j, metric in enumerate(['train_entropy_mean', 'test_entropy_mean']):
        for i, order in enumerate(orders):
            df[df.order == order].plot(
                'alpha',
                metric,
                kind='scatter',
                label='order={}'.format(order),
                legend=True,
                ax=axes[j],
                color=cmap(order / 4),
            )
        axes[j].set_title(metric)
        # axes[j].set_xticks(alphas)
        axes[j].set_ylabel('entropy')
        # axes[j].legend(loc='lower right')
    plt.suptitle('lidstone hyperparameters')
    plt.show()


def plot_kn(df):
    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    orders = sorted(set(df.order))
    discounts = sorted(set(df.discount))  # noqa
    for j, metric in enumerate(['train_entropy_mean', 'test_entropy_mean']):
        for i, order in enumerate(orders):
            df[df.order == order].plot(
                'discount',
                metric,
                kind='scatter',
                label='order={}'.format(order),
                legend=True,
                ax=axes[j],
                color=cmap(order / 4),
            )
        axes[j].set_title(metric)
        # axes[j].set_xticks(discounts)
        axes[j].set_ylabel('entropy')
        # axes[j].legend(loc='upper left')
    plt.suptitle('kneser-ney hyperparameters')
    plt.show()


def compile_all_lidstone() -> pd.DataFrame:
    dfs = []
    for lidstone_filename in glob.glob('data/model_selection_lidstone_*.csv'):
        df = read_csv_to_df(lidstone_filename)
        dfs.append(df)
    return pd.concat(
        dfs,
        ignore_index=True,
    ).drop_duplicates(
        subset=['order', 'alpha'], keep='last',
    ).reset_index(drop=True)


def compile_all_kn() -> pd.DataFrame:
    dfs = []
    for kn_filename in glob.glob('data/model_selection_kn_*.csv'):
        df = read_csv_to_df(kn_filename)
        dfs.append(df)
    return pd.concat(
        dfs,
        ignore_index=True,
    ).drop_duplicates(
        subset=['order', 'discount'], keep='last',
    ).reset_index(drop=True)


def get_min_per_order(
    input_df: pd.DataFrame,
    column: str = 'test_entropy_mean',
) -> pd.DataFrame:
    min_indices = []
    for group, df in input_df.groupby('order'):
        min_index = df[column].idxmin()
        min_indices.append(min_index)
    return input_df.iloc[min_indices]



if __name__ == '__main__':
    training_corpus = get_train_set()

    # lidstone_results = kfold_validation_entropy(
    #     lidstone_models, training_corpus, n_splits=4,
    # )
    # lidstone_filename = write_lidstone_results_to_file(lidstone_results)

    # # lidstone_filename = 'data/model_selection_lidstone_1616008852.csv'
    # # # lidstone_df = read_csv_to_df(lidstone_filename)
    # lidstone_df = compile_all_lidstone()
    # plot_lidstone(lidstone_df)
    # print(get_min_per_order(lidstone_df))

    # # kn_results = kfold_validation_entropy(
    # #     kn_models, training_corpus, n_splits=4,
    # # )
    # # kn_filename = write_kn_results_to_file(kn_results)

    # # kn_filename = 'data/model_selection_kn_1616024156.csv'
    # # kn_df = read_csv_to_df(kn_filename)
    # kn_df = compile_all_kn()
    # plot_kn(kn_df)
    # print(get_min_per_order(kn_df))


'''

$ python code/model_selection.py
(params minimizing test_entropy_mean for each order)

        model      order     alpha  train_entropy_mean  test_entropy_mean   ...
78   Lidstone          1    1.0000            8.423954           8.511394   ...
112  Lidstone          2    0.0040            4.805368           6.489021   ...
34   Lidstone          3    0.0006            2.480749           5.965648   ...
23   Lidstone          4    0.0002            1.557747           5.899271   ...

        model      order  discount  train_entropy_mean  test_entropy_mean   ...
41  KneserNeyInt.      1      0.20           11.402517          11.402517   ...
48  KneserNeyInt.      2      0.50            4.659039           6.263645   ...
99  KneserNeyInt.      3      0.42            2.383805           5.761089   ...
79  KneserNeyInt.      4      0.28            1.523350           5.751098   ...

'''

# test for... generating sensible sentences? we are gonna just not use the
# unigram models

best_model_defs = [
    LMDef(Lidstone, [0.004, 2]),
    LMDef(Lidstone, [0.006, 3]),
    LMDef(Lidstone, [0.002, 4]),
    LMDef(KneserNeyInterpolated, [2], {'discount': 0.50}),
    LMDef(KneserNeyInterpolated, [3], {'discount': 0.42}),
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
    test_models: List[LMDef], dataset: List[str], n_splits: int = 4,
) -> None:
    dataset = np.array(dataset)  # type: ignore
    model_map: Dict[Tuple[int, int], MyNGram] = {
        (model_id, fold_id): MyNGram(
            model_def.class_(*model_def.args, **model_def.kwargs)
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
    model_keys = list(model_map.keys())
    while True:
        key = random.choice(model_keys)
        model = model_map[key]
        print(' '.join(model.generate(20)))
        score = input('rate this sentence (-2 to 3): ')
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
    scores = [c for c in df.columns if type(c) == int]
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
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
