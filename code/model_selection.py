
import csv
from statistics import mean, stdev
import time
from typing import Any, Dict, List, NamedTuple

import matplotlib.pyplot as plt
from nltk.lm.models import (
    LanguageModel, Lidstone, KneserNeyInterpolated,
)
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import tqdm

from code.ngrams import MyNGram


class LMDef(NamedTuple):
    # class & __init__ arguments for each model/hyperparameters we will test
    class_: LanguageModel
    args: List[Any] = []
    kwargs: Dict[str, Any] = dict()


test_orders = [1, 2, 3, 4]
# lidstone_test_alphas = [0.001, 0.01, 0.1, 1, 5]
# lidstone_test_alphas = np.linspace(0, 1, 11)[1:]  # 0-1 excluding 0
lidstone_test_alphas = np.linspace(0, 0.005, 11)[1:]  # 0-0.005 excluding 0

# kn_test_discounts = [0.001, 0.01, 0.1, 1]
# kn_test_discounts = np.linspace(0, 1, 11)[1:]  # 0-1 excluding 0
kn_test_discounts = np.linspace(0.1, 0.6, 11)  # 0.1-0.6 inclusive


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



def kfold_validation(
    test_models: List[LMDef], dataset=List[str], n_splits: int = 5,
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
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    orders = sorted(set(df.order))
    alphas = sorted(set(df.alpha))
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
        axes[j].set_xticks(alphas)
        axes[j].set_ylabel('entropy')
        axes[j].legend(loc='lower right')
    plt.suptitle('lidstone hyperparameters')
    plt.show()


def plot_kn(df):
    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    orders = sorted(set(df.order))
    discounts = sorted(set(df.discount))
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
        axes[j].set_xticks(discounts)
        axes[j].set_ylabel('entropy')
        # axes[j].legend(loc='upper left')
    plt.suptitle('kneser-ney hyperparameters')
    plt.show()



if __name__ == '__main__':
    training_corpus = get_train_set()
    # lidstone_results = kfold_validation(
    #     lidstone_models, training_corpus, n_splits=4,
    # )
    # lidstone_filename = write_lidstone_results_to_file(lidstone_results)

    # lidstone_filename = 'data/model_selection_lidstone_1616008852.csv'
    # lidstone_df = read_csv_to_df(lidstone_filename)
    # plot_lidstone(lidstone_df)

    # kn_results = kfold_validation(
    #     kn_models, training_corpus, n_splits=4,
    # )
    # kn_filename = write_kn_results_to_file(kn_results)

    kn_filename = 'data/model_selection_kn_1616015340.csv'
    kn_df = read_csv_to_df(kn_filename)
    plot_kn(kn_df)
