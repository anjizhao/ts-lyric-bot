
import csv
import glob
from statistics import mean, stdev
import time
from typing import Any, Dict, List, NamedTuple, Optional

import matplotlib.pyplot as plt
from nltk.lm.models import Lidstone, KneserNeyInterpolated
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import tqdm

from code.ngrams import MyNGram
from code.save_utils import LMDef


# LMDef, but with any additional parameters for testing purposes
class LMTestDef(NamedTuple):
    # definition for the LanguageModel to test (incl hyperparameters)
    model_def: LMDef
    # do we want to test how the model performs on its training set
    evaluate_on_training_set: bool = True


test_orders = [2, 3, 4]

lidstone_test_alphas = sorted(set(
    np.concatenate((
        np.linspace(0, .005, 6),
        np.linspace(0, .001, 11),
        np.linspace(.0035, .0045, 6),
    ))
))[1:]  # 0-0.005 excluding 0
lidstone_test_alphas_simple = np.linspace(0, .005, 6)[1:]

kn_test_discounts = sorted(set(
    np.concatenate((
        np.linspace(0, 1, 6),
        np.linspace(0.41, 0.51, 11),
        np.linspace(0.19, 0.29, 11),
    ))
))[1:]
kn_test_discounts_simple = np.linspace(0, 1, 6)[1:]  # 0-1 excluding 0

assert set(lidstone_test_alphas_simple).issubset(lidstone_test_alphas)
assert set(kn_test_discounts_simple).issubset(kn_test_discounts)


lidstone_models = [
    LMTestDef(
        LMDef(
            Lidstone,
            args=[alpha, order],
        ),
        evaluate_on_training_set=(alpha in lidstone_test_alphas_simple),
    )
    for alpha in lidstone_test_alphas for order in test_orders
]

kn_models = [
    LMTestDef(
        LMDef(
            KneserNeyInterpolated,
            args=[order],
            kwargs={'discount': discount},
        ),
        evaluate_on_training_set=(discount in kn_test_discounts_simple),
    )
    for discount in kn_test_discounts for order in test_orders
]


def get_train_set() -> List[str]:
    with open('data/lyrics_dataset_train.txt') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines if line]
    return lines


def get_test_set() -> List[str]:
    with open('data/lyrics_dataset_test.txt') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines if line]
    return lines


def kfold_validation_entropy(
    test_models: List[LMTestDef],
    dataset=List[str],
    n_splits: int = 4,
    progressbars: Optional[str] = None,
) -> List[Dict[str, Any]]:

    results: List[Dict[str, Any]] = []
    dataset = np.array(dataset)

    for model_test_def in tqdm.tqdm(
        test_models,
        desc='test models entropy',
        disable=(progressbars != 'outer'),
    ):
        model_def = model_test_def.model_def
        # print('testing model', model_def)

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
            # print('training fold', index)
            model.train(raw_sentences=train_set)
            # print('calculating entropies', index)
            train_entropy = None
            if model_test_def.evaluate_on_training_set:
                train_entropy = model.test_texts_avg_entropy(
                    train_set, show_progressbar=(progressbars == 'inner'),
                )
                train_entropies.append(train_entropy)
            test_entropy = model.test_texts_avg_entropy(
                test_set, show_progressbar=(progressbars == 'inner'),
            )
            test_entropies.append(test_entropy)

        results.append({
            'model': model_def.class_.__name__,
            'args': model_def.args,
            'kwargs': model_def.kwargs,
            'train_entropy_mean': (
                mean(train_entropies) if train_entropies else None
            ),
            'test_entropy_mean': mean(test_entropies),
            'train_entropy_stdev': (
                stdev(train_entropies) if train_entropies else None
            ),
            'test_entropy_stdev': stdev(test_entropies),
        })

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


def plot_tuning_results(
    df: pd.DataFrame,
    model_name: str = 'model',
    hp: str = 'alpha',  # `hp` is the name of the hyperparameter (e.g. alpha)
):
    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    orders = sorted(set(df.order))
    hps = sorted(set(df[hp]))  # noqa
    for j, metric in enumerate(['train_entropy_mean', 'test_entropy_mean']):
        for i, order in enumerate(orders):
            df[df.order == order].plot(
                hp,
                metric,
                kind='scatter',
                label='order={}'.format(order),
                legend=True,
                ax=axes[j],
                color=cmap(order / 4),
            )
        axes[j].set_title(metric)
        # axes[j].set_xticks(hps)
        axes[j].set_ylabel('entropy')
        # axes[j].legend(loc='lower right')
    plt.suptitle('{} hyperparameters'.format(model_name))
    plt.show()


def plot_tuning_results_compare(
    df1, df2,
    model_name: str = 'model',
    hp: str = 'alpha',
    metric='test_entropy_mean',
):
    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
    orders = sorted(set(df1.order))
    hps = sorted(set(df1[hp]))  # noqa
    for j, df in enumerate([df1, df2]):
        for i, order in enumerate(orders):
            df[df.order == order].plot(
                hp,
                metric,
                kind='scatter',
                label='order={}'.format(order),
                legend=True,
                ax=axes[j],
                color=cmap(order / 4),
            )
        axes[j].grid(True)
        axes[j].set_title('df{}'.format(j))
        # axes[j].set_xticks(hps)
        axes[j].set_ylabel('entropy')
        # axes[j].legend(loc='lower right')
    plt.suptitle('{} hyperparameters'.format(model_name))
    plt.show()


def compile_all_to_df(path: str, index_columns: List[str]):
    dfs = []
    for filename in glob.glob(path):
        df = read_csv_to_df(filename)
        dfs.append(df)
    return pd.concat(
        dfs,
        ignore_index=True,
    ).drop_duplicates(
        subset=index_columns, keep='last',
    ).reset_index(drop=True)


def compile_all_lidstone_to_df(
    path: str = 'data/model_selection_lidstone_*.csv',
) -> pd.DataFrame:
    return compile_all_to_df(path, ['order', 'alpha'])


def compile_all_kn_to_df(
    path: str = 'data/model_selection_kn_*.csv',
) -> pd.DataFrame:
    return compile_all_to_df(path, ['order', 'discount'])


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
    #     lidstone_models, training_corpus, n_splits=4, progressbars='inner',
    # )
    # lidstone_filename = write_lidstone_results_to_file(lidstone_results)

    # lidstone_filename = 'data/model_selection_lidstone_1616176024.csv'
    # lidstone_df = read_csv_to_df(lidstone_filename)
    lidstone_df = compile_all_lidstone_to_df()
    # lidstone_df_old = compile_all_lidstone_to_df(
    #     'data/model_selection_0/model_selection_lidstone_*.csv',
    # )

    # plot_tuning_results(lidstone_df, 'lidstone', 'alpha')
    print(get_min_per_order(lidstone_df))

    # plot_tuning_results_compare(lidstone_df_old, lidstone_df)

    # kn_results = kfold_validation_entropy(
    #     kn_models, training_corpus, n_splits=4, progressbars='inner',
    # )
    # kn_filename = write_kn_results_to_file(kn_results)

    # kn_filename = 'data/model_selection_kn_1616185500.csv'
    # kn_df = read_csv_to_df(kn_filename)
    kn_df = compile_all_kn_to_df()
    # kn_df_old = compile_all_kn_to_df(
    #     'data/model_selection_0/model_selection_kn_*.csv',
    # )
    # plot_tuning_results(kn_df, 'kneser-ney', 'discount')
    print(get_min_per_order(kn_df))

    # plot_tuning_results_compare(
    #     kn_df_old, kn_df, model_name='kneser-ney', hp='discount',
    # )
