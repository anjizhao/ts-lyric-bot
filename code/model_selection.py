
import csv
from statistics import mean, stdev
import time
from typing import Any, Dict, List, NamedTuple

from nltk.lm.models import (
    LanguageModel, Lidstone, KneserNeyInterpolated,
)
import numpy as np
from sklearn.model_selection import KFold
import tqdm

from code.ngrams import MyNGram


class LMDef(NamedTuple):
    # class & __init__ arguments for each model/hyperparameters we will test
    class_: LanguageModel
    args: List[Any] = []
    kwargs: Dict[str, Any] = dict()


test_orders = [1, 2, 3]
lidstone_test_alphas = [0.001, 0.01, 0.1, 1, 5]
kn_test_discounts = [0.001, 0.01, 0.1, 1]


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


def write_lidstone_results_to_file(results: Dict[str, Any]) -> None:
    filename = 'data/model_selection_lidstone_{}.txt'.format(int(time.time()))
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


def write_kn_results_to_file(results: Dict[str, Any]) -> None:
    filename = 'data/model_selection_kn_{}.txt'.format(int(time.time()))
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



if __name__ == '__main__':
    training_corpus = get_train_set()
    # lidstone_results = kfold_validation(
    #     lidstone_models, training_corpus, n_splits=5,
    # )
    # write_lidstone_results_to_file(lidstone_results)
    kn_results = kfold_validation(
        kn_models, training_corpus, n_splits=5,
    )
    write_kn_results_to_file(kn_results)

