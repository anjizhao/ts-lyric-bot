
from typing import List

from nltk.lm.models import LanguageModel, Lidstone, KneserNeyInterpolated  # noqa

from code.ngrams import MyNGram
from code.save_utils import LMDef, load_model, save_model


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
        'data/lyrics_dataset_raw_1618429296.txt',
    )
    # ts_data_edited = get_all_lyrics_from_file(
    #     'data/lyrics_dataset_raw_edited.txt',
    # )
    ts_data_long = get_all_lyrics_from_file(
        'data/lyrics_dataset_raw_long_1618429296.txt',
    )
    adj_data = get_all_lyrics_from_file(
        'data/lyrics_dataset_raw_adj_long_1617654229.txt',
    )

    model = MyNGram(
        model_def.class_(*model_def.args, **model_def.kwargs),
    )
    model.train(ts_data)
    # model.train(ts_data_edited)
    model.train(ts_data_long)
    model.train(adj_data, update_vocab=True)
    print(model.generate_sentence())

    filename = save_model(model.model, model_def, 'kn3_all_tv')
    print(filename)
    loaded_model = MyNGram(load_model(filename))
    print(loaded_model.generate_sentence())


if __name__ == '__main__':
    main()
