
from collections import Counter, defaultdict
import json
import time
from typing import Any, Dict, List, NamedTuple, Optional, Union

from nltk.lm.counter import NgramCounter
import nltk.lm.models
from nltk.lm.models import LanguageModel
from nltk.lm.vocabulary import Vocabulary
from nltk.probability import ConditionalFreqDist, FreqDist


Dist = Union[ConditionalFreqDist, FreqDist]


class LMDef(NamedTuple):
    '''
    contains everything we need to instantiate an (untrained)
    nltk LanguageModel.

    usage:

    model_def = LMDef(...)
    nltk_model = model_def.class_(*model_def.args, **model_def.kwargs)
    '''
    class_: LanguageModel
    args: List[Any] = []  # args and kwargs for the nltk LanguageModel
    kwargs: Dict[str, Any] = dict()


def jsonproof_dict(d: Dict) -> Dict:
    '''
    json doesn't accept tuple keys, so map them to something valid.
    ** this doesn't work with lists of dicts (todo: fix it lol)
    '''

    if not d:  # empty dict is :ok:
        return {}

    if (  # normal, un-nested dict is :ok:
        not any(isinstance(k, tuple) for k in d.keys()) and
        not any(isinstance(v, dict) for v in d.values())
    ):
        return d

    safe_dict = {
        'remapped': [
            {'key': k, 'value': jsonproof_dict(v)}
            if isinstance(v, dict)
            else {'key': k, 'value': v}
            for k, v in d.items()
        ]
    }

    return safe_dict


def un_jsonproof_dict(d: Dict) -> Dict:
    ''' this should be the exact inverse of jsonproof_dict '''

    if not d:  # empty dict stays the same
        return {}

    if d.get('remapped'):
        return {
            tuple(i['key'])  # convert "list" keys back to tuples
            if isinstance(i['key'], list)
            else i['key']:
            un_jsonproof_dict(i['value'])
            if isinstance(i['value'], dict)
            else i['value']
            for i in d['remapped']
        }

    else:
        return {
            k: un_jsonproof_dict(v)
            if isinstance(v, dict)
            else v
            for k, v in d.items()
        }


def vocab_to_dict(vocab: Vocabulary) -> Dict[str, int]:
    c = vocab.counts  # this is a Counter object
    return dict(c)


def dict_to_vocab(d: Dict[str, int]) -> Vocabulary:
    c = Counter(d)
    return Vocabulary(counts=c)


def _counts_to_dict(_counts: Dict[int, Dist]) -> Dict[int, Dict]:
    return {
        k: _dist_to_dict(v) for k, v in _counts.items()
    }


def _dist_to_dict(dist: Dist) -> Dict:
    return {
        k: _dist_to_dict(v) if isinstance(v, FreqDist) else v
        for k, v in dist.items()
    }


def dict_to_counts(d: Dict) -> Dict[int, Dist]:
    _counts_dict: Dict[int, Dist] = defaultdict(ConditionalFreqDist)
    for k, v in d.items():  # top level, k is order
        _counts_dict[k] = dict_to_dist(v)
    return _counts_dict


def dict_to_dist(d: Dict) -> Dist:
    if not d:
        return FreqDist({})

    # freqdist is word to count - inherits from Counter!
    # conditionalfreqdist is tuple to freqdist - inherits from defaultdict!

    # are dict keys strings or tuples?
    # str means make a FreqDist, tuple means make another ConditionalFreqDist
    key = next(iter(d.keys()))

    if isinstance(key, str):
        return FreqDist(d)

    if isinstance(key, tuple):
        conddist = ConditionalFreqDist()
        for k, v in d.items():
            conddist[k] = dict_to_dist(v)
        return conddist

    raise Exception('invalid type for key', key, type(key))


def ngram_counter_to_dict(counts: NgramCounter) -> Dict:
    return _counts_to_dict(counts._counts)


def dict_to_ngram_counter(d: Dict) -> NgramCounter:
    ngram_counter = NgramCounter()
    ngram_counter._counts = dict_to_counts(d)
    return ngram_counter


def model_def_to_dict(model_def: LMDef) -> Dict[str, Any]:
    model_def_dict = model_def._asdict()
    # pop 'class_' and convert it to string instead
    class_ = model_def_dict.pop('class_')
    model_def_dict['class_'] = class_.__name__
    return model_def_dict


def dict_to_model_def(model_def_dict: Dict[str, Any]) -> LMDef:
    # convert 'class_' str back to the actual class
    class_name = model_def_dict.pop('class_')
    model_def_dict['class_'] = getattr(nltk.lm.models, class_name)
    return LMDef(**model_def_dict)


def nltk_model_to_dict(nltk_model: LanguageModel, model_def: LMDef) -> Dict:
    vocab_dict = vocab_to_dict(nltk_model.vocab)
    counts_dict = ngram_counter_to_dict(nltk_model.counts)
    model_def_dict = model_def_to_dict(model_def)
    return {
        'vocab': vocab_dict,
        'counts': counts_dict,
        'model_def': model_def_dict,
    }


def dict_to_model(d: Dict[str, Dict]) -> LanguageModel:
    model_def = dict_to_model_def(d['model_def'])
    counts = dict_to_ngram_counter(d['counts'])
    vocab = dict_to_vocab(d['vocab'])
    nltk_model = model_def.class_(
        *model_def.args,
        **model_def.kwargs,
        vocabulary=vocab,  # vocabulary & counts need to be IN the constructor!
        counter=counts,
    )
    return nltk_model


def save_model(
    nltk_model: LanguageModel,
    model_def: LMDef,
    name: str = 'model',
    indent: Optional[int] = 1,
) -> str:
    file_path = 'data/saved_models/{}_{}.json'.format(name, int(time.time()))
    model_dict = nltk_model_to_dict(nltk_model, model_def)
    jsonsafe_dict = jsonproof_dict(model_dict)
    # note: we are using json.dumps() to make a str in memory & then separately
    # write that string to a file. the json.dump() function doesn't use new
    # memory as it dumps data in chunks directly to the file, but it is slower.
    # if memory ever becomes a concern u can switch to dump (no s)
    jsonified_model = json.dumps(jsonsafe_dict, indent=indent)
    with open(file_path, 'w') as jsonfile:
        jsonfile.write(jsonified_model)
    return file_path


def load_model(file_path: str) -> LanguageModel:
    with open(file_path, 'r') as jsonfile:
        text = jsonfile.read()
    jsonsafe_dict = json.loads(text.strip())
    # same note as above on dump vs dumps
    model_dict = un_jsonproof_dict(jsonsafe_dict)
    return dict_to_model(model_dict)
