import logging
import os
import json
import numpy as np

from wikidata_query.gcn_qa_model import GCN_QA
from wikidata_query.read_data import get_json_data
from wikidata_query.utils import get_words, infer_vector_from_word

_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data')

if not os.path.exists(_saving_dir):
    os.makedirs(_saving_dir)

_logger = logging.getLogger(__name__)
_logging_level = logging.INFO
logging.basicConfig(level=_logging_level, format="%(asctime)s: %(levelname)-1.1s %(name)s] %(message)s")

#_dataset_path_train = os.path.join(_path, '../../dataset/wikidata-disambig-train.json')
#_dataset_path_train = os.path.join(_path, '../../dataset/wikidata-disambig-train.medium.json')
_dataset_path_train = os.path.join(_path, '../../dataset/wikidata-disambig-train.sample.json')
#_dataset_path_val = os.path.join(_path, '../../dataset/wikidata-disambig-dev.json')
_dataset_path_val = os.path.join(_path, '../../dataset/wikidata-disambig-dev.sample.json')


_is_relevant = [.0, 1.]
_is_not_relevant = [1., 0.]


def train(data, model, saving_dir, name_prefix, epochs=20, batch_size=32):
    datasets = []
    for dataset in data:
        dataset_reshaped = {
                'text': [item['text'] for item in dataset],
                'node_vectors': [item['graph']['vectors'] for item in dataset],
                'item_vector': [item['item_vector'] for item in dataset],
                'item_pbg': [item['item_pbg'] for item in dataset],
                'question_vectors': [item['question_vectors'] for item in dataset],
                'question_mask': [item['question_mask'] for item in dataset],
                'y': [item['answer'] for item in dataset]
        }
        datasets.append(dataset_reshaped)

    saving_path = f'{saving_dir}/{name_prefix}'
    model.train(datasets, saving_path, epochs, batch_size)


if __name__ == '__main__':
    # train dataset
    with open(_dataset_path_train, encoding='utf8') as f:
        json_data_train = json.load(f)
    data_train = get_json_data(json_data_train)
    # validation dataset
    with open(_dataset_path_val, encoding='utf8') as f:
        json_data_val = json.load(f)
    data_val = get_json_data(json_data_val)

    data = [data_train, data_val]

    model = GCN_QA(dropout=1.0)
    train(data, model, _saving_dir,
            name_prefix='model-20210518-2',
            epochs=20,
            batch_size=32
    )
