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

#_dataset_path = os.path.join(_path, '../../dataset/wikidata-disambig-train.json')
#_dataset_path = os.path.join(_path, '../../dataset/wikidata-disambig-train.medium.json')
_dataset_path = os.path.join(_path, '../../dataset/wikidata-disambig-train.sample.json')


_is_relevant = [.0, 1.]
_is_not_relevant = [1., 0.]


def train(data, model, saving_dir, name_prefix, epochs=20, batch_size=32):
    dataset = {
            'text': [item['text'] for item in data],
            'node_vectors': [item['graph']['vectors'] for item in data],
            'item_vector': [item['item_vector'] for item in data],
            'item_pbg': [item['item_pbg'] for item in data],
            'question_vectors': [item['question_vectors'] for item in data],
            'question_mask': [item['question_mask'] for item in data],
            'y': [item['answer'] for item in data]
    }

    saving_path = f'{saving_dir}/{name_prefix}'
    model.train(dataset, saving_path, epochs, batch_size)


if __name__ == '__main__':
    with open(_dataset_path, encoding='utf8') as f:
        json_data = json.load(f)
    data = get_json_data(json_data)
    model = GCN_QA(dropout=1.0)
    train(data, model, _saving_dir,
            name_prefix='model-20210503-1',
            epochs=20,
            batch_size=32
    )
