import datetime
import logging
import os
import json
import numpy as np

from wikidata_query.gcn_qa_model import GCN_QA
from wikidata_query.read_data import get_json_data
from wikidata_query.utils import bin_data_into_buckets, get_words, infer_vector_from_word

dirs = {
    'logging' : os.path.join(os.getcwd(), 'log'),
    'models' : os.path.join(os.getcwd(), 'data', 'models'),
    'datasets' : os.path.join(os.getcwd(), '..', 'dataset')
    }

# Create directories
for path in dirs.values():
    if not os.path.exists(path):
        print(f'Create directory {path}')
        os.makedirs(path)

datasets = {
    #'train' : os.path.join(dirs['datasets'], 'wikidata-disambig-train.json'),
    #'train' : os.path.join(dirs['datasets'], 'wikidata-disambig-train.medium.json'),
    'train' : os.path.join(dirs['datasets'], 'wikidata-disambig-train.sample.json'),
    #'val' : os.path.join(dirs['datasets'], 'wikidata-disambig-dev.json')
    'val' : os.path.join(dirs['datasets'], 'wikidata-disambig-dev.sample.json')
    }

_model_name = 'model-20210521-1'
_epochs = 3
_batch_size = 1 # TODO allow 32 by using RaggedTensors?
_dropout = 1.0

_is_relevant = [.0, 1.]
_is_not_relevant = [1., 0.]

# Logging
log_level = logging.INFO
log_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_filename = f'{log_timestamp}-{_model_name}-train'
log_path = os.path.join(dirs['logging'], f'{log_filename}.log')
log_format = "%(asctime)s: %(levelname)-1.1s %(name)s:%(lineno)d] %(message)s"
logging.basicConfig(level=log_level, format=log_format,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
_logger = logging.getLogger()


def train(data, model, saving_dir, name_prefix, epochs=20, batch_size=32):
    datasets = []
    for dataset in data:
        dataset_reshaped = {
                'text': [item['text'] for item in dataset],
                'node_vectors': [item['graph']['vectors'] for item in dataset],
                'item_vector': [item['item_vector'] for item in dataset],
                'node_type': [item['graph']['types'] for item in dataset], # TODO only GCN, GAT
                'A_fw': [item['graph']['A_bw'] for item in dataset], # TODO only GCN, GAT
                'question_vectors': [item['question_vectors'] for item in dataset],
                'question_mask': [item['question_mask'] for item in dataset],
                'y': [item['answer'] for item in dataset]
        }
        datasets.append(dataset_reshaped)

    saving_path = f'{saving_dir}/{name_prefix}'
    model.train(datasets, saving_path, epochs, batch_size)


if __name__ == '__main__':
    # train dataset
    _logger.info("=== Load training dataset ===")
    with open(datasets['train'], encoding='utf8') as f:
        json_data_train = json.load(f)
    data_train = get_json_data(json_data_train, use_bert=False, use_pbg=False) # TODO
    # validation dataset
    _logger.info("=== Load validation dataset ===")
    with open(datasets['val'], encoding='utf8') as f:
        json_data_val = json.load(f)
    data_val = get_json_data(json_data_val, use_bert=False, use_pbg=False) # TODO
    data = [data_train, data_val]

    # train
    model = GCN_QA(dropout=_dropout)
    train(data, model, dirs['models'],
            name_prefix=_model_name,
            epochs=_epochs,
            batch_size=_batch_size
    )
