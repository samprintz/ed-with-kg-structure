import logging
import os
import json
import numpy as np

from wikidata_query.gcn_qa_model import GCN_QA
from wikidata_query.read_data import get_json_data
from wikidata_query.utils import bin_data_into_buckets, get_words, infer_vector_from_word

_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data')
_bucket_size = 10
_minimum_trace = 10
_fast_mode = 3

_logger = logging.getLogger(__name__)
_logging_level = logging.INFO
logging.basicConfig(level=_logging_level, format="%(asctime)s: %(levelname)-1.1s %(name)s] %(message)s")

if _fast_mode == 0:
    _dataset_path = os.path.join(_path, '../../dataset/wikidata-disambig-train.json')
elif _fast_mode == 1:
    _dataset_path = os.path.join(_path, '../../dataset/wikidata-disambig-train.medium.json')
else:
    _dataset_path = os.path.join(_path, '../../dataset/wikidata-disambig-train.sample.json')


def get_answers_and_questions_from_json(filename):
    questions_and_answers = []
    dataset_dicts = json.loads(open(filename).read())
    for item in dataset_dicts:
        questions_and_answers.append({'question': item['qText'], 'answers': item['answers']})
    return questions_and_answers


def find_position_of_best_match(candidate_vectors, answer_vector):
    old_distance = 10000
    position = -1
    for index, candidate in enumerate(candidate_vectors):
        distance = np.linalg.norm(candidate - answer_vector)
        if distance < old_distance:
            position = index
            old_distance = distance
    return position


def get_vector_list_from_sentence(model, sentence):
    words = get_words(sentence)
    vectors = []
    for word in words:
        vectors.append(infer_vector_from_word(model, word))
    return vectors


_is_relevant = [.0, 1.]
_is_not_relevant = [1., 0.]


def train(data, model, saving_dir, name_prefix, epochs=20, batch_size=32):
    dataset = {
            'text': [item['text'] for item in data],
            'node_vectors': [item['graph']['vectors'] for item in data],
            'item_vector': [item['item_vector'] for item in data],
            'question_vectors': [item['question_vectors'] for item in data],
            'question_mask': [item['question_mask'] for item in data],
            'y': [item['answer'] for item in data]
    }

    model.train(dataset, epochs=epochs, batch_size=batch_size)

    save_filename = f'{saving_dir}/{name_prefix}.tf'
    _logger.info(f'Saving into {save_filename}')
    model.save(save_filename)


if __name__ == '__main__':
    with open(_dataset_path, encoding='utf8') as f:
        json_data = json.load(f)
    data = get_json_data(json_data)
    model = GCN_QA(dropout=1.0)
    train(data, model, _saving_dir,
            name_prefix='model-20210420-2',
            epochs=20,
            batch_size=1
    )
