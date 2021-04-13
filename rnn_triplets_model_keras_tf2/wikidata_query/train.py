import logging
import os
import json
import numpy as np

from wikidata_query.gcn_qa_model import GCN_QA
from wikidata_query.read_data import get_json_data
from wikidata_query.utils import bin_data_into_buckets, get_words, infer_vector_from_word

_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/')
_bucket_size = 10
_minimum_trace = 10
_fast_mode = 3

_logger = logging.getLogger(__name__)
_logging_level = logging.DEBUG
logging.basicConfig(level=_logging_level, format="%(asctime)s: %(levelname)-1.1s %(name)s] %(message)s")

_logger.info(f'Logging level: {_logging_level}')

print(f'Fast mode: {_fast_mode}')

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


def train(data, model, saving_dir, name_prefix, epochs=20, bucket_size=10, trace_every=1):
    import random
    import sys

    buckets = bin_data_into_buckets(data, bucket_size)
    losses = []
    for i in range(epochs):
        random_buckets = sorted(buckets, key=lambda x: random.random())
        bucket_count = 0
        item_count = 0
        item_count_all = sum([len(b) for b in random_buckets])
        print(f'--------- Epoch {str(i)}/{str(epochs)} ---------')
        for bucket in random_buckets:
            bucket_count += 1
            graph_bucket = []
            try:
                for item in bucket:
                    node_vectors = item['graph']['vectors']
                    y = item['answer']
                    item_vector = item['item_vector']
                    question_vectors = item['question_vectors']
                    question_mask = item['question_mask']
                    graph_bucket.append((node_vectors, item_vector, question_vectors, question_mask, y))
                if len(graph_bucket) > 0:
                    loss = model.train(graph_bucket, 1)
                    item_count += len(graph_bucket)
                    losses.append(loss)
                    loss = "{:.5f}".format(loss)
                    avg_loss = "{:.5f}".format(sum(losses)/len(losses))
                    print(f'Item {item_count}/{item_count_all}, bucket {bucket_count}/{len(random_buckets)} (loss={loss}, avg_loss={avg_loss})', end='\r')
            except Exception as e:
                raise e
                #print(f'Exception caught during training: {str(e)}')
        print(f'\nAverage loss of epoch {str(i)}: {str(sum(losses)/len(losses))}')
        if i % trace_every == 0:
            save_filename = saving_dir + name_prefix + '-' + str(i) + '.tf'
            sys.stderr.write('Saving into ' + save_filename + '\n')
            model.save(save_filename)


if __name__ == '__main__':
    with open(_dataset_path, encoding='utf8') as f:
        json_data = json.load(f)
    data = get_json_data(json_data)
    nn_model = GCN_QA(dropout=1.0)
    train(data,
          nn_model,
          _saving_dir,
          name_prefix='model',
          epochs=20,
          bucket_size=10,
          trace_every=1,
          )
