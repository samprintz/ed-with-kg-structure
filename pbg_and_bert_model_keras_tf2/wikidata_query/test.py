import logging
import os
import json
import numpy as np

from wikidata_query.gcn_qa_model import GCN_QA
from wikidata_query.read_data import get_json_data
from wikidata_query.utils import get_words, infer_vector_from_word

_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data')

_logger = logging.getLogger(__name__)
_logging_level = logging.INFO
logging.basicConfig(level=_logging_level, format="%(asctime)s: %(levelname)-1.1s %(name)s] %(message)s")

#_dataset_path = os.path.join(_path, '../../dataset/wikidata-disambig-test.json')
_dataset_path = os.path.join(_path, '../../dataset/wikidata-disambig-test.sample.json')


_is_relevant = [.0, 1.]
_is_not_relevant = [1., 0.]


def test(data, model):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for item in data:
        expected = item['answer']
        text = item['text']
        node_vectors = item['graph']['vectors']
        item_vector = item['item_vector']
        item_pbg = item['item_pbg']
        question_vectors = item['question_vectors']
        question_mask = item['question_mask']
#        try:
        prediction = model.predict(text, node_vectors, item_vector, item_pbg, question_vectors, question_mask)
        if prediction == expected and expected == _is_relevant:
            true_positives += 1
        if prediction == expected and expected == _is_not_relevant:
            true_negatives += 1
        if prediction != expected and expected == _is_relevant:
            false_negatives += 1
        if prediction != expected and expected == _is_not_relevant:
            false_positives += 1
#        except Exception as e:
#            print('Exception caught during training: ' + str(e))
    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * 1 / (1 / precision + 1 / recall)
        print('precision', precision)
        print('recall', recall)
        print('f1', f1)
    except:
        print('Cannot compute precision and recall.')


if __name__ == '__main__':
    with open(_dataset_path) as f:
        json_data = json.load(f)
    data = get_json_data(json_data)
    name_prefix='model-20210518-2'
    model_dir = f'{_saving_dir}/{name_prefix}'
    epochs = 20
    for epoch in range(1, epochs + 1):
        print(f'--------- Epoch {str(epoch)}/{str(epochs)} ---------')
        model_path = f'{model_dir}/cp-{epoch:04d}.ckpt'
        model = GCN_QA.load(model_path)
        test(data, model)
