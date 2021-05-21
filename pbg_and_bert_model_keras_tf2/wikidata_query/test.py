import datetime
import logging
import os
import json
import numpy as np

from wikidata_query.gcn_qa_model import GCN_QA
from wikidata_query.read_data import get_json_data
from wikidata_query.utils import get_words, infer_vector_from_word

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
    #'test' : os.path.join(dirs['datasets'], 'wikidata-disambig-test.json')
    'test' : os.path.join(dirs['datasets'], 'wikidata-disambig-test.sample.json')
    }

_model_name = 'model-20210521-1'
_model_dir = f'{dirs["models"]}/{_model_name}'
_epochs = 20

_is_relevant = [.0, 1.]
_is_not_relevant = [1., 0.]

# Logging
log_level = logging.INFO
log_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_filename = f'{log_timestamp}-{_model_name}-test'
log_path = os.path.join(dirs['logging'], f'{log_filename}.log')
log_format = "%(asctime)s: %(levelname)-1.1s %(name)s:%(lineno)d] %(message)s"
logging.basicConfig(level=log_level, format=log_format,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
_logger = logging.getLogger()


def test(data, model):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    count = 0
    count_all = len(data)
    for item in data:
        count += 1
        expected = item['answer']
        text = item['text'].strip()
        mention = item['item']
        wikidata_id = item['wikidata_id']
        node_vectors = item['graph']['vectors']
        item_vector = item['item_vector']
        item_pbg = item['item_pbg']
        question_vectors = item['question_vectors']
        question_mask = item['question_mask']
#        try:
        prediction = model.predict(text, node_vectors, item_vector, item_pbg, question_vectors, question_mask)
        if prediction == expected and expected == _is_relevant:
            true_positives += 1
            _logger.info(f'Item {str(count)}/{str(count_all)}: [TP] Predicted {wikidata_id} consistent for "{mention}" in "{text}"')
        if prediction == expected and expected == _is_not_relevant:
            true_negatives += 1
            _logger.info(f'Item {str(count)}/{str(count_all)}: [TN] Predicted {wikidata_id} not consistent for "{mention}" in "{text}"')
        if prediction != expected and expected == _is_relevant:
            false_negatives += 1
            _logger.info(f'Item {str(count)}/{str(count_all)}: [FN] Predicted {wikidata_id} not consistent for "{mention}" in "{text}"')
        if prediction != expected and expected == _is_not_relevant:
            false_positives += 1
            _logger.info(f'Item {str(count)}/{str(count_all)}: [FP] Predicted {wikidata_id} consistent for "{mention}" in "{text}"')
#        except Exception as e:
#            print('Exception caught during training: ' + str(e))
    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * 1 / (1 / precision + 1 / recall)
        _logger.info(f'Precision {precision}')
        _logger.info(f'Recall {recall}')
        _logger.info(f'F1 {f1}')
    except:
        _logger.warning('Cannot compute precision and recall.')
        _logger.warning(str(e))


if __name__ == '__main__':
    with open(datasets['test']) as f:
        json_data = json.load(f)
    data = get_json_data(json_data, use_bert=True, use_pbg=True)

    # Test
    for epoch in range(1, _epochs + 1):
        _logger.info('')
        _logger.info(f'--------- Epoch {str(epoch)}/{str(_epochs)} ---------')
        model_path = f'{_model_dir}/cp-{epoch:04d}.ckpt'
        model = GCN_QA.load(model_path)
        test(data, model)
