import os
import json
import numpy as np

from wikidata_query.gcn_qa_model import GCN_QA
from wikidata_query.read_data import get_json_data
from wikidata_query.utils import get_words, infer_vector_from_word

_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/')
_bucket_size = 10
_fast_mode = 3

if _fast_mode == 0:
    _dataset_path = os.path.join(_path, '../../dataset/wikidata-disambig-dev.json')
elif _fast_mode == 1:
    _dataset_path = os.path.join(_path, '../../dataset/wikidata-disambig-dev.medium.json')
else:
    _dataset_path = os.path.join(_path, '../../dataset/wikidata-disambig-dev.sample.json')


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


def erase_edges_with_mask(A, mask):
    for i, erase_row in enumerate(mask):
        if erase_row:
            A[i] = 0
    return A


def get_prediction_from_models(A_fw, A_bw, vectors, types, question_vectors, models):
    predictions = {}
    for i, model in enumerate(models):
        prediction = model.predict(A_fw, A_bw, vectors, types, question_vectors)
        for j, item in enumerate(prediction):
            predictions[(j, i)] = item
    prediction_list = []
    for j in range(len(vectors)):
        all_predictions = [predictions[(j, i)] for i in range(len(models))]
        item_dict = {}
        for item in all_predictions:
            try:
                item_dict[tuple(item)] += 1
            except:
                item_dict[tuple(item)] = 1
        final_prediction = list(sorted(item_dict.items(), key=lambda x: -x[1])[0][0])
        prediction_list.append(final_prediction)
    return prediction_list


def test(data, model):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    total = 0

    for item in data:
        text = item['text']
        expected = item['answer']
        node_vectors = item['graph']['vectors']
        item_vector = item['item_vector']
        question_vectors = item['question_vectors']
        question_mask = item['question_mask']

        prediction = model.predict(text, node_vectors, item_vector, question_vectors, question_mask)
        if prediction == expected and expected == _is_relevant:
            true_positives += 1
        if prediction == expected and expected == _is_not_relevant:
            true_negatives += 1
        if prediction != expected and expected == _is_relevant:
            false_negatives += 1
        if prediction != expected and expected == _is_not_relevant:
            false_positives += 1

        total += 1
        print(f'{total}/{len(data)}', end='\r')

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
    nn_models = GCN_QA.load(os.path.join(_path, '../data/qa.tf'))
    test(data, nn_models)
