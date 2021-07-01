import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # INFO messages are not printed on import TensorFlow

import random
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Bidirectional, Activation, Dropout, Concatenate, BatchNormalization, Masking

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


class GCN_QA(object):
    _max_text_length = 512
    _mask_size = 200
    _item_pbg_vocab_size = 200
    _question_vocab_size = 300
    _nodes_vector_size = 150
    _item_pbg_vector_size = 150
    _question_vector_size = 150
    _hidden_layer1_size = 250
    _hidden_layer2_size = 250
    _output_size = 2
    _memory_dim = 100

    def __init__(self, config, dropout=1.0):
        tf.compat.v1.reset_default_graph()
        self._config = config

        # Logging
        self._logger = logging.getLogger(__name__) # own logger
        self._tf_logger = tf.get_logger() # TensorFlow logger
        self._tf_logger.handlers = [] # remove the original handler from the TensorFlow logger
        logging.basicConfig(level=self._config.log_level, format=self._config.log_format,
                handlers=[logging.FileHandler(self._config.log_path), logging.StreamHandler()])

        # Part I: Question text sequence -> Bi-GRU
        fw_gru = GRU(self._memory_dim, return_sequences=True)
        bw_gru = GRU(self._memory_dim, return_sequences=True, go_backwards=True)

        input_question_vectors = Input(shape=(None, self._question_vocab_size), name='question_vectors')
        input_question_masks = Input(shape=(None, self._mask_size), name='question_mask')
        question_outputs = Bidirectional(layer=fw_gru, backward_layer=bw_gru)(input_question_vectors)
        question_outputs_masked = tf.math.reduce_mean(tf.math.multiply(input_question_masks, question_outputs), axis=1)
        question_outputs = Dense(self._question_vector_size)(question_outputs_masked)
        question_outputs = Activation('relu')(question_outputs)

        # Part II: Entity -> PyTorch Big Graph embedding
        input_item_pbg = Input(shape=(self._item_pbg_vocab_size), name='item_pbg')
        item_pbg_outputs = Dense(self._item_pbg_vector_size)(input_item_pbg)
        item_pbg_outputs = Activation('relu')(item_pbg_outputs)

        # Part III: Comparator
        # concatenation size = _nodes_vector_size + _question_vector_size
        concatenated = Concatenate(axis=1)([question_outputs, item_pbg_outputs])
        mlp_outputs = Dense(self._hidden_layer2_size)(concatenated)
        mlp_outputs = Activation('relu')(mlp_outputs)
        mlp_outputs = Dropout(0.0)(mlp_outputs)
        mlp_outputs = Dense(self._output_size)(mlp_outputs) # 2-dim. output
        mlp_outputs = Activation('softmax')(mlp_outputs)

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        self._model = tf.keras.models.Model(
                inputs=[input_question_vectors, input_question_masks, input_item_pbg],
                outputs=mlp_outputs)
        self._model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        #self._model.summary()

    def __tokenize(self, sentences, tokenizer, max_length):
        input_ids, input_masks = [], []
        for sentence in sentences:
            inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_length,
                    padding='max_length', return_attention_mask=True)
            input_ids.append(inputs['input_ids'])
            input_masks.append(inputs['attention_mask'])
        return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')

    def __generate_data(self, dataset, batch_size):
        # https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator
        i = 0
        while True:
            # get a batch from the shuffled dataset, preprocess it, and give it to the model
            batch = {
                    'question_vectors': [],
                    'question_mask': [],
                    'item_pbg': [],
                    'y': []
                }

            # draw a (ordered) batch from the (shuffled) dataset
            for b in range(batch_size):
                if i == len(dataset['text']): # re-shuffle when processed whole dataset
                    i = 0
                    lists = list(zip(
                            dataset['question_vectors'],
                            dataset['question_mask'],
                            dataset['item_pbg'],
                            dataset['y']))
                    random.shuffle(lists)
                    dataset['question_vectors'], dataset['question_mask'], dataset['item_pbg'], dataset['y'] = zip(*lists)
                    #TODO rather stop iteration?
                    # raise StopIteration
                # add sample
                batch['question_vectors'].append(dataset['question_vectors'][i])
                batch['question_mask'].append(dataset['question_mask'][i])
                batch['item_pbg'].append(dataset['item_pbg'][i])
                batch['y'].append(dataset['y'][i])
                i += 1

            # preprocess batch (array, pad, tokenize)
            X = {}
            #X['question_vectors'] = np.asarray(batch['question_vectors'])
            #X['question_mask'] = np.asarray(batch['question_mask'])
            X['question_vectors'] = tf.keras.preprocessing.sequence.pad_sequences(
                    batch['question_vectors'], maxlen=self._max_text_length, value=0.0)
            X['question_mask'] = tf.keras.preprocessing.sequence.pad_sequences(
                    batch['question_mask'], maxlen=self._max_text_length, value=0.0)
            X['item_pbg'] = np.asarray(batch['item_pbg'])
            y = np.asarray(batch['y'])

            yield X, y

    def __train(self, datasets, saving_dir, epochs=1, batch_size=1):
        saving_path = saving_dir + "/cp-{epoch:04d}.ckpt"
        save_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path,
                save_weights_only=False)
        logging_callback = tf.keras.callbacks.LambdaCallback(
                on_epoch_end = lambda epoch, logs: self._logger.info(f'\nEpoch {epoch + 1}: loss: {logs["loss"]} - accuracy: {logs["accuracy"]} - val_loss: {logs["val_loss"]} - val_accuracy: {logs["val_accuracy"]}')
        )

        # train dataset
        dataset_train = datasets[0]
        dataset_length_train = len(dataset_train['text'])
        steps_per_epoch = dataset_length_train // batch_size
        # validation dataset
        dataset_val = datasets[1]
        dataset_length_val = len(dataset_val['text'])
        validation_steps_per_epoch = dataset_length_val // batch_size

        self._logger.info('')
        self._logger.info('=== Training settings ===')
        self._logger.info(f'epochs={epochs}, batch_size={batch_size}, dataset_length_train={dataset_length_train}, dataset_length_val={dataset_length_val}, steps_per_epoch={steps_per_epoch}')

        history = self._model.fit(
                self.__generate_data(dataset_train, batch_size),
                epochs = epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=self.__generate_data(dataset_val, batch_size),
                validation_steps=validation_steps_per_epoch,
                callbacks=[save_model_callback, logging_callback]
        )
        return history

    def train(self, datasets, saving_dir, epochs=20, batch_size=32):
        self.__train(datasets, saving_dir, epochs, batch_size)

    def __predict(self, item_pbg, question_vectors, question_mask):
        output = self._model.predict([question_vectors, question_mask, item_pbg])
        return output

    def __standardize_item(self, item):
        if item[0] < item[1]:
            return [0., 1.]
        return [1., 0.]

    def predict(self, text, node_X, item_vector, item_pbg, question_vectors, question_mask):
        # in contrast to train(), no generator method is required, as the dev set is small enough to fit into memory also with padding
        item_pbg = np.expand_dims(item_pbg, axis=0)
        question_vectors = np.expand_dims(question_vectors, axis=0)
        question_mask = np.expand_dims(question_mask, axis=0)

        output = self.__predict(item_pbg, question_vectors, question_mask)
        return self.__standardize_item(output[0])

    # Loading and saving functions

    def save(self, filename):
        saver = self._model.save(filename)

    @classmethod
    def load(self, filename, config, dropout=1.0):
        model = GCN_QA(config, dropout)
        #model._model.load_weights(filename)
        model._model = tf.keras.models.load_model(filename)
        model._config = config
        return model
