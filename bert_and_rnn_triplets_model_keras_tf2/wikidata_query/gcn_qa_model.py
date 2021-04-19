import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # INFO messages are not printed

import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Bidirectional, Activation, Dropout, Concatenate

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


class GCN_QA(object):
    _nodes_vocab_size = 300 * 3
    _question_vocab_size = 300
    _nodes_vector_size = 150
    _question_vector_size = 150
    _types_size = 3
    _mask_size = 200
    _types_proj_size = 5
    _word_proj_size = 50
    _word_proj_size_for_rnn = 50
    _word_proj_size_for_item = 50
    _internal_proj_size = 250
    _hidden_layer1_size = 250
    _hidden_layer2_size = 250
    _output_size = 2

    _memory_dim = 100
    _stack_dimension = 2

    def __init__(self, dropout=1.0):
        tf.compat.v1.reset_default_graph()

        # Part I: Question text sequence -> Bi-GRU
        fw_gru = GRU(self._memory_dim, return_sequences=True)
        bw_gru = GRU(self._memory_dim, return_sequences=True, go_backwards=True)

        question_inputs = Input(shape=(None, self._question_vocab_size))
        question_mask_inputs = Input(shape=(None, self._mask_size))
        question_outputs = Bidirectional(layer=fw_gru, backward_layer=bw_gru)(question_inputs)
        question_outputs_masked = tf.math.reduce_mean(tf.math.multiply(question_mask_inputs, question_outputs), axis=1)
        question_outputs = Dense(self._question_vector_size)(question_outputs_masked)
        question_outputs = Activation('relu')(question_outputs)

        # Part II: Entity graph node (as text) -> Bi-LSTM
        fw_lstm = LSTM(self._memory_dim)
        bw_lstm = LSTM(self._memory_dim, go_backwards=True)

        nodes_inputs = Input(shape=(None, self._nodes_vocab_size))
        nodes_outputs = Bidirectional(layer=fw_lstm, backward_layer=bw_lstm)(nodes_inputs)
        nodes_outputs = Dense(self._nodes_vector_size)(nodes_outputs)
        nodes_outputs = Activation('relu')(nodes_outputs)

        # Part III: Comparator
        # concatenation size = _nodes_vector_size + _question_vector_size
        concatenated = Concatenate(axis=1)([question_outputs, nodes_outputs])
        mlp_outputs = Dense(self._hidden_layer2_size)(concatenated)
        mlp_outputs = Activation('relu')(mlp_outputs)
        mlp_outputs = Dropout(0.0)(mlp_outputs)
        mlp_outputs = Dense(self._output_size)(mlp_outputs) # 2-dim. output
        mlp_outputs = Activation('softmax')(mlp_outputs)

        # Compile and fit model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        self._model = tf.keras.models.Model(inputs=[question_inputs, question_mask_inputs, nodes_inputs], outputs=mlp_outputs)
        self._model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    def __train(self, node_X, item_vector, question_vectors, question_mask, y, epochs=1, batch_size=1):
        #TODO use custom fit method
        # https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        history = self._model.fit([question_vectors, question_mask, node_X], y, epochs=epochs, batch_size=batch_size)
        return history

    def train(self, data, epochs=20):
        for epoch in range(epochs):
            node_X = [data[i][0] for i in range(len(data))]
            item_vector = [data[i][1] for i in range(len(data))]
            question_vectors = [data[i][2] for i in range(len(data))]
            question_mask = [data[i][3] for i in range(len(data))]
            y = [data[i][4] for i in range(len(data))]

            node_X = np.asarray(node_X)
            item_vector = np.asarray(item_vector)
            question_vectors = np.asarray(question_vectors)
            question_mask = np.asarray(question_mask)
            y = np.asarray(y)

            history = self.__train(node_X, item_vector, question_vectors, question_mask, y)
            loss = history.history['loss'][0]
            return loss

    def __predict(self, node_X, item_vector, question_vectors, question_mask):
        node_X = np.array(node_X)
        question_vectors = np.array(question_vectors)
        question_mask = np.array(question_mask)

        output = self._model.predict([question_vectors, question_mask, node_X])
        return output

    def __standardize_item(self, item):
        if item[0] < item[1]:
            return [0., 1.]
        return [1., 0.]

    def predict(self, node_X, item_vector, question_vectors, question_mask):
        output = self.__predict([node_X], [item_vector], [question_vectors], [question_mask])
        return self.__standardize_item(output[0])

    # Loading and saving functions

    def save(self, filename):
        saver = self._model.save(filename)

    @classmethod
    def load(self, filename, dropout=1.0):
        model = GCN_QA(dropout)
        model._model = tf.keras.models.load_model(filename)
        return model
