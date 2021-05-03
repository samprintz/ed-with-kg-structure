import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # INFO messages are not printed

import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Bidirectional, Activation, Dropout, Concatenate
from spektral.layers import GCNConv

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


class GCN_QA(object):
    _nodes_vocab_size = 300
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
    _gcn_channels = _hidden_layer1_size
    _gcn_dropout = 0.5 # TODO

    _memory_dim = 100
    _stack_dimension = 2

    def __init__(self, dropout=1.0):
        tf.compat.v1.reset_default_graph()

        # Part I: Question text sequence -> Bi-GRU
        fw_gru = GRU(self._memory_dim, return_sequences=True)
        bw_gru = GRU(self._memory_dim, return_sequences=True, go_backwards=True)

        question_inputs = Input(shape=(None, self._question_vocab_size), name='question')
        question_mask_inputs = Input(shape=(None, self._mask_size), name='question_mask')
        question_outputs = Bidirectional(layer=fw_gru, backward_layer=bw_gru)(question_inputs)
        question_outputs_masked = tf.math.reduce_mean(tf.math.multiply(question_mask_inputs, question_outputs), axis=1)
        question_outputs = Dense(self._question_vector_size)(question_outputs_masked)
        question_outputs = Activation('relu')(question_outputs)

        # Part II: Entity graph node -> GCN
        nodes_inputs = Input(shape=(None, self._nodes_vocab_size), name='node_vectors')
        types_inputs = Input(shape=(None, self._types_size), name='node_type')
        atilde_inputs = Input(shape=(None, None), name='atilde_fw')

        nodes_outputs = Dense(self._word_proj_size)(nodes_inputs)
        nodes_outputs = Activation('relu')(nodes_outputs)

        types_outputs = Dense(self._types_proj_size)(types_inputs)
        types_outputs = Activation('relu')(types_outputs)

        concatenated_for_gcn = Concatenate(axis=2)([nodes_outputs, types_outputs])
        concatenated_for_gcn = Dense(self._internal_proj_size)(concatenated_for_gcn)
        concatenated_for_gcn = Activation('relu')(concatenated_for_gcn)

        gcn_output = GCNConv(self._gcn_channels, activation='relu',
                dropout_rate=self._gcn_dropout)([concatenated_for_gcn, atilde_inputs])
        gcn_output = Dropout(self._gcn_dropout)(gcn_output)
        gcn_output = GCNConv(self._gcn_channels, activation='relu',
                dropout_rate=self._gcn_dropout)([gcn_output, atilde_inputs])
        gcn_output = Dropout(self._gcn_dropout)(gcn_output)
        gcn_output = GCNConv(self._gcn_channels, activation='relu',
                dropout_rate=self._gcn_dropout)([gcn_output, atilde_inputs])
        gcn_output = Dropout(self._gcn_dropout)(gcn_output)
        gcn_output = GCNConv(self._gcn_channels, activation='relu',
                dropout_rate=self._gcn_dropout)([gcn_output, atilde_inputs])
        gcn_output = Dropout(self._gcn_dropout)(gcn_output)

        gcn_output_transposed = tf.transpose(gcn_output, perm=[1, 0, 2]) # to (n_nodes, batch, n_node_features)
        first_node = gcn_output_transposed[0] # use only first node

        # Part III: Comparator
        # concatenation size = _nodes_vector_size + _question_vector_size
        concatenated = Concatenate(axis=1)([question_outputs, first_node])
        mlp_outputs = Dense(self._hidden_layer2_size)(concatenated)
        mlp_outputs = Activation('relu')(mlp_outputs)
        mlp_outputs = Dropout(0.0)(mlp_outputs)
        mlp_outputs = Dense(self._output_size)(mlp_outputs) # 2-dim. output
        mlp_outputs = Activation('softmax')(mlp_outputs)

        # Compile and fit model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        self._model = tf.keras.models.Model(
                inputs=[question_inputs, question_mask_inputs, nodes_inputs, types_inputs, atilde_inputs],
                outputs=mlp_outputs)
        self._model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    def _add_identity(self, A):
        num_nodes = A.shape[0]
        identity = np.identity(num_nodes)
        return identity + A

    def __train(self, Atilde_fw, node_X, types, item_vector, question_vectors, question_mask, y, epochs=1, batch_size=1):
        history = self._model.fit([question_vectors, question_mask, node_X, types, Atilde_fw], y,
                epochs=epochs, batch_size=batch_size)
        return history

    def train(self, data, epochs=20):
        for epoch in range(epochs):
            A_fw = [data[i][0] for i in range(len(data))]
            node_X = [data[i][1] for i in range(len(data))]
            types = [data[i][2] for i in range(len(data))]
            item_vector = [data[i][3] for i in range(len(data))]
            question_vectors = [data[i][4] for i in range(len(data))]
            question_mask = [data[i][5] for i in range(len(data))]
            y = [data[i][6] for i in range(len(data))]

            Atilde_fw = np.asarray([self._add_identity(item) for item in A_fw])
            node_X = np.asarray(node_X)
            types = np.asarray(types)
            item_vector = np.asarray(item_vector)
            question_vectors = np.asarray(question_vectors)
            question_mask = np.asarray(question_mask)
            y = np.asarray(y)

            history = self.__train(Atilde_fw, node_X, types, item_vector, question_vectors, question_mask, y, epochs)
            loss = history.history['loss'][0]
            return loss

    def __predict(self, A_fw, node_X, types, item_vector, question_vectors, question_mask):
        node_X = np.array(node_X)
        Atilde_fw = np.array([self._add_identity(item) for item in A_fw])
        types = np.array(types)
        item_vector = np.asarray(item_vector)
        question_vectors = np.array(question_vectors)
        question_mask = np.array(question_mask)

        output = self._model.predict([question_vectors, question_mask, node_X, types, Atilde_fw])
        return output

    def __standardize_item(self, item):
        if item[0] < item[1]:
            return [0., 1.]
        return [1., 0.]

    def predict(self, A_fw, node_X, types, item_vector, question_vectors, question_mask):
        output = self.__predict([A_fw], [node_X], [types], [item_vector], [question_vectors], [question_mask])
        return self.__standardize_item(output[0])


    # Loading and saving functions

    def save(self, filename):
        saver = self._model.save_weights(filename)
        #saver = self._model.save(filename)

    @classmethod
    def load(self, filename, dropout=1.0):
        model = GCN_QA(dropout)
        model._model.load_weights(filename)
        #model._model = tf.keras.models.load_model(filename)
        return model
