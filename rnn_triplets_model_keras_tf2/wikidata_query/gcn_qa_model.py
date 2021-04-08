import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Bidirectional, Activation, Dropout, Concatenate

tf.compat.v1.disable_eager_execution()

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

    def __train(self, node_X, item_vector, question_vectors, question_mask, y, epochs, batch_size):
        # Part I: Question text sequence -> Bi-GRU
        fw_gru = GRU(self._memory_dim)
        bw_gru = GRU(self._memory_dim, go_backwards=True)

        question_inputs = Input(shape=(None, self._question_vocab_size))
        question_outputs = Bidirectional(layer=fw_gru, backward_layer=bw_gru)(question_inputs)
        question_outputs_masked = tf.math.reduce_mean(tf.math.multiply(
                question_mask, question_outputs), axis=0)
        question_outputs = Dense(self._question_vector_size)(question_outputs)
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

        self._model = tf.keras.models.Model(inputs=[question_inputs, nodes_inputs], outputs=mlp_outputs)
        self._model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        self._model.fit([question_vectors, node_X], y, epochs=epochs, batch_size=batch_size)

    def train(self, data, epochs=20, batch_size=32):
        node_X_list = [data[i][0] for i in range(len(data))]
        node_X = tf.keras.preprocessing.sequence.pad_sequences(node_X_list, value=0.0)
        item_vector = np.stack(data[:,1])
        question_vectors = np.stack(data[:,2])
        question_mask = np.stack(data[:,3]).astype(np.float32)
        y = np.stack(data[:,4])

        self.__train(node_X, item_vector, question_vectors, question_mask, y, epochs, batch_size)

    def __predict(self, node_X, item_vector, question_vectors, question_mask):
        output = self._model.predict([question_vectors, node_X])
        return output

    def __standardize_item(self, item):
        if item[0] < item[1]:
            return [0., 1.]
        return [1., 0.]

    def predict(self, node_X, item_vector, question_vectors, question_mask):
        question_vectors = np.expand_dims(question_vectors, axis=0)
        node_X = np.expand_dims(node_X, axis=0)

        output = self.__predict(node_X, item_vector, question_vectors, question_mask)

        return self.__standardize_item(output[0])

    # Loading and saving functions

    def save(self, filename):
        self._model.save(filename)

    @classmethod
    def load(self, filename, dropout=1.0):
        model = GCN_QA(dropout)
        model._model = tf.keras.models.load_model(filename)
        return model
