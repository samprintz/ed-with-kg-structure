import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Bidirectional, Activation

tf.compat.v1.disable_eager_execution()

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

TINY = 1e-6
ONE = tf.constant(1.)
NAMESPACE = 'gcn_qa'
forbidden_weight = 1.
_weight_for_positive_matches = 1.
_rw = 1e-1


class GCN_QA(object):
    _nodes_vocab_size = 300 * 3
    _question_vocab_size = 300
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

    _memory_dim = 20 # 50 also ok, 100 not: Resource exhausted: OOM when allocating tensor with shape[676,32,100] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
    _stack_dimension = 2

    def __init__(self, dropout=1.0):
        tf.compat.v1.reset_default_graph()

    def __train(self, node_X, item_vector, question_vectors, question_mask, y, epochs):
        # Part I: Question text sequence - Bi-GRU
        #fw_gru_cells = [GRUCell(self._memory_dim) for _ in range(self._stack_dimension)]
        #bw_gru_cells = [GRUCell(self._memory_dim) for _ in range(self._stack_dimension)]
        #fw_gru = StackedRNNCell(fw_gru_cells, state_is_tuple=True)
        #bw_gru = StackedRNNCell(bw_gru_cells, state_is_tuple=True)
        fw_gru = GRU(self._memory_dim)
        bw_gru = GRU(self._memory_dim, go_backwards=True)

        question_inputs = Input(shape=(None, self._question_vocab_size))
        question_outputs = Bidirectional(layer=fw_gru, backward_layer=bw_gru)(question_inputs)
        question_outputs = Dense(2)(question_outputs) # 2-dim. output
        question_outputs = Activation('relu')(question_outputs)

        # Part II: Entity graph node (as text) -> Bi-LSTM
        fw_lstm = LSTM(self._memory_dim)
        bw_lstm = LSTM(self._memory_dim, go_backwards=True)

        nodes_inputs = Input(shape=(None, self._nodes_vocab_size))
        nodes_outputs = Bidirectional(layer=fw_lstm, backward_layer=bw_lstm)(nodes_inputs)
        nodes_outputs = Dense(2)(nodes_outputs) # 2-dim. output
        nodes_outputs = Activation('relu')(nodes_outputs)

        # Part III: Comparator
        # Concatenation of y_text and y_node
        #concatenated = tf.concat([question_outputs, self.nodes_output])
        #input_mlp = Input(shape=(None, self._nodes_vocab_size + self._nodes_vocab_size))


        # Compile and fit model
        self._model = tf.keras.models.Model(inputs=question_inputs, outputs=question_outputs)
        #self._model = tf.keras.models.Model(inputs=nodes_inputs, outputs=nodes_outputs)
        self._model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
        self._model.metrics_names

        self._model.fit(question_vectors, y, epochs=epochs)
        #self._model.fit(node_X, y, epochs=epochs)

    def train(self, data, epochs=20):
        node_X_list = [data[i][0] for i in range(len(data))]
        node_X = tf.keras.preprocessing.sequence.pad_sequences(node_X_list, value=0.0)
        item_vector = np.stack(data[:,1])
        question_vectors = np.stack(data[:,2])
        question_mask = np.stack(data[:,3])
        y = np.stack(data[:,4])

        self.__train(node_X, item_vector, question_vectors, question_mask, y, epochs)

    def __predict(self, node_X, item_vector, question_vectors, question_mask):
        output = self._model.predict(question_vectors)
        return output

    def __standardize_item(self, item):
        if item[0] < item[1]:
            return [0., 1.]
        return [1., 0.]

    def predict(self, node_X, item_vector, question_vectors, question_mask):
        question_vectors = np.stack(question_vectors)
        question_vectors = np.expand_dims(question_vectors, axis=0)

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
