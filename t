[1mdiff --git a/attentive_gcn_model_keras_tf2/wikidata_query/gcn_qa_model.py b/attentive_gcn_model_keras_tf2/wikidata_query/gcn_qa_model.py[m
[1mindex ab57c02..8c8bb8b 100644[m
[1m--- a/attentive_gcn_model_keras_tf2/wikidata_query/gcn_qa_model.py[m
[1m+++ b/attentive_gcn_model_keras_tf2/wikidata_query/gcn_qa_model.py[m
[36m@@ -13,8 +13,6 @@[m [mgpu_devices = tf.config.experimental.list_physical_devices('GPU')[m
 for gpu in gpu_devices:[m
     tf.config.experimental.set_memory_growth(gpu, True)[m
 [m
[31m-_logger = logging.getLogger()[m
[31m-[m
 [m
 class GCN_QA(object):[m
     _max_text_length = 512[m
[36m@@ -41,6 +39,15 @@[m [mclass GCN_QA(object):[m
 [m
     def __init__(self, dropout=1.0):[m
         tf.compat.v1.reset_default_graph()[m
[32m+[m[32m        self._config = config[m
[32m+[m
[32m+[m[32m        # Logging[m
[32m+[m[32m        self._logger = logging.getLogger(__name__) # own logger[m
[32m+[m[32m        self._tf_logger = tf.get_logger() # TensorFlow logger[m
[32m+[m[32m        self._tf_logger.handlers = [] # remove the original handler from the TensorFlow logger[m
[32m+[m[32m        logging.basicConfig(level=self._config.log_level, format=self._config.log_format,[m
[32m+[m[32m                handlers=[logging.FileHandler(self._config.log_path), logging.StreamHandler()])[m
[32m+[m
 [m
         # Part I: Question text sequence -> Bi-GRU[m
         fw_gru = GRU(self._memory_dim, return_sequences=True)[m
[36m@@ -197,9 +204,9 @@[m [mclass GCN_QA(object):[m
         dataset_length_val = len(dataset_val['text'])[m
         validation_steps_per_epoch = dataset_length_val // batch_size[m
 [m
[31m-        _logger.info('')[m
[31m-        _logger.info('=== Training settings ===')[m
[31m-        _logger.info(f'epochs={epochs}, batch_size={batch_size}, dataset_length_train={dataset_length_train}, dataset_length_val={dataset_length_val}, steps_per_epoch={steps_per_epoch}')[m
[32m+[m[32m        self._logger.info('')[m
[32m+[m[32m        self._logger.info('=== Training settings ===')[m
[32m+[m[32m        self._logger.info(f'epochs={epochs}, batch_size={batch_size}, dataset_length_train={dataset_length_train}, dataset_length_val={dataset_length_val}, steps_per_epoch={steps_per_epoch}')[m
 [m
         history = self._model.fit([m
                 self.__generate_data(dataset_train, batch_size),[m
[1mdiff --git a/bert_and_rnn_triplets_model_keras_tf2/wikidata_query/gcn_qa_model.py b/bert_and_rnn_triplets_model_keras_tf2/wikidata_query/gcn_qa_model.py[m
[1mindex 4a1856a..74ebeeb 100644[m
[1m--- a/bert_and_rnn_triplets_model_keras_tf2/wikidata_query/gcn_qa_model.py[m
[1m+++ b/bert_and_rnn_triplets_model_keras_tf2/wikidata_query/gcn_qa_model.py[m
[36m@@ -13,8 +13,6 @@[m [mgpu_devices = tf.config.experimental.list_physical_devices('GPU')[m
 for gpu in gpu_devices:[m
     tf.config.experimental.set_memory_growth(gpu, True)[m
 [m
[31m-_logger = logging.getLogger()[m
[31m-[m
 [m
 class GCN_QA(object):[m
     _max_text_length = 512[m
[36m@@ -31,8 +29,16 @@[m [mclass GCN_QA(object):[m
     _memory_dim = 100[m
     _mask_value = -10.0[m
 [m
[31m-    def __init__(self, dropout=1.0):[m
[32m+[m[32m    def __init__(self, config, dropout=1.0):[m
         tf.compat.v1.reset_default_graph()[m
[32m+[m[32m        self._config = config[m
[32m+[m
[32m+[m[32m        # Logging[m
[32m+[m[32m        self._logger = logging.getLogger(__name__) # own logger[m
[32m+[m[32m        self._tf_logger = tf.get_logger() # TensorFlow logger[m
[32m+[m[32m        self._tf_logger.handlers = [] # remove the original handler from the TensorFlow logger[m
[32m+[m[32m        logging.basicConfig(level=self._config.log_level, format=self._config.log_format,[m
[32m+[m[32m                handlers=[logging.FileHandler(self._config.log_path), logging.StreamHandler()])[m
 [m
         # Tokenizer[m
         self._tokenizer = DistilBertTokenizer.from_pretrained(self._distil_bert, do_lower_case=True,[m
[36m@@ -150,9 +156,9 @@[m [mclass GCN_QA(object):[m
         dataset_length_val = len(dataset_val['text'])[m
         validation_steps_per_epoch = dataset_length_val // batch_size[m
 [m
[31m-        _logger.info('')[m
[31m-        _logger.info('=== Training settings ===')[m
[31m-        _logger.info(f'epochs={epochs}, batch_size={batch_size}, dataset_length_train={dataset_length_train}, dataset_length_val={dataset_length_val}, steps_per_epoch={steps_per_epoch}')[m
[32m+[m[32m        self._logger.info('')[m
[32m+[m[32m        self._logger.info('=== Training settings ===')[m
[32m+[m[32m        self._logger.info(f'epochs={epochs}, batch_size={batch_size}, dataset_length_train={dataset_length_train}, dataset_length_val={dataset_length_val}, steps_per_epoch={steps_per_epoch}')[m
 [m
         history = self._model.fit([m
                 self.__generate_data(dataset_train, batch_size),[m
[1mdiff --git a/gcn_only_model_keras_tf2/wikidata_query/gcn_qa_model.py b/gcn_only_model_keras_tf2/wikidata_query/gcn_qa_model.py[m
[1mindex 848ce09..96214e5 100644[m
[1m--- a/gcn_only_model_keras_tf2/wikidata_query/gcn_qa_model.py[m
[1m+++ b/gcn_only_model_keras_tf2/wikidata_query/gcn_qa_model.py[m
[36m@@ -13,8 +13,6 @@[m [mgpu_devices = tf.config.experimental.list_physical_devices('GPU')[m
 for gpu in gpu_devices:[m
     tf.config.experimental.set_memory_growth(gpu, True)[m
 [m
[31m-_logger = logging.getLogger()[m
[31m-[m
 [m
 class GCN_QA(object):[m
     _max_text_length = 512[m
[36m@@ -39,8 +37,17 @@[m [mclass GCN_QA(object):[m
     _stack_dimension = 2[m
     _mask_value = -10.0[m
 [m
[31m-    def __init__(self, dropout=1.0):[m
[32m+[m[32m    def __init__(self, config, dropout=1.0):[m
         tf.compat.v1.reset_default_graph()[m
[32m+[m[32m        self._config = config[m
[32m+[m
[32m+[m[32m        # Logging[m
[32m+[m[32m        self._logger = logging.getLogger(__name__) # own logger[m
[32m+[m[32m        self._tf_logger = tf.get_logger() # TensorFlow logger[m
[32m+[m[32m        self._tf_logger.handlers = [] # remove the original handler from the TensorFlow logger[m
[32m+[m[32m        logging.basicConfig(level=self._config.log_level, format=self._config.log_format,[m
[32m+[m[32m                handlers=[logging.FileHandler(self._config.log_path), logging.StreamHandler()])[m
[32m+[m
 [m
         # Part I: Question text sequence -> Bi-GRU[m
         fw_gru = GRU(self._memory_dim, return_sequences=True)[m
[36m@@ -197,9 +204,9 @@[m [mclass GCN_QA(object):[m
         dataset_length_val = len(dataset_val['text'])[m
         validation_steps_per_epoch = dataset_length_val // batch_size[m
 [m
[31m-        _logger.info('')[m
[31m-        _logger.info('=== Training settings ===')[m
[31m-        _logger.info(f'epochs={epochs}, batch_size={batch_size}, dataset_length_train={dataset_length_train}, dataset_length_val={dataset_length_val}, steps_per_epoch={steps_per_epoch}')[m
[32m+[m[32m        self._logger.info('')[m
[32m+[m[32m        self._logger.info('=== Training settings ===')[m
[32m+[m[32m        self._logger.info(f'epochs={epochs}, batch_size={batch_size}, dataset_length_train={dataset_length_train}, dataset_length_val={dataset_length_val}, steps_per_epoch={steps_per_epoch}')[m
 [m
         history = self._model.fit([m
                 self.__generate_data(dataset_train, batch_size),[m
[1mdiff --git a/pbg_and_bert_model_keras_tf2/wikidata_query/gcn_qa_model.py b/pbg_and_bert_model_keras_tf2/wikidata_query/gcn_qa_model.py[m
[1mindex dedb89a..e7f4665 100644[m
[1m--- a/pbg_and_bert_model_keras_tf2/wikidata_query/gcn_qa_model.py[m
[1m+++ b/pbg_and_bert_model_keras_tf2/wikidata_query/gcn_qa_model.py[m
[36m@@ -1,7 +1,7 @@[m
[32m+[m[32mimport datetime[m
 import logging[m
 import os[m
[31m-os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # INFO messages are not printed[m
[31m-[m
[32m+[m[32mos.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # INFO messages are not printed on import TensorFlow[m
 import random[m
 import sys[m
 import tensorflow as tf[m
[36m@@ -13,8 +13,6 @@[m [mgpu_devices = tf.config.experimental.list_physical_devices('GPU')[m
 for gpu in gpu_devices:[m
     tf.config.experimental.set_memory_growth(gpu, True)[m
 [m
[31m-_logger = logging.getLogger()[m
[31m-[m
 [m
 class GCN_QA(object):[m
     _max_text_length = 512[m
[36m@@ -33,8 +31,16 @@[m [mclass GCN_QA(object):[m
     _memory_dim = 100[m
     _mask_value = -10.0[m
 [m
[31m-    def __init__(self, dropout=1.0):[m
[32m+[m[32m    def __init__(self, config, dropout=1.0):[m
         tf.compat.v1.reset_default_graph()[m
[32m+[m[32m        self._config = config[m
[32m+[m
[32m+[m[32m        # Logging[m
[32m+[m[32m        self._logger = logging.getLogger(__name__) # own logger[m
[32m+[m[32m        self._tf_logger = tf.get_logger() # TensorFlow logger[m
[32m+[m[32m        self._tf_logger.handlers = [] # remove the original handler from the TensorFlow logger[m
[32m+[m[32m        logging.basicConfig(level=self._config.log_level, format=self._config.log_format,[m
[32m+[m[32m                handlers=[logging.FileHandler(self._config.log_path), logging.StreamHandler()])[m
 [m
         # Tokenizer[m
         self._tokenizer = DistilBertTokenizer.from_pretrained(self._distil_bert, do_lower_case=True,[m
[36m@@ -142,6 +148,9 @@[m [mclass GCN_QA(object):[m
         saving_path = saving_dir + "/cp-{epoch:04d}.ckpt"[m
         save_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path,[m
                 save_weights_only=False)[m
[32m+[m[32m        logging_callback = tf.keras.callbacks.LambdaCallback([m
[32m+[m[32m                on_epoch_end = lambda epoch, logs: self._logger.info(f'\nEpoch {epoch + 1}: loss: {logs["loss"]} - accuracy: {logs["accuracy"]} - val_loss: {logs["val_loss"]} - val_accuracy: {logs["val_accuracy"]}')[m
[32m+[m[32m        )[m
 [m
         # train dataset[m
         dataset_train = datasets[0][m
[36m@@ -152,9 +161,9 @@[m [mclass GCN_QA(object):[m
         dataset_length_val = len(dataset_val['text'])[m
         validation_steps_per_epoch = dataset_length_val // batch_size[m
 [m
[31m-        _logger.info('')[m
[31m-        _logger.info('=== Training settings ===')[m
[31m-        _logger.info(f'epochs={epochs}, batch_size={batch_size}, dataset_length_train={dataset_length_train}, dataset_length_val={dataset_length_val}, steps_per_epoch={steps_per_epoch}')[m
[32m+[m[32m        self._logger.info('')[m
[32m+[m[32m        self._logger.info('=== Training settings ===')[m
[32m+[m[32m        self._logger.info(f'epochs={epochs}, batch_size={batch_size}, dataset_length_train={dataset_length_train}, dataset_length_val={dataset_length_val}, steps_per_epoch={steps_per_epoch}')[m
 [m
         history = self._model.fit([m
                 self.__generate_data(dataset_train, batch_size),[m
[36m@@ -162,7 +171,7 @@[m [mclass GCN_QA(object):[m
                 steps_per_epoch=steps_per_epoch,[m
                 validation_data=self.__generate_data(dataset_val, batch_size),[m
                 validation_steps=validation_steps_per_epoch,[m
[31m-                callbacks=[save_model_callback][m
[32m+[m[32m                callbacks=[save_model_callback, logging_callback][m
         )[m
         return history[m
 [m
[36m@@ -198,8 +207,9 @@[m [mclass GCN_QA(object):[m
         saver = self._model.save(filename)[m
 [m
     @classmethod[m
[31m-    def load(self, filename, dropout=1.0):[m
[31m-        model = GCN_QA(dropout)[m
[32m+[m[32m    def load(self, filename, config, dropout=1.0):[m
[32m+[m[32m        model = GCN_QA(config, dropout)[m
         #model._model.load_weights(filename)[m
         model._model = tf.keras.models.load_model(filename)[m
[32m+[m[32m        model._config = config[m
         return model[m
