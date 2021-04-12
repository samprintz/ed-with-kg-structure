import numpy as np
import os
from gensim.models import KeyedVectors
from wikidata_query.utils import capitalize, low_case


class GloveModel:

    def __init__(self, dir_path, fast_mode):
        self._fast_mode = fast_mode
        self._dir = dir_path
        self._cache_dir = os.path.join(dir_path, '../../data/glove_cache/')
        self._vectors_dict = {}
        self._model = None

        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)


    def __getitem__(self, item):
        return self._vectors_dict[item]


    def __setitem__(self, index, value):
        self._vectors_dict[index] = value


    def __load_vectors(self):
        if self._fast_mode == 0:
            print("Loading GloVe")
            self._glove_dir = '../../data/glove_2.2M.txt'
        elif self._fast_mode == 1 or self._fast_mode == 2:
            print("Loading GloVe (medium)")
            self._glove_dir = '../../data/glove_2.2M.medium.txt'
        else:
            print("Loading GloVe (dummy)")
            self._glove_dir = '../../data/glove_2.2M.dummy.txt'
        self._glove_path = os.path.join(self._dir, self._glove_dir)
        self._model = KeyedVectors.load_word2vec_format(self._glove_path)
        print("GloVe loaded")


    def __get_vector_from_glove(self, word):
        if self._model is None:
            self.__load_vectors()

        try:
            return self._model[word]
        except:
            try:
                return self._model[capitalize(word)]
            except:
                try:
                    return self._model[low_case(word)]
                except:
                    print(f'Could not find embedding for "{word}"')
                    return None


    def __get_vector_from_cache(self, word):
        cache_file = f'{self._cache_dir}{word}.txt'
        if not os.path.exists(cache_file):
            print(f'No embedding found in cache for "{word}"')
            return None
        word_vector = np.loadtxt(cache_file)
        if not word_vector.any(): # zero vector
            print(f'Warning: Read zero vector (300,) from cache for "{word}"')
        else:
            print(f'Read embedding of "{word}" from cache')
        return word_vector


    def __save_vector_to_cache(self, word, word_vector):
        cache_file = f'{self._cache_dir}{word}.txt'
        np.savetxt(cache_file, word_vector)
        print(f'Wrote embedding of "{word}" to cache')


    def __get_vector_from_memory(self, word):
        try:
            word_vector = self[word]
            print(f'Read embedding of "{word}" from memory')
            return word_vector
        except:
            return None


    def __store_vector_in_memory(self, word, word_vector):
        self[word] = word_vector
        print(f'Stored embedding of "{word}" in memory')


    def infer_vector_from_word(self, word):
        word_vector = self.__get_vector_from_memory(word)

        if word_vector is not None:
            return word_vector

        word_vector = self.__get_vector_from_cache(word)

        if word_vector is not None:
            self.__store_vector_in_memory(word, word_vector)
            return word_vector

        word_vector = self.__get_vector_from_glove(word)

        if word_vector is not None:
            self.__store_vector_in_memory(word, word_vector)
            self.__save_vector_to_cache(word, word_vector)
            return word_vector

        word_vector = np.zeros(300)
        self.__store_vector_in_memory(word, word_vector)
        self.__save_vector_to_cache(word, word_vector)
        return word_vector

