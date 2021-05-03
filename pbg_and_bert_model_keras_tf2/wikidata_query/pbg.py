import json
import os
import numpy as np
from tqdm import tqdm

class PBG():
    _dir = '../../data/pbg'
    _cache_dir = '../../data/pbg_cache'
    _url_prefix = 'http://www.wikidata.org/entity/'
    _names = None
    _vectors = None
    _names_dict = {}


    def __init__(self, path, logger, sample_mode=False, use_cache=False):
        self._use_cache = use_cache
        self._dir_path = os.path.join(path, self._dir)
        self._logger = logger

        self.__read_names(sample_mode)
        self.__read_vectors(sample_mode)
        self.__create_names_index()

        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)


    def __read_names(self, sample_mode):
        if sample_mode:
            self._logger.info('Reading PyTorch Big Graph Wikidata item names (sample)...')
            f_names = 'wikidata_translation_v1_names.sample.json'
        else:
            self._logger.info('Reading PyTorch Big Graph Wikidata item names...')
            f_names = 'wikidata_translation_v1_names.json'

        self._path_names = os.path.join(self._dir_path, f_names)
        with open(self._path_names, 'r') as myfile:
            data = myfile.read()
        self._names = json.loads(data)
        self._logger.info(f'Read {len(self._names)} names')


    def __read_vectors(self, sample_mode):
        self._logger.info('Reading PyTorch Big Graph vectors...')
        f_vectors = 'wikidata_translation_v1_vectors.npy'
        self._path_vectors = os.path.join(self._dir_path, f_vectors)
        self._vectors = np.load(self._path_vectors, mmap_mode='r') # mmap_mode for not loading the file into memory
        self._logger.info('Read vectors (mmap)')


    def __create_names_index(self):
        self._logger.info('Creating PyTorch Big Graph names index...')
        i = 0
        for name in tqdm(self._names):
            self._names_dict[name] = i
            i += 1
        self._logger.info('Created index')


    def __get_uri_of_item_id(self, item_id):
        return f'<{self._url_prefix}{item_id}>'


    def __get_index_by_item_id(self, item_id):
        item_uri = self.__get_uri_of_item_id(item_id)
        try:
            return self._names_dict[item_uri]
        except KeyError:
            return None


    def __get_vector_from_pbg(self, item_id):
        item_index = self.__get_index_by_item_id(item_id)
        if item_index is None:
            raise Exception(f'Warning: No embedding found for {item_id}')
        else:
            self._logger.debug(f'Found embedding of {item_id}')
        return self._vectors[item_index]


    def __get_vector_from_cache(self, item_id):
        cache_file = f'{self._cache_dir}/{item_id}.txt'
        if not os.path.exists(cache_file):
            return None
        # TODO Change to np.loadtxt()
        with open(cache_file) as cache:
            line = cache.readline().strip()
        if not line:
            self._logger.warning(f'Corrupted embedding for {item_id} in cache, re-read it')
            return None
        item_vector = np.asarray(line.split())
        self._logger.debug(f'Read embedding of {item_id} from cache')
        return item_vector


    def __save_vector_to_cache(self, item_id, item_vector):
        # TODO Change to np.savetxt()
        with open(f'{self._cache_dir}{item_id}.txt', "w+") as cache:
            item_vector = str(item_vector).replace('\n', ' ').replace('[', '').replace(']', '')
            cache.write(item_vector)


    def get_item_embedding(self, item_id):
        if self._use_cache:
            item_vector = self.__get_vector_from_cache(item_id)
            if item_vector is not None:
                return item_vector
        item_vector = self.__get_vector_from_pbg(item_id)
        if self._use_cache:
            if item_vector is not None:
                self.__save_vector_to_cache(item_id, item_vector)
                self._logger.debug(f'Wrote embedding of {item_id} to cache')
        return item_vector

