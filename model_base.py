import tensorflow as tf
import numpy as np
import abc
import os

from common import common
from vocabularies import Code2VecVocabs
from config import Config


class ModelBase(abc.ABC):
    def __init__(self, config: Config):
        self.config = config

        if not config.TRAIN_DATA_PATH_PREFIX and not config.MODEL_LOAD_PATH:
            raise ValueError("Must train or load a model.")

        self._init_num_of_examples()
        self.vocabs = Code2VecVocabs.load_or_create(config)
        self.vocabs.target_vocab.get_index_to_word_lookup_table()  # just to initialize it (if not already initialized)
        self._load_or_create_inner_model()

    def _init_num_of_examples(self):
        print('Checking number of examples ... ', end='')
        if self.config.TRAIN_DATA_PATH_PREFIX:
            self.config.NUM_TRAIN_EXAMPLES = self._get_num_of_examples_for_dataset(self.config.train_data_path)
        if self.config.TEST_DATA_PATH:
            self.config.NUM_TEST_EXAMPLES = self._get_num_of_examples_for_dataset(self.config.TEST_DATA_PATH)
        print('Done')

    @staticmethod
    def _get_num_of_examples_for_dataset(dataset_path: str) -> int:
        dataset_num_examples_file_path = dataset_path + '.num_examples'
        if os.path.isfile(dataset_num_examples_file_path):
            with open(dataset_num_examples_file_path, 'r') as file:
                num_examples_in_dataset = int(file.readline())
        else:
            num_examples_in_dataset = common.rawincount(dataset_path)
            with open(dataset_num_examples_file_path, 'w') as file:
                file.write(str(num_examples_in_dataset))
        return num_examples_in_dataset

    def load_or_build(self):
        self.vocabs = Code2VecVocabs.load_or_create(self.config)
        self._load_or_create_inner_model()

    def save(self, model_save_path=None):
        if model_save_path is None:
            model_save_path = self.config.MODEL_SAVE_PATH
        self.vocabs.save(self.config.get_vocabularies_path_from_model_path(model_save_path))
        self._save_inner_model(model_save_path)

    def _write_code_vectors(self, file, code_vectors):
        for vec in code_vectors:
            file.write(' '.join(map(str, vec)) + '\n')

    def get_attention_per_path(self, source_strings, path_strings, target_strings, attention_weights):
        attention_weights = np.squeeze(attention_weights)  # (max_contexts, )
        attention_per_context = {}
        for source, path, target, weight in zip(source_strings, path_strings, target_strings, attention_weights):
            string_triplet = (
                common.binary_to_string(source), common.binary_to_string(path), common.binary_to_string(target))
            attention_per_context[string_triplet] = weight
        return attention_per_context

    def close_session(self):
        # can be overridden by the implementation model class.
        # default implementation just does nothing.
        pass

    @abc.abstractmethod
    def train(self):
        ...

    @abc.abstractmethod
    def evaluate(self):
        ...

    @abc.abstractmethod
    def predict(self, predict_data_lines):
        ...

    @abc.abstractmethod
    def _save_inner_model(self, path):
        ...

    def _load_or_create_inner_model(self):
        if self.config.MODEL_LOAD_PATH:
            self._load_inner_model()
        else:
            self._create_inner_model()

    @abc.abstractmethod
    def _load_inner_model(self):
        ...

    def _create_inner_model(self):
        # can be overridden by the implementation model class.
        # default implementation just does nothing.
        pass

    @abc.abstractmethod
    def save_word2vec_format(self, dest, source):
        ...
