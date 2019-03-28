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
        self.sess = tf.Session()

        if not config.TRAIN_DATA_PATH and not config.MODEL_LOAD_PATH:
            raise ValueError("Must train or load a model.")

        self._init_num_of_examples()
        self.vocabs = Code2VecVocabs.load_or_create(config)
        self.vocabs.target_vocab.get_index_to_word_lookup_table()  # just to initialize it (if not already initialized)
        self._load_or_build_inner_model()

    def _init_num_of_examples(self):
        print('Checking number of examples ... ', end='')
        if self.config.TRAIN_DATA_PATH:
            if os.path.isfile(self.config.TRAIN_DATA_PATH + '.train.c2v.num_examples'):
                with open(self.config.TRAIN_DATA_PATH + '.train.c2v.num_examples', 'r') as file:
                    self.config.NUM_TRAIN_EXAMPLES = int(file.readline())
            else:
                self.config.NUM_TRAIN_EXAMPLES = common.rawincount(self.config.TRAIN_DATA_PATH + '.train.c2v')
                with open(self.config.TRAIN_DATA_PATH + '.train.c2v.num_examples', 'w') as file:
                    file.write(str(self.config.NUM_TRAIN_EXAMPLES))
        if self.config.TEST_DATA_PATH:
            if os.path.isfile(self.config.TEST_DATA_PATH + '.num_examples'):
                with open(self.config.TEST_DATA_PATH + '.num_examples', 'r') as file:
                    self.config.NUM_TEST_EXAMPLES = int(file.readline())
            else:
                self.config.NUM_TEST_EXAMPLES = common.rawincount(self.config.TEST_DATA_PATH)
                with open(self.config.TEST_DATA_PATH + '.num_examples', 'w') as file:
                    file.write(str(self.config.NUM_TEST_EXAMPLES))
        print('Done')

    def load_or_build(self):
        self.vocabs = Code2VecVocabs.load_or_create(self.config)
        self._load_or_build_inner_model()

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
        self.sess.close()

    @abc.abstractmethod
    def train(self):
        ...

    @abc.abstractmethod
    def evaluate(self):
        ...

    @abc.abstractmethod
    def predict(self, predict_data_lines):
        ...

    @staticmethod
    def _get_vocabularies_path(model_file_path):
        vocabularies_save_file_name = "vocabularies.bin"
        return '/'.join(model_file_path.split('/')[:-1] + [vocabularies_save_file_name])

    @abc.abstractmethod
    def _save_inner_model(self, path):
        ...

    @abc.abstractmethod
    def _load_or_build_inner_model(self):
        ...

    @abc.abstractmethod
    def save_word2vec_format(self, dest, source):
        ...

    def initialize_variables(self):
        self.sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))
        print('Initalized variables')
