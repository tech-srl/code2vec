import tensorflow as tf
import numpy as np
import time
import pickle
import abc

from common import common, VocabType, SpecialDictWords
from config import Config


class ModelBase(abc.ABC):
    def __init__(self, config: Config):
        self.config = config
        self.sess = tf.Session()

        if not config.TRAIN_DATA_PATH and not config.MODEL_LOAD_PATH:
            raise ValueError("Must train or load a model.")

        self._init_num_of_examples()
        self._load_or_create_vocab_dict()
        self._create_index_to_target_word_map()
        self._load_or_build_inner_model()

    def _init_num_of_examples(self):
        print('Checking number of examples ... ', end='')
        if self.config.TRAIN_DATA_PATH:
            self.config.NUM_TRAIN_EXAMPLES = common.rawincount(self.config.TRAIN_DATA_PATH + '.train.c2v')
        if self.config.TEST_DATA_PATH:
            self.config.NUM_TEST_EXAMPLES = common.rawincount(self.config.TEST_DATA_PATH)
        print('Done')

    def load_or_build(self):
        self._load_or_create_vocab_dict()
        self._load_or_build_inner_model()

    def save(self, path=None):
        if path is None:
            path = self.config.MODEL_SAVE_PATH
        self._save_vocab_dict(path)
        self._save_inner_model(path)

    def _load_or_create_vocab_dict(self):
        if self.config.TRAIN_DATA_PATH and not self.config.MODEL_LOAD_PATH:
            with open('{}.dict.c2v'.format(self.config.TRAIN_DATA_PATH), 'rb') as file:
                word_to_count = pickle.load(file)
                path_to_count = pickle.load(file)
                target_to_count = pickle.load(file)
                print('Dictionaries loaded.')
            self.word_to_index, self.index_to_word, self.word_vocab_size = \
                common.load_vocab_from_dict(word_to_count, self.config.MAX_WORDS_VOCAB_SIZE,
                                            start_from=SpecialDictWords.index_to_start_dict_from())
            print('Loaded word vocab. size: %d' % self.word_vocab_size)

            self.target_word_to_index, self.index_to_target_word, self.target_word_vocab_size = \
                common.load_vocab_from_dict(target_to_count, self.config.MAX_TARGET_VOCAB_SIZE,
                                            start_from=SpecialDictWords.index_to_start_dict_from())
            print('Loaded target word vocab. size: %d' % self.target_word_vocab_size)

            self.path_to_index, self.index_to_path, self.path_vocab_size = \
                common.load_vocab_from_dict(path_to_count, self.config.MAX_PATHS_VOCAB_SIZE,
                                            start_from=SpecialDictWords.index_to_start_dict_from())
            print('Loaded paths vocab. size: %d' % self.path_vocab_size)
        else:
            dictionaries_path = self._get_dictionaries_path(self.config.MODEL_LOAD_PATH)
            with open(dictionaries_path, 'rb') as file:
                print('Loading model dictionaries from: %s ...' % dictionaries_path, end='')
                self.word_to_index = pickle.load(file)
                self.index_to_word = pickle.load(file)
                self.word_vocab_size = pickle.load(file)

                self.target_word_to_index = pickle.load(file)
                self.index_to_target_word = pickle.load(file)
                self.target_word_vocab_size = pickle.load(file)

                self.path_to_index = pickle.load(file)
                self.index_to_path = pickle.load(file)
                self.path_vocab_size = pickle.load(file)
                print(' Done')

    def _save_vocab_dict(self, path):
        with open(self._get_dictionaries_path(path), 'wb') as file:
            pickle.dump(self.word_to_index, file)
            pickle.dump(self.index_to_word, file)
            pickle.dump(self.word_vocab_size, file)

            pickle.dump(self.target_word_to_index, file)
            pickle.dump(self.index_to_target_word, file)
            pickle.dump(self.target_word_vocab_size, file)

            pickle.dump(self.path_to_index, file)
            pickle.dump(self.index_to_path, file)
            pickle.dump(self.path_vocab_size, file)

    def _create_index_to_target_word_map(self):
        self.index_to_target_word_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(list(self.index_to_target_word.keys()),
                                                        list(self.index_to_target_word.values()),
                                                        key_dtype=tf.int64, value_dtype=tf.string),
            default_value=tf.constant(SpecialDictWords.OOV.name, dtype=tf.string))

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
    def _get_dictionaries_path(model_file_path):
        dictionaries_save_file_name = "dictionaries.bin"
        return '/'.join(model_file_path.split('/')[:-1] + [dictionaries_save_file_name])

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
