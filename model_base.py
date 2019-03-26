import tensorflow as tf
import numpy as np
import time
import pickle
import abc
import os

from common import common, VocabType, SpecialVocabWords, Vocab
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
                token_to_count = pickle.load(file)
                path_to_count = pickle.load(file)
                target_to_count = pickle.load(file)
            print('Word frequencies dictionaries loaded. Now creating vocabularies.')
            self.token_vocab = Vocab.create_from_freq_dict(
                token_to_count, self.config.MAX_TOKEN_VOCAB_SIZE,
                special_words=[SpecialVocabWords.PAD, SpecialVocabWords.OOV])
            print('Created token vocab. size: %d' % self.token_vocab.size)
            self.path_vocab = Vocab.create_from_freq_dict(
                path_to_count, self.config.MAX_PATH_VOCAB_SIZE,
                special_words=[SpecialVocabWords.PAD, SpecialVocabWords.OOV])
            print('Created path vocab. size: %d' % self.path_vocab.size)
            self.target_vocab = Vocab.create_from_freq_dict(
                target_to_count, self.config.MAX_TARGET_VOCAB_SIZE,
                special_words=[SpecialVocabWords.OOV])
            print('Created target vocab. size: %d' % self.target_vocab.size)
        else:
            vocabularies_path = self._get_vocabularies_path(self.config.MODEL_LOAD_PATH)
            with open(vocabularies_path, 'rb') as file:
                print('Loading model vocabularies from: %s ... ' % vocabularies_path, end='')
                self.token_vocab = pickle.load(file)
                self.path_vocab = pickle.load(file)
                self.target_vocab = pickle.load(file)
                print('Done')

    def _save_vocab_dict(self, path):
        with open(self._get_vocabularies_path(path), 'wb') as file:
            pickle.dump(self.token_vocab, file)
            pickle.dump(self.path_vocab, file)
            pickle.dump(self.target_vocab, file)

    def _create_index_to_target_word_map(self):
        self.index_to_target_word_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(list(self.target_vocab.index_to_word.keys()),
                                                        list(self.target_vocab.index_to_word.values()),
                                                        key_dtype=tf.int64, value_dtype=tf.string),
            default_value=tf.constant(SpecialVocabWords.OOV, dtype=tf.string))

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
