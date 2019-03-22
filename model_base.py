import tensorflow as tf
import numpy as np
import time
import pickle
import abc

from common import common, VocabType
from config import Config


class ModelBase(abc.ABC):
    num_batches_to_log = 100  # TODO: consider exporting to Config or to method param with default value.

    def __init__(self, config: Config):
        self.config = config
        self.sess = tf.Session()

        self.eval_data_lines = None
        self.eval_queue = None
        self.predict_queue = None

        self.eval_placeholder = None
        self.predict_placeholder = None
        self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op, self.eval_code_vectors = None, None, None, None
        self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op = None, None, None

        if config.TRAIN_PATH:
            with open('{}.dict.c2v'.format(config.TRAIN_PATH), 'rb') as file:
                word_to_count = pickle.load(file)
                path_to_count = pickle.load(file)
                target_to_count = pickle.load(file)
                num_training_examples = pickle.load(file)
                self.config.NUM_EXAMPLES = num_training_examples
                print('Dictionaries loaded.')

        if config.LOAD_PATH:
            self.load_model(sess=None)
        else:
            self.word_to_index, self.index_to_word, self.word_vocab_size = \
                common.load_vocab_from_dict(word_to_count, config.WORDS_VOCAB_SIZE,
                                            start_from=common.SpecialDictWords.index_to_start_dict_from())
            print('Loaded word vocab. size: %d' % self.word_vocab_size)

            self.target_word_to_index, self.index_to_target_word, self.target_word_vocab_size = \
                common.load_vocab_from_dict(target_to_count, config.TARGET_VOCAB_SIZE,
                                            start_from=common.SpecialDictWords.index_to_start_dict_from())
            print('Loaded target word vocab. size: %d' % self.target_word_vocab_size)

            self.path_to_index, self.index_to_path, self.path_vocab_size = \
                common.load_vocab_from_dict(path_to_count, config.PATHS_VOCAB_SIZE,
                                            start_from=common.SpecialDictWords.index_to_start_dict_from())
            print('Loaded paths vocab. size: %d' % self.path_vocab_size)

        self.create_index_to_target_word_map()

    def create_index_to_target_word_map(self):
        self.index_to_target_word_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(list(self.index_to_target_word.keys()),
                                                        list(self.index_to_target_word.values()),
                                                        key_dtype=tf.int64, value_dtype=tf.string),
            default_value=tf.constant(common.SpecialDictWords.NoSuchWord.name, dtype=tf.string))

    def close_session(self):
        self.sess.close()

    @abc.abstractmethod
    def train(self):
        ...

    def trace(self, sum_loss, batch_num, multi_batch_start_time):
        multi_batch_elapsed = time.time() - multi_batch_start_time
        avg_loss = sum_loss / (self.num_batches_to_log * self.config.BATCH_SIZE)
        print('Average loss at batch %d: %f, \tthroughput: %d samples/sec' % (batch_num, avg_loss,
                                                                              self.config.BATCH_SIZE * self.num_batches_to_log / (
                                                                                  multi_batch_elapsed if multi_batch_elapsed > 0 else 1)))

    @abc.abstractmethod
    def evaluate(self):
        ...

    def write_code_vectors(self, file, code_vectors):
        for vec in code_vectors:
            file.write(' '.join(map(str, vec)) + '\n')

    def update_per_subtoken_statistics(self, results, true_positive, false_positive, false_negative):
        for original_name, top_words in results:
            prediction = common.filter_impossible_names(top_words)[0]
            original_subtokens = common.get_subtokens(original_name)
            predicted_subtokens = common.get_subtokens(prediction)
            for subtok in predicted_subtokens:
                if subtok in original_subtokens:
                    true_positive += 1
                else:
                    false_positive += 1
            for subtok in original_subtokens:
                if not subtok in predicted_subtokens:
                    false_negative += 1
        return true_positive, false_positive, false_negative

    @staticmethod
    def calculate_results(true_positive, false_positive, false_negative):
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def trace_evaluation(output_file, correct_predictions, total_predictions, elapsed, total_examples):
        state_message = 'Evaluated %d/%d examples...' % (total_predictions, total_examples)
        throughput_message = "Prediction throughput: %d samples/sec" % int(
            total_predictions / (elapsed if elapsed > 0 else 1))
        print(state_message)
        print(throughput_message)

    def update_correct_predictions(self, num_correct_predictions, output_file, results):
        for original_name, top_words in results:
            normalized_original_name = common.normalize_word(original_name)
            predicted_something = False
            for i, predicted_word in enumerate(common.filter_impossible_names(top_words)):
                if i == 0:
                    output_file.write('Original: ' + original_name + ', predicted 1st: ' + predicted_word + '\n')
                predicted_something = True
                normalized_suggestion = common.normalize_word(predicted_word)
                if normalized_original_name == normalized_suggestion:
                    output_file.write('\t\t predicted correctly at rank: ' + str(i + 1) + '\n')
                    for j in range(i, self.config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION):
                        num_correct_predictions[j] += 1
                    break
            if not predicted_something:
                output_file.write('No results for predicting: ' + original_name)
        return num_correct_predictions

    @abc.abstractmethod
    def predict(self, predict_data_lines):
        # TODO: make `predict()` a base method, and add a new abstract methods for the actual framework-dependant.
        ...

    def get_attention_per_path(self, source_strings, path_strings, target_strings, attention_weights):
        attention_weights = np.squeeze(attention_weights)  # (max_contexts, )
        attention_per_context = {}
        for source, path, target, weight in zip(source_strings, path_strings, target_strings, attention_weights):
            string_triplet = (
                common.binary_to_string(source), common.binary_to_string(path), common.binary_to_string(target))
            attention_per_context[string_triplet] = weight
        return attention_per_context

    @staticmethod
    def get_dictionaries_path(model_file_path):
        dictionaries_save_file_name = "dictionaries.bin"
        return '/'.join(model_file_path.split('/')[:-1] + [dictionaries_save_file_name])

    @abc.abstractmethod
    def save_model(self, sess, path):
        ...

    @abc.abstractmethod
    def load_model(self, sess):
        ...

    @abc.abstractmethod
    def save_word2vec_format(self, dest, source):
        ...

    @staticmethod
    def initialize_session_variables(sess):
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))
        print('Initalized variables')

    def get_should_reuse_variables(self):
        if self.config.TRAIN_PATH:
            return True
        else:
            return None
