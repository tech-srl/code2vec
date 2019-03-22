import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.metrics import Metric
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.backend import init_ops, math_ops, control_flow_ops, state_ops
from tensorflow.python.ops import string_ops
import abc


class WordsSubtokenMetricBase(Metric):
    def __init__(self, index_to_word_table: tf.contrib.lookup.HashTable, predicted_word_output,
                 subtokens_delimiter: str = ' | ', name=None, dtype=None):
        super(WordsSubtokenMetricBase, self).__init__(name=None, dtype=None)
        self.tp = self.add_weight('true_positives', shape=(), initializer=init_ops.zeros_initializer)
        self.fp = self.add_weight('false_positives', shape=(), initializer=init_ops.zeros_initializer)
        self.fn = self.add_weight('false_negatives', shape=(), initializer=init_ops.zeros_initializer)
        self.index_to_word_table = index_to_word_table
        self.predicted_word_output = predicted_word_output
        self.subtokens_delimiter = subtokens_delimiter

    def update_state(self, true_target_word_index, y_pred, sample_weight=None):
        """Accumulates true positive, false positive and false negative statistics."""
        if sample_weight is not None:
            raise NotImplemented("WordsSubtokenMetricBase with non-None `sample_weight` is not implemented.")

        true_target_word_index = tf.reshape(
            tf.cast(true_target_word_index, dtype=self.index_to_word_table.key_dtype), [-1])
        true_target_word_string = self.index_to_word_table.lookup(true_target_word_index)

        true_target_subwords = string_ops.string_split(true_target_word_string, delimiter=self.subtokens_delimiter)
        prediction_subwords = string_ops.string_split(self.predicted_word_output, delimiter=self.subtokens_delimiter)

        batch_true_positive = math_ops.cast(
            tf.sets.size(tf.sets.intersection(true_target_subwords, prediction_subwords)), dtype=tf.float32)
        batch_false_positive = math_ops.cast(
            tf.sets.size(tf.sets.difference(prediction_subwords, true_target_subwords)), dtype=tf.float32)
        batch_false_negative = math_ops.cast(
            tf.sets.size(tf.sets.difference(true_target_subwords, prediction_subwords)), dtype=tf.float32)

        update_ops = [
            state_ops.assign_add(self.tp, math_ops.reduce_sum(batch_true_positive)),
            state_ops.assign_add(self.fp, math_ops.reduce_sum(batch_false_positive)),
            state_ops.assign_add(self.fn, math_ops.reduce_sum(batch_false_negative))
        ]
        return control_flow_ops.group(update_ops)

    @abc.abstractmethod
    def result(self):
        ...

    def reset_states(self):
        for v in self.variables:
            K.set_value(v, 0)


class WordsSubtokenPrecisionMetric(WordsSubtokenMetricBase):
    def result(self):
        precision = math_ops.div_no_nan(self.tp, self.tp + self.fp)
        return precision


class WordsSubtokenRecallMetric(WordsSubtokenMetricBase):
    def result(self):
        recall = math_ops.div_no_nan(self.tp, self.tp + self.fn)
        return recall


class WordsSubtokenF1Metric(WordsSubtokenMetricBase):
    def result(self):
        recall = math_ops.div_no_nan(self.tp, self.tp + self.fn)
        precision = math_ops.div_no_nan(self.tp, self.tp + self.fp)
        f1 = math_ops.div_no_nan(2 * precision * recall, precision + recall + K.epsilon())
        return f1
