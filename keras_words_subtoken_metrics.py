import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.metrics import Metric
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.backend import init_ops, math_ops, control_flow_ops, state_ops
from tensorflow.python.ops import string_ops
import abc
from typing import Optional


class WordsSubtokenMetricBase(Metric):
    def __init__(self,
                 index_to_word_table: Optional[tf.contrib.lookup.HashTable] = None,
                 predicted_word_output=None,
                 subtokens_delimiter: str = ' | ', name=None, dtype=None):
        super(WordsSubtokenMetricBase, self).__init__(name=name, dtype=dtype)
        self.tp = self.add_weight('true_positives', shape=(), initializer=init_ops.zeros_initializer)
        self.fp = self.add_weight('false_positives', shape=(), initializer=init_ops.zeros_initializer)
        self.fn = self.add_weight('false_negatives', shape=(), initializer=init_ops.zeros_initializer)
        self.index_to_word_table = index_to_word_table
        self.predicted_word_output = predicted_word_output
        self.subtokens_delimiter = subtokens_delimiter

    def _get_true_target_word_string(self, true_target_word):
        if self.index_to_word_table is None:
            return true_target_word
        true_target_word_index = tf.cast(true_target_word, dtype=self.index_to_word_table.key_dtype)
        return self.index_to_word_table.lookup(true_target_word_index)

    def update_state(self, true_target_word, y_pred, sample_weight=None):
        """Accumulates true positive, false positive and false negative statistics."""
        if sample_weight is not None:
            raise NotImplemented("WordsSubtokenMetricBase with non-None `sample_weight` is not implemented.")

        predicted_word_output = y_pred if self.predicted_word_output is None else self.predicted_word_output

        true_target_word_string = self._get_true_target_word_string(true_target_word)
        true_target_word_string = tf.reshape(true_target_word_string, [-1])

        true_target_subwords = string_ops.string_split(true_target_word_string, delimiter=self.subtokens_delimiter)
        prediction_subwords = string_ops.string_split(predicted_word_output, delimiter=self.subtokens_delimiter)

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
