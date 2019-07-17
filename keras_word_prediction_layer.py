import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
import tensorflow.python.keras.backend as K
from typing import Optional, List, Callable
from functools import reduce
from common import common


class WordPredictionLayer(Layer):
    FilterType = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]

    def __init__(self,
                 top_k: int,
                 index_to_word_table: tf.contrib.lookup.HashTable,
                 predicted_words_filters: Optional[List[FilterType]] = None,
                 **kwargs):
        kwargs['dtype'] = tf.string
        kwargs['trainable'] = False
        super(WordPredictionLayer, self).__init__(**kwargs)
        self.top_k = top_k
        self.index_to_word_table = index_to_word_table
        self.predicted_words_filters = predicted_words_filters

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("Input shape for WordPredictionLayer should be of 2 dimension.")
        super(WordPredictionLayer, self).build(input_shape)
        self.trainable = False

    def call(self, y_pred, **kwargs):
        y_pred.shape.assert_has_rank(2)
        top_k_pred_indices = tf.cast(tf.nn.top_k(y_pred, k=self.top_k).indices,
                                     dtype=self.index_to_word_table.key_dtype)
        predicted_target_words_strings = self.index_to_word_table.lookup(top_k_pred_indices)

        # apply given filter
        masks = []
        if self.predicted_words_filters is not None:
            masks = [fltr(top_k_pred_indices, predicted_target_words_strings) for fltr in self.predicted_words_filters]
        if masks:
            # assert all(mask.shape.assert_is_compatible_with(top_k_pred_indices) for mask in masks)
            legal_predicted_target_words_mask = reduce(tf.logical_and, masks)
        else:
            legal_predicted_target_words_mask = tf.cast(tf.ones_like(top_k_pred_indices), dtype=tf.bool)

        # the first legal predicted word is our prediction
        first_legal_predicted_target_word_mask = common.tf_get_first_true(legal_predicted_target_words_mask)
        first_legal_predicted_target_word_idx = tf.where(first_legal_predicted_target_word_mask)
        first_legal_predicted_word_string = tf.gather_nd(predicted_target_words_strings,
                                                         first_legal_predicted_target_word_idx)

        prediction = tf.reshape(first_legal_predicted_word_string, [-1])
        return prediction

    def compute_output_shape(self, input_shape):
        return input_shape[0],  # (batch,)
