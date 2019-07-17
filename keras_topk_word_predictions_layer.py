import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from collections import namedtuple


TopKWordPredictionsLayerResult = namedtuple('TopKWordPredictionsLayerResult', ['words', 'scores'])


class TopKWordPredictionsLayer(Layer):
    def __init__(self,
                 top_k: int,
                 index_to_word_table: tf.lookup.StaticHashTable,
                 **kwargs):
        kwargs['dtype'] = tf.string
        kwargs['trainable'] = False
        super(TopKWordPredictionsLayer, self).__init__(**kwargs)
        self.top_k = top_k
        self.index_to_word_table = index_to_word_table

    def build(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError("Input shape for TopKWordPredictionsLayer should be of >= 2 dimensions.")
        if input_shape[-1] < self.top_k:
            raise ValueError("Last dimension of input shape for TopKWordPredictionsLayer should be of >= `top_k`.")
        super(TopKWordPredictionsLayer, self).build(input_shape)
        self.trainable = False

    def call(self, y_pred, **kwargs) -> TopKWordPredictionsLayerResult:
        top_k_pred_scores, top_k_pred_indices = tf.nn.top_k(y_pred, k=self.top_k, sorted=True)
        top_k_pred_indices = tf.cast(top_k_pred_indices, dtype=self.index_to_word_table.key_dtype)
        top_k_pred_words = self.index_to_word_table.lookup(top_k_pred_indices)

        return TopKWordPredictionsLayerResult(words=top_k_pred_words, scores=top_k_pred_scores)

    def compute_output_shape(self, input_shape):
        output_shape = tuple(input_shape[:-1]) + (self.top_k, )
        return output_shape, output_shape
