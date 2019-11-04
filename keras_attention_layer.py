import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from typing import Optional


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, inputs_shape):
        inputs_shape = inputs_shape if isinstance(inputs_shape, list) else [inputs_shape]

        if len(inputs_shape) < 1 or len(inputs_shape) > 2:
            raise ValueError("AttentionLayer expect one or two inputs.")

        # The first (and required) input is the actual input to the layer
        input_shape = inputs_shape[0]

        # Expected input shape consists of a triplet: (batch, input_length, input_dim)
        if len(input_shape) != 3:
            raise ValueError("Input shape for AttentionLayer should be of 3 dimension.")

        self.input_length = int(input_shape[1])
        self.input_dim = int(input_shape[2])
        attention_param_shape = (self.input_dim, 1)

        self.attention_param = self.add_weight(
            name='attention_param',
            shape=attention_param_shape,
            initializer='uniform',
            trainable=True,
            dtype=tf.float32)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = inputs if isinstance(inputs, list) else [inputs]

        if len(inputs) < 1 or len(inputs) > 2:
            raise ValueError("AttentionLayer expect one or two inputs.")

        actual_input = inputs[0]
        mask = inputs[1] if len(inputs) > 1 else None
        if mask is not None and not (((len(mask.shape) == 3 and mask.shape[2] == 1) or len(mask.shape) == 2)
                                     and mask.shape[1] == self.input_length):
            raise ValueError("`mask` should be of shape (batch, input_length) or (batch, input_length, 1) "
                             "when calling an AttentionLayer.")

        assert actual_input.shape[-1] == self.attention_param.shape[0]

        # (batch, input_length, input_dim) * (input_dim, 1) ==> (batch, input_length, 1)
        attention_weights = K.dot(actual_input, self.attention_param)

        if mask is not None:
            if len(mask.shape) == 2:
                mask = K.expand_dims(mask, axis=2)  # (batch, input_length, 1)
            mask = K.log(mask)
            attention_weights += mask

        attention_weights = K.softmax(attention_weights, axis=1)  # (batch, input_length, 1)
        result = K.sum(actual_input * attention_weights, axis=1)  # (batch, input_length)  [multiplication uses broadcast]
        return result, attention_weights

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]  # (batch, input_dim)
