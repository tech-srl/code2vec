import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

from path_context_reader import PathContextInputTensors, PathContextReader
import numpy as np
import time
import pickle
from common import common, VocabType
from keras_attention_layer import AttentionLayer
from common import Config
from model_base import ModelBase


class Model(ModelBase):
    def __init__(self, config: Config):
        super(Model, self).__init__(config)
        K.set_session(self.sess)

    def build_keras_model(self) -> tensorflow.keras.Model:
        # Each input sample consists of a bag of x`MAX_CONTEXTS` tuples (source_terminal, path, target_terminal).
        # The valid mask indicates for each context whether it actually exists or it is just a padding.
        source_terminals_input = Input((self.config.MAX_CONTEXTS,))
        paths_input = Input((self.config.MAX_CONTEXTS,))
        target_terminals_input = Input((self.config.MAX_CONTEXTS,))
        valid_mask = Input((self.config.MAX_CONTEXTS,))

        # TODO: consider set embedding initializer or leave it default? [for 2 embeddings below and last dense layer]

        # Input paths are indexes, we embed these here.
        paths_embedded = Embedding(self.path_vocab_size + 1, self.config.EMBEDDINGS_SIZE)(paths_input)

        # Input terminals are indexes, we embed these here.
        terminals_embedding_shared_layer = Embedding(self.word_vocab_size + 1, self.config.EMBEDDINGS_SIZE)
        source_terminals_embedded = terminals_embedding_shared_layer(source_terminals_input)
        target_terminals_embedded = terminals_embedding_shared_layer(target_terminals_input)

        # `Context` is a concatenation of the 2 terminals & path embedding.
        # Each context is a vector of size 3 * EMBEDDINGS_SIZE.
        context_embedded = Concatenate()([source_terminals_embedded, paths_embedded, target_terminals_embedded])
        context_embedded = Dropout(1 - self.config.DROPOUT_KEEP_RATE)(context_embedded)

        # Lets get dense: Apply a dense layer for each context vector (using same weights for all of the context).
        context_after_dense = TimeDistributed(
            Dense(self.config.EMBEDDINGS_SIZE * 3, input_dim=self.config.EMBEDDINGS_SIZE * 3, activation='tanh')
        )(context_embedded)

        # The final code vectors are received by applying attention to the "densed" context vectors.
        code_vectors = AttentionLayer()(context_after_dense, mask=valid_mask)

        # "Decode": Now we use another dense layer to get the target word embedding from each code vector.
        out = Dense(self.target_word_vocab_size + 1, use_bias=False)(code_vectors)

        inputs = [source_terminals_input, paths_input, target_terminals_input, valid_mask]
        model = tensorflow.keras.Model(inputs=inputs, outputs=out)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        return model

    def train(self):
        input_reader = PathContextReader(token_to_index=self.word_to_index,
                                         path_to_index=self.path_to_index,
                                         target_to_index=self.target_word_to_index,
                                         config=self.config)

        model = self.build_keras_model()
        print('Keras model built')

        # TODO: set keras model saver [max_to_keep=self.config.MAX_TO_KEEP]

        # TODO: initialize the evaluation reader (and pass to model.fit() call)

        self.initialize_session_variables(self.sess)
        print('Initalized variables')

        input_reader.reset(self.sess)
        model.fit(
            input_reader.iterator, steps_per_epoch=self.config.steps_per_epoch, epochs=self.config.NUM_EPOCHS)

    def evaluate(self):
        raise NotImplemented()  # TODO: implement!

    def predict(self, predict_data_lines):
        # TODO: make `predict()` a base method, and add a new abstract methods for the actual framework-dependant.
        raise NotImplemented()  # TODO: implement!

    def save_model(self, sess, path):
        raise NotImplemented()  # TODO: implement!

    def load_model(self, sess):
        raise NotImplemented()  # TODO: implement!

    def save_word2vec_format(self, dest, source):
        raise NotImplemented()  # TODO: implement!
