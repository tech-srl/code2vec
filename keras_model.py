import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Input, Embedding, Concatenate, Dropout, TimeDistributed, Dense
import tensorflow.python.keras.backend as K

from path_context_reader import PathContextInputTensors, PathContextReader
import numpy as np
import time
import pickle
from common import common, VocabType
from keras_attention_layer import AttentionLayer
from keras_word_prediction_layer import WordPredictionLayer
from keras_words_subtoken_metrics import WordsSubtokenPrecisionMetric, WordsSubtokenRecallMetric, WordsSubtokenF1Metric
from config import Config
from model_base import ModelBase


class Code2VecModel(ModelBase):
    def __init__(self, config: Config):
        super(Code2VecModel, self).__init__(config)
        K.set_session(self.sess)

    def _build_keras_model(self) -> keras.Model:
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
        final_softmax_out = Dense(self.target_word_vocab_size + 1, use_bias=False, activation='softmax')(code_vectors)

        target_word_prediction = WordPredictionLayer(
            self.topk,
            self.index_to_target_word_table,
            predicted_words_filters=[
                lambda word_indices, _: tf.not_equal(word_indices, common.SpecialDictWords.NoSuchWord.value),
                lambda _, word_strings: tf.strings.regex_full_match(word_strings, r'^[a-zA-Z\|]+$')
            ])(final_softmax_out)

        # TODO: what to do with the output `target_word_prediction`? consider create a predict-only model.

        inputs = [source_terminals_input, paths_input, target_terminals_input, valid_mask]
        model = keras.Model(inputs=inputs, outputs=final_softmax_out)
        metrics = [
            WordsSubtokenPrecisionMetric(self.index_to_target_word_table, target_word_prediction),
            WordsSubtokenRecallMetric(self.index_to_target_word_table, target_word_prediction),
            WordsSubtokenF1Metric(self.index_to_target_word_table, target_word_prediction)
        ]
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=metrics)
        return model

    def train(self):
        train_data_input_reader = PathContextReader(
            token_to_index=self.word_to_index,
            path_to_index=self.path_to_index,
            target_to_index=self.target_word_to_index,
            config=self.config)

        keras_model = self._build_keras_model()
        print('Keras model built')

        # TODO: set keras model saver [max_to_keep=self.config.MAX_TO_KEEP]

        # TODO: initialize the evaluation reader (and pass to model.fit() call)

        self.initialize_session_variables(self.sess)

        keras_model.fit(train_data_input_reader.dataset, steps_per_epoch=self.config.steps_per_epoch, epochs=self.config.NUM_EPOCHS)

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
