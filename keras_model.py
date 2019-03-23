import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Input, Embedding, Concatenate, Dropout, TimeDistributed, Dense
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint

from path_context_reader import PathContextReader, ModelInputTensorsFormer, ReaderInputTensors
import numpy as np
import time
import pickle
from common import common, VocabType
from keras_attention_layer import AttentionLayer
from keras_word_prediction_layer import WordPredictionLayer
from keras_words_subtoken_metrics import WordsSubtokenPrecisionMetric, WordsSubtokenRecallMetric, WordsSubtokenF1Metric
from config import Config
from model_base import ModelBase


class _KerasModelInputTensorsFormer(ModelInputTensorsFormer):
    def to_model_input_form(self, input_tensors: ReaderInputTensors):
        inputs = tuple(input_tensors)[:4]
        targets = {'y_hat': input_tensors.target_index}
        return inputs, targets

    def from_model_input_form(self, input_row) -> ReaderInputTensors:
        inputs = input_row[0]
        targets = {'target_index': input_row[1]['y_hat']}
        return ReaderInputTensors(*inputs, **targets)


class Code2VecModel(ModelBase):
    def __init__(self, config: Config):
        super(Code2VecModel, self).__init__(config)
        K.set_session(self.sess)
        self.keras_model = self._build_keras_model()
        print('Keras model built')

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
        y_hat = Dense(self.target_word_vocab_size + 1, use_bias=False, activation='softmax', name='y_hat')(code_vectors)

        # Actual target word prediction (as string). Used as a second output layer.
        # `predict()` method just have to return the output of this layer.
        # Also used for the evaluation metrics calculations.
        target_word_prediction = WordPredictionLayer(
            self.config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION,
            self.index_to_target_word_table,
            predicted_words_filters=[
                lambda word_indices, _: tf.not_equal(word_indices, common.SpecialDictWords.NoSuchWord.value),
                lambda _, word_strings: tf.strings.regex_full_match(word_strings, r'^[a-zA-Z\|]+$')
            ], name='target_word_prediction')(y_hat)

        # Wrap the layers into a Keras model, using our subtoken-metrics and the CE loss.
        inputs = [source_terminals_input, paths_input, target_terminals_input, valid_mask]
        keras_model = keras.Model(inputs=inputs, outputs=[y_hat, target_word_prediction])
        metrics = {'y_hat': [
            WordsSubtokenPrecisionMetric(self.index_to_target_word_table, target_word_prediction, name='precision'),
            WordsSubtokenRecallMetric(self.index_to_target_word_table, target_word_prediction, name='recall'),
            WordsSubtokenF1Metric(self.index_to_target_word_table, target_word_prediction, name='f1')
        ]}
        keras_model.compile(loss={'y_hat': 'sparse_categorical_crossentropy'}, optimizer='adam', metrics=metrics)
        return keras_model

    def train(self):
        # initialize the train input pipeline reader
        train_data_input_reader = PathContextReader(
            token_to_index=self.word_to_index,
            path_to_index=self.path_to_index,
            target_to_index=self.target_word_to_index,
            config=self.config,
            model_input_tensors_former=_KerasModelInputTensorsFormer())

        # initialize the validation input pipeline reader
        val_data_input_reader = PathContextReader(
            token_to_index=self.word_to_index,
            path_to_index=self.path_to_index,
            target_to_index=self.target_word_to_index,
            config=self.config,
            model_input_tensors_former=_KerasModelInputTensorsFormer(),
            is_evaluating=True)

        # TODO: set max_to_keep=self.config.MAX_TO_KEEP
        # TODO: set the correct `monitor` quantity (example: monitor='val_acc', mode='max')
        checkpoint = ModelCheckpoint(
            self.config.SAVE_PATH + '_epoch{epoch:08d}',
            verbose=1, save_best_only=False, save_weights_only=True,
            period=self.config.SAVE_EVERY_EPOCHS)

        self.initialize_session_variables(self.sess)

        self.keras_model.fit(
            train_data_input_reader.dataset,
            steps_per_epoch=2,  #self.config.train_steps_per_epoch,
            epochs=self.config.NUM_EPOCHS,
            validation_data=val_data_input_reader.dataset,
            # validation_steps=self.config.test_steps_per_epoch,  # FIXME: how to obtain #VALIDATION_EXAMPLES?
            callbacks=[checkpoint])

    def evaluate(self):
        raise NotImplemented()  # TODO: implement!

    def predict(self, predict_data_lines):
        # TODO: make `predict()` a base method, and add a new abstract methods for the actual framework-dependant.
        raise NotImplemented()  # TODO: implement!

    def save_model(self, sess, path):
        self.keras_model.save_weights(self.config.SAVE_PATH)

    def load_model(self, sess):
        self.keras_model.load_weights(self.config.LOAD_PATH)

    def save_word2vec_format(self, dest, source):
        raise NotImplemented()  # TODO: implement!
