import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Input, Embedding, Concatenate, Dropout, TimeDistributed, Dense
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.metrics import sparse_top_k_categorical_accuracy

from path_context_reader import PathContextReader, ModelInputTensorsFormer, ReaderInputTensors
import os
import numpy as np
import time
import pickle
from functools import partial
from common import common, VocabType, SpecialDictWords
from keras_attention_layer import AttentionLayer
from keras_word_prediction_layer import WordPredictionLayer
from keras_words_subtoken_metrics import WordsSubtokenPrecisionMetric, WordsSubtokenRecallMetric, WordsSubtokenF1Metric
from config import Config
from model_base import ModelBase


class _KerasModelInputTensorsFormer(ModelInputTensorsFormer):
    def to_model_input_form(self, input_tensors: ReaderInputTensors):
        inputs = (input_tensors.path_source_indices, input_tensors.path_indices,
                  input_tensors.path_target_indices, input_tensors.context_valid_mask)
        targets = {'y_hat': input_tensors.target_index}
        return inputs, targets

    def from_model_input_form(self, input_row) -> ReaderInputTensors:
        inputs = input_row[0]
        targets = input_row[1]
        return ReaderInputTensors(
            path_source_indices=inputs[0],
            path_indices=inputs[1],
            path_target_indices=inputs[2],
            context_valid_mask=inputs[3],
            target_index=targets['y_hat']
        )


class Code2VecModel(ModelBase):
    def __init__(self, config: Config):
        super(Code2VecModel, self).__init__(config)

    def _build_keras_model(self) -> keras.Model:
        # Each input sample consists of a bag of x`MAX_CONTEXTS` tuples (source_terminal, path, target_terminal).
        # The valid mask indicates for each context whether it actually exists or it is just a padding.
        source_terminals_input = Input((self.config.MAX_CONTEXTS,), dtype=tf.int32)
        paths_input = Input((self.config.MAX_CONTEXTS,), dtype=tf.int32)
        target_terminals_input = Input((self.config.MAX_CONTEXTS,), dtype=tf.int32)
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
            Dense(self.config.EMBEDDINGS_SIZE * 3, input_dim=self.config.EMBEDDINGS_SIZE * 3,
                  use_bias=False, activation='tanh')
        )(context_embedded)

        # The final code vectors are received by applying attention to the "densed" context vectors.
        code_vectors = AttentionLayer(name='code_vectors')(context_after_dense, mask=valid_mask)

        # "Decode": Now we use another dense layer to get the target word embedding from each code vector.
        y_hat = Dense(self.target_word_vocab_size + 1, use_bias=False, activation='softmax', name='y_hat')(code_vectors)

        # Actual target word prediction (as string). Used as a second output layer.
        # `predict()` method just have to return the output of this layer.
        # Also used for the evaluation metrics calculations.
        target_word_prediction = WordPredictionLayer(
            self.config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION,
            self.index_to_target_word_table,
            predicted_words_filters=[
                lambda word_indices, _: tf.not_equal(word_indices, SpecialDictWords.OOV.index),
                lambda _, word_strings: tf.strings.regex_full_match(word_strings, r'^[a-zA-Z\|]+$')
            ], name='target_word_prediction')(y_hat)

        # Wrap the layers into a Keras model, using our subtoken-metrics and the CE loss.
        keras_model = keras.Model(
            inputs=[source_terminals_input, paths_input, target_terminals_input, valid_mask],
            outputs=[y_hat, code_vectors, target_word_prediction])
        top_k_acc = partial(sparse_top_k_categorical_accuracy, k=self.config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION)
        top_k_acc.__name__ = 'top{k}_acc'.format(k=self.config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION)
        metrics = {'y_hat': [
            top_k_acc,
            WordsSubtokenPrecisionMetric(self.index_to_target_word_table, target_word_prediction,
                                         name='subtoken_precision'),
            WordsSubtokenRecallMetric(self.index_to_target_word_table, target_word_prediction, name='subtoken_recall'),
            WordsSubtokenF1Metric(self.index_to_target_word_table, target_word_prediction, name='subtoken_f1')
        ]}
        keras_model.compile(loss={'y_hat': 'sparse_categorical_crossentropy'},
                            optimizer=tf.train.AdamOptimizer(),
                            metrics=metrics)
        return keras_model

    def _create_data_reader(self, is_evaluating: bool = False):
        return PathContextReader(
            token_to_index=self.word_to_index,
            path_to_index=self.path_to_index,
            target_to_index=self.target_word_to_index,
            config=self.config,
            model_input_tensors_former=_KerasModelInputTensorsFormer(),
            is_evaluating=is_evaluating)

    def train(self):
        # initialize the input pipeline readers
        train_data_input_reader = self._create_data_reader(is_evaluating=False)
        val_data_input_reader = self._create_data_reader(is_evaluating=True)

        # TODO: set max_to_keep=self.config.MAX_TO_KEEP
        # TODO: set the correct `monitor` quantity (example: monitor='val_acc', mode='max')
        # TODO: consider using tf.train.CheckpointManager
        # TODO: do we want to use early stopping?
        checkpoint = ModelCheckpoint(
            self.config.MODEL_SAVE_PATH + '_epoch{epoch}',
            verbose=1, save_best_only=False, save_weights_only=self.config.RELEASE,
            period=self.config.SAVE_EVERY_EPOCHS)

        self.initialize_variables()

        self.keras_model.fit(
            train_data_input_reader.dataset,
            steps_per_epoch=self.config.train_steps_per_epoch,
            epochs=self.config.NUM_EPOCHS,
            batch_size=self.config.TRAIN_BATCH_SIZE,
            validation_data=val_data_input_reader.dataset,
            validation_steps=self.config.test_steps_per_epoch,
            callbacks=[checkpoint])

    def evaluate(self):
        val_data_input_reader = self._create_data_reader(is_evaluating=True)
        self.initialize_variables()
        return self.keras_model.evaluate(val_data_input_reader.dataset, steps=self.config.test_steps_per_epoch)

    def predict(self, predict_data_lines):
        val_data_input_reader = self._create_data_reader(is_evaluating=True)
        self.initialize_variables()
        return self.keras_model.predict(val_data_input_reader.dataset, steps=self.config.test_steps_per_epoch)

    def _save_inner_model(self, path):
        if self.config.RELEASE:
            self.keras_model.save_weights(path + '.keras_weights')
        else:
            self.keras_model.save(path + '.keras_model')

    def _load_or_build_inner_model(self):
        K.set_session(self.sess)

        if not self.config.MODEL_LOAD_PATH:
            self.keras_model = self._build_keras_model()
        else:
            model_file_path = self.config.MODEL_LOAD_PATH + '.keras_model'
            weights_file_path = self.config.MODEL_LOAD_PATH + '.keras_weights'

            # when loading the model for further training, we must use the full saved model file (not just weights).
            must_use_full_model = self.config.TRAIN_DATA_PATH
            if must_use_full_model and not os.path.isfile(model_file_path):
                raise ValueError(
                    "There is no model at path `{model_file_path}`. When loading the model for further training,"
                    "we must use a full saved model file (not just weights).".format(model_file_path=model_file_path))
            use_full_model = must_use_full_model or not os.path.isfile(weights_file_path)

            if use_full_model:
                self.keras_model = keras.models.load_model(model_file_path)
            else:
                # load the "released" model (only the weights).
                self.keras_model = self._build_keras_model()
                self.keras_model.load_weights(weights_file_path)

        self.keras_model.summary()

    def save_word2vec_format(self, dest, source):
        raise NotImplemented()  # TODO: implement!
