import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

import PathContextReader
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

    def build_keras_training_model(self) -> keras.Model:
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
        model = keras.Model(inputs=inputs, outputs=out)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        return model

    def train(self):
        # TODO: Split to `_train_with_tf()` and `_train_with_keras()`.
        #       Keras `model.fit()` is much more elegant. No need for such manual evaluating & logging.

        print('Starting training')
        start_time = time.time()

        batch_num = 0
        sum_loss = 0
        multi_batch_start_time = time.time()
        num_batches_to_evaluate = max(int(
            self.config.NUM_EXAMPLES / self.config.BATCH_SIZE * self.config.SAVE_EVERY_EPOCHS), 1)

        self.queue_thread = PathContextReader.PathContextReader(word_to_index=self.word_to_index,
                                                                path_to_index=self.path_to_index,
                                                                target_word_to_index=self.target_word_to_index,
                                                                config=self.config)
        input_tensors = self.queue_thread.input_tensors()

        K.set_session(self.sess)
        model = self.build_keras_training_model()
        # TODO: set keras model saver [max_to_keep=self.config.MAX_TO_KEEP]

        self.initialize_session_variables(self.sess)
        print('Initalized variables')
        if self.config.LOAD_PATH:
            self.load_model(self.sess)
        with self.queue_thread.start(self.sess):
            time.sleep(1)
            print('Started reader...')
            try:
                while True:
                    # Each iteration = batch. We iterate as long as the tf iterator (reader) yields batches.
                    batch_num += 1

                    nr_batches_per_epoch = batch_num // num_batches_to_evaluate  # FIXME: change to the actual value
                    # Note: we call here `model.fit()` on each batch! it is not the wanted behaviour.
                    # TODO: make the `tf.data` iterator produce batches for `keras_model.fit_generator()`
                    #       without the external `while True` loop.
                    model.fit(input_tensors[1:], input_tensors[0], steps_per_epoch=nr_batches_per_epoch)

                    # TODO: Remove this. All of this part is actually relevant only for the tensorflow implementation.
                    if batch_num % self.num_batches_to_log == 0:
                        self.trace(sum_loss, batch_num, multi_batch_start_time)
                        print('Number of waiting examples in queue: %d' % self.sess.run(
                            "shuffle_batch/random_shuffle_queue_Size:0"))
                        sum_loss = 0
                        multi_batch_start_time = time.time()
                    if batch_num % num_batches_to_evaluate == 0:
                        epoch_num = int((batch_num / num_batches_to_evaluate) * self.config.SAVE_EVERY_EPOCHS)
                        save_target = self.config.SAVE_PATH + '_iter' + str(epoch_num)
                        self.save_model(self.sess, save_target)
                        print('Saved after %d epochs in: %s' % (epoch_num, save_target))
                        results, precision, recall, f1 = self.evaluate()
                        print('Accuracy after %d epochs: %s' % (epoch_num, results[:5]))
                        print('After ' + str(epoch_num) + ' epochs: Precision: ' + str(precision) + ', recall: ' + str(
                            recall) + ', F1: ' + str(f1))
            except tf.errors.OutOfRangeError:
                print('Done training')

        if self.config.SAVE_PATH:
            self.save_model(self.sess, self.config.SAVE_PATH)
            print('Model saved in file: %s' % self.config.SAVE_PATH)

        elapsed = int(time.time() - start_time)
        print("Training time: %sH:%sM:%sS\n" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))

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
