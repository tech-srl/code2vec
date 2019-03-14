import tensorflow as tf
import numpy as np
import time
import pickle

import PathContextReader
from common import common, VocabType
from common import Config
from model_base import ModelBase


class Model(ModelBase):
    def __init__(self, config: Config):
        super(Model, self).__init__(config)

    def train(self):
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
        optimizer, train_loss = self.build_tf_training_graph(input_tensors)
        self.saver = tf.train.Saver(max_to_keep=self.config.MAX_TO_KEEP)

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

                    # Actual training for the current batch.
                    _, batch_loss = self.sess.run([optimizer, train_loss])

                    sum_loss += batch_loss
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
                # Means the reader have no more batches to produce.
                print('Done training')

        if self.config.SAVE_PATH:
            self.save_model(self.sess, self.config.SAVE_PATH)
            print('Model saved in file: %s' % self.config.SAVE_PATH)

        elapsed = int(time.time() - start_time)
        print("Training time: %sH:%sM:%sS\n" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))

    def evaluate(self):
        eval_start_time = time.time()
        if self.eval_queue is None:
            self.eval_queue = PathContextReader.PathContextReader(word_to_index=self.word_to_index,
                                                                  path_to_index=self.path_to_index,
                                                                  target_word_to_index=self.target_word_to_index,
                                                                  config=self.config, is_evaluating=True)
            self.eval_placeholder = self.eval_queue.get_input_placeholder()
            self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op, _, _, _, _, self.eval_code_vectors = \
                self.build_test_graph(self.eval_queue.get_filtered_batches())
            self.saver = tf.train.Saver()

        if self.config.LOAD_PATH and not self.config.TRAIN_PATH:
            self.initialize_session_variables(self.sess)
            self.load_model(self.sess)
            if self.config.RELEASE:
                release_name = self.config.LOAD_PATH + '.release'
                print('Releasing model, output model: %s' % release_name )
                self.saver.save(self.sess, release_name )
                return None

        if self.eval_data_lines is None:
            print('Loading test data from: ' + self.config.TEST_PATH)
            self.eval_data_lines = common.load_file_lines(self.config.TEST_PATH)
            print('Done loading test data')

        with open('log.txt', 'w') as output_file:
            if self.config.EXPORT_CODE_VECTORS:
                code_vectors_file = open(self.config.TEST_PATH + '.vectors', 'w')
            num_correct_predictions = np.zeros(self.topk)
            total_predictions = 0
            total_prediction_batches = 0
            true_positive, false_positive, false_negative = 0, 0, 0
            start_time = time.time()

            for batch in common.split_to_batches(self.eval_data_lines, self.config.TEST_BATCH_SIZE):
                top_words, top_scores, original_names, code_vectors = self.sess.run(
                    [self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op, self.eval_code_vectors],
                    feed_dict={self.eval_placeholder: batch})
                top_words, original_names = common.binary_to_string_matrix(top_words), common.binary_to_string_matrix(
                    original_names)
                # Flatten original names from [[]] to []
                original_names = [w for l in original_names for w in l]

                num_correct_predictions = self.update_correct_predictions(num_correct_predictions, output_file,
                                                                          zip(original_names, top_words))
                true_positive, false_positive, false_negative = self.update_per_subtoken_statistics(
                    zip(original_names, top_words),
                    true_positive, false_positive, false_negative)

                total_predictions += len(original_names)
                total_prediction_batches += 1
                if self.config.EXPORT_CODE_VECTORS:
                    self.write_code_vectors(code_vectors_file, code_vectors)
                if total_prediction_batches % self.num_batches_to_log == 0:
                    elapsed = time.time() - start_time
                    # start_time = time.time()
                    self.trace_evaluation(output_file, num_correct_predictions, total_predictions, elapsed, len(self.eval_data_lines))

            print('Done testing, epoch reached')
            output_file.write(str(num_correct_predictions / total_predictions) + '\n')
        if self.config.EXPORT_CODE_VECTORS:
            code_vectors_file.close()
        
        elapsed = int(time.time() - eval_start_time)
        precision, recall, f1 = self.calculate_results(true_positive, false_positive, false_negative)
        print("Evaluation time: %sH:%sM:%sS" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))
        del self.eval_data_lines
        self.eval_data_lines = None
        return num_correct_predictions / total_predictions, precision, recall, f1

    def build_tf_training_graph(self, input_tensors):
        words_input, source_input, path_input, target_input, valid_mask = input_tensors  # (batch, 1),   (batch, max_contexts)

        with tf.variable_scope('model'):
            words_vocab = tf.get_variable('WORDS_VOCAB', shape=(self.word_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                     mode='FAN_OUT',
                                                                                                     uniform=True))
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(
                                                     self.target_word_vocab_size + 1, self.config.EMBEDDINGS_SIZE * 3),
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                            mode='FAN_OUT',
                                                                                                            uniform=True))
            attention_param = tf.get_variable('ATTENTION',
                                              shape=(self.config.EMBEDDINGS_SIZE * 3, 1), dtype=tf.float32)
            paths_vocab = tf.get_variable('PATHS_VOCAB', shape=(self.path_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32,
                                          initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                     mode='FAN_OUT',
                                                                                                     uniform=True))

            code_vectors, _ = self.calculate_weighted_contexts(words_vocab, paths_vocab, attention_param,
                                                                            source_input, path_input, target_input,
                                                                            valid_mask)

            logits = tf.matmul(code_vectors, target_words_vocab, transpose_b=True)
            batch_size = tf.to_float(tf.shape(words_input)[0])
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(words_input, [-1]),
                logits=logits)) / batch_size

            optimizer = tf.train.AdamOptimizer().minimize(loss)

        return optimizer, loss

    def calculate_weighted_contexts(self, words_vocab, paths_vocab, attention_param, source_input, path_input,
                                    target_input, valid_mask, is_evaluating=False):
        source_word_embed = tf.nn.embedding_lookup(params=words_vocab, ids=source_input)  # (batch, max_contexts, dim)
        path_embed = tf.nn.embedding_lookup(params=paths_vocab, ids=path_input)  # (batch, max_contexts, dim)
        target_word_embed = tf.nn.embedding_lookup(params=words_vocab, ids=target_input)  # (batch, max_contexts, dim)

        context_embed = tf.concat([source_word_embed, path_embed, target_word_embed],
                                  axis=-1)  # (batch, max_contexts, dim * 3)
        if not is_evaluating:
            context_embed = tf.nn.dropout(context_embed, self.config.DROPOUT_KEEP_RATE)

        flat_embed = tf.reshape(context_embed, [-1, self.config.EMBEDDINGS_SIZE * 3])  # (batch * max_contexts, dim * 3)
        transform_param = tf.get_variable('TRANSFORM',
                                          shape=(self.config.EMBEDDINGS_SIZE * 3, self.config.EMBEDDINGS_SIZE * 3),
                                          dtype=tf.float32)

        flat_embed = tf.tanh(tf.matmul(flat_embed, transform_param))  # (batch * max_contexts, dim * 3)

        contexts_weights = tf.matmul(flat_embed, attention_param)  # (batch * max_contexts, 1)
        batched_contexts_weights = tf.reshape(contexts_weights,
                                              [-1, self.config.MAX_CONTEXTS, 1])  # (batch, max_contexts, 1)
        mask = tf.log(valid_mask)  # (batch, max_contexts)
        mask = tf.expand_dims(mask, axis=2)  # (batch, max_contexts, 1)
        batched_contexts_weights += mask  # (batch, max_contexts, 1)
        attention_weights = tf.nn.softmax(batched_contexts_weights, axis=1)  # (batch, max_contexts, 1)

        batched_embed = tf.reshape(flat_embed, shape=[-1, self.config.MAX_CONTEXTS, self.config.EMBEDDINGS_SIZE * 3])
        code_vectors = tf.reduce_sum(tf.multiply(batched_embed, attention_weights), axis=1)  # (batch, dim * 3)

        return code_vectors, attention_weights

    def build_test_graph(self, input_tensors, normalize_scores=False):
        with tf.variable_scope('model', reuse=self.get_should_reuse_variables()):
            words_vocab = tf.get_variable('WORDS_VOCAB', shape=(self.word_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)
            target_words_vocab = tf.get_variable('TARGET_WORDS_VOCAB',
                                                 shape=(
                                                     self.target_word_vocab_size + 1, self.config.EMBEDDINGS_SIZE * 3),
                                                 dtype=tf.float32, trainable=False)
            attention_param = tf.get_variable('ATTENTION',
                                              shape=(self.config.EMBEDDINGS_SIZE * 3, 1),
                                              dtype=tf.float32, trainable=False)
            paths_vocab = tf.get_variable('PATHS_VOCAB',
                                          shape=(self.path_vocab_size + 1, self.config.EMBEDDINGS_SIZE),
                                          dtype=tf.float32, trainable=False)

            target_words_vocab = tf.transpose(target_words_vocab)  # (dim * 3, target_word_vocab+1)

            words_input, source_input, path_input, target_input, valid_mask, source_string, path_string, path_target_string = input_tensors  # (batch, 1), (batch, max_contexts)

            code_vectors, attention_weights = self.calculate_weighted_contexts(words_vocab, paths_vocab,
                                                                                            attention_param,
                                                                                            source_input, path_input,
                                                                                            target_input,
                                                                                            valid_mask, True)

        scores = tf.matmul(code_vectors, target_words_vocab) # (batch, target_word_vocab+1)

        topk_candidates = tf.nn.top_k(scores, k=tf.minimum(self.topk, self.target_word_vocab_size))
        top_indices = tf.to_int64(topk_candidates.indices)
        top_words = self.index_to_target_word_table.lookup(top_indices)
        original_words = words_input
        top_scores = topk_candidates.values
        if normalize_scores:
            top_scores = tf.nn.softmax(top_scores)

        return top_words, top_scores, original_words, attention_weights, source_string, path_string, path_target_string, code_vectors

    def predict(self, predict_data_lines):
        # TODO: make `predict()` a base method, and add a new abstract methods for the actual framework-dependant.
        if self.predict_queue is None:
            self.predict_queue = PathContextReader.PathContextReader(word_to_index=self.word_to_index,
                                                                     path_to_index=self.path_to_index,
                                                                     target_word_to_index=self.target_word_to_index,
                                                                     config=self.config, is_evaluating=True)
            self.predict_placeholder = self.predict_queue.get_input_placeholder()
            self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op, \
            self.attention_weights_op, self.predict_source_string, self.predict_path_string, self.predict_path_target_string, self.predict_code_vectors = \
                self.build_test_graph(self.predict_queue.get_filtered_batches(), normalize_scores=True)

            self.initialize_session_variables(self.sess)
            self.saver = tf.train.Saver()
            self.load_model(self.sess)

        code_vectors = []
        results = []
        for batch in common.split_to_batches(predict_data_lines, 1):
            top_words, top_scores, original_names, attention_weights, source_strings, path_strings, target_strings, batch_code_vectors = self.sess.run(
                [self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op,
                 self.attention_weights_op, self.predict_source_string, self.predict_path_string,
                 self.predict_path_target_string, self.predict_code_vectors],
                feed_dict={self.predict_placeholder: batch})
            top_words, original_names = common.binary_to_string_matrix(top_words), common.binary_to_string_matrix(
                original_names)
            # Flatten original names from [[]] to []
            attention_per_path = self.get_attention_per_path(source_strings, path_strings, target_strings,
                                                             attention_weights)
            original_names = [w for l in original_names for w in l]
            results.append((original_names[0], top_words[0], top_scores[0], attention_per_path))
            if self.config.EXPORT_CODE_VECTORS:
                code_vectors.append(batch_code_vectors)
        if len(code_vectors) > 0:
            code_vectors = np.vstack(code_vectors)
        return results, code_vectors

    def save_model(self, sess, path):
        self.saver.save(sess, path)
        with open(self.get_dictionaries_path(path), 'wb') as file:
            pickle.dump(self.word_to_index, file)
            pickle.dump(self.index_to_word, file)
            pickle.dump(self.word_vocab_size, file)

            pickle.dump(self.target_word_to_index, file)
            pickle.dump(self.index_to_target_word, file)
            pickle.dump(self.target_word_vocab_size, file)

            pickle.dump(self.path_to_index, file)
            pickle.dump(self.index_to_path, file)
            pickle.dump(self.path_vocab_size, file)

    def load_model(self, sess):
        if not sess is None:
            print('Loading model weights from: ' + self.config.LOAD_PATH)
            self.saver.restore(sess, self.config.LOAD_PATH)
            print('Done')
        dictionaries_path = self.get_dictionaries_path(self.config.LOAD_PATH)
        with open(dictionaries_path , 'rb') as file:
            print('Loading model dictionaries from: %s' % dictionaries_path)
            self.word_to_index = pickle.load(file)
            self.index_to_word = pickle.load(file)
            self.word_vocab_size = pickle.load(file)

            self.target_word_to_index = pickle.load(file)
            self.index_to_target_word = pickle.load(file)
            self.target_word_vocab_size = pickle.load(file)

            self.path_to_index = pickle.load(file)
            self.index_to_path = pickle.load(file)
            self.path_vocab_size = pickle.load(file)
            print('Done')

    def save_word2vec_format(self, dest, source):
        with tf.variable_scope('model', reuse=None):
            if source is VocabType.Token:
                vocab_size = self.word_vocab_size
                embedding_size = self.config.EMBEDDINGS_SIZE
                index = self.index_to_word
                var_name = 'WORDS_VOCAB'
            elif source is VocabType.Target:
                vocab_size = self.target_word_vocab_size
                embedding_size = self.config.EMBEDDINGS_SIZE * 3
                index = self.index_to_target_word
                var_name = 'TARGET_WORDS_VOCAB'
            else:
                raise ValueError('vocab type should be VocabType.Token or VocabType.Target.')
            embeddings = tf.get_variable(var_name, shape=(vocab_size + 1, embedding_size), dtype=tf.float32,
                                         trainable=False)
            self.saver = tf.train.Saver()
            self.load_model(self.sess)
            np_embeddings = self.sess.run(embeddings)
        with open(dest, 'w') as words_file:
            common.save_word2vec_file(words_file, vocab_size, embedding_size, index, np_embeddings)
