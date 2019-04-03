import tensorflow as tf
import numpy as np
import time
from typing import Dict

from path_context_reader import PathContextReader, ModelInputTensorsFormer, ReaderInputTensors
from common import common
from vocabularies import VocabType
from config import Config
from model_base import ModelBase


class _TFTrainModelInputTensorsFormer(ModelInputTensorsFormer):
    def to_model_input_form(self, input_tensors: ReaderInputTensors):
        return input_tensors.target_index, input_tensors.path_source_token_indices, input_tensors.path_indices, \
               input_tensors.path_target_token_indices, input_tensors.context_valid_mask

    def from_model_input_form(self, input_row) -> ReaderInputTensors:
        return ReaderInputTensors(
            target_index=input_row[0],
            path_source_token_indices=input_row[1],
            path_indices=input_row[2],
            path_target_token_indices=input_row[3],
            context_valid_mask=input_row[4]
        )


class _TFEvaluateModelInputTensorsFormer(ModelInputTensorsFormer):
    def to_model_input_form(self, input_tensors: ReaderInputTensors):
        return input_tensors.target_string, input_tensors.path_source_token_indices, input_tensors.path_indices, \
               input_tensors.path_target_token_indices, input_tensors.context_valid_mask, \
               input_tensors.path_source_token_strings, input_tensors.path_strings, \
               input_tensors.path_target_token_strings

    def from_model_input_form(self, input_row) -> ReaderInputTensors:
        return ReaderInputTensors(
            target_string=input_row[0],
            path_source_token_indices=input_row[1],
            path_indices=input_row[2],
            path_target_token_indices=input_row[3],
            context_valid_mask=input_row[4],
            path_source_token_strings=input_row[5],
            path_strings=input_row[6],
            path_target_token_strings=input_row[7]
        )


class Code2VecModel(ModelBase):
    num_batches_to_log = 100  # TODO: consider exporting to Config or to method param with default value.

    def __init__(self, config: Config):
        self.sess = tf.Session()
        self.saver = None

        self.eval_reader = None
        self.eval_input_iterator_reset_op = None
        self.predict_reader = None

        # self.eval_placeholder = None
        self.predict_placeholder = None
        self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op, self.eval_code_vectors = None, None, None, None
        self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op = None, None, None

        self.vocab_type_to_tf_variable_name_mapping: Dict[VocabType, str] = {
            VocabType.Token: 'TOKENS_VOCAB',
            VocabType.Target: 'TARGETS_VOCAB',
            VocabType.Path: 'PATHS_VOCAB'
        }

        super(Code2VecModel, self).__init__(config)

    def train(self):
        print('Starting training')
        start_time = time.time()

        batch_num = 0
        sum_loss = 0
        multi_batch_start_time = time.time()
        num_batches_to_evaluate = max(int(
            self.config.NUM_TRAIN_EXAMPLES / self.config.TRAIN_BATCH_SIZE * self.config.SAVE_EVERY_EPOCHS), 1)

        self.train_reader = PathContextReader(vocabs=self.vocabs,
                                              model_input_tensors_former=_TFTrainModelInputTensorsFormer(),
                                              config=self.config)
        input_iterator = self.train_reader.dataset.make_initializable_iterator()
        input_iterator_reset_op = input_iterator.initializer
        input_tensors = input_iterator.get_next()

        optimizer, train_loss = self._build_tf_training_graph(input_tensors)
        self.saver = tf.train.Saver(max_to_keep=self.config.MAX_TO_KEEP)

        self._initialize_session_variables()

        if self.config.MODEL_LOAD_PATH:
            self._load_inner_model(self.sess)

        self.sess.run(input_iterator_reset_op)
        time.sleep(1)
        print('Started reader...')
        # run evaluation in a loop until iterator is exhausted.
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
                    save_target = self.config.MODEL_SAVE_PATH + '_iter' + str(epoch_num)
                    self._save_inner_model(self.sess, save_target)
                    print('Saved after %d epochs in: %s' % (epoch_num, save_target))
                    results, precision, recall, f1 = self.evaluate()
                    print('Accuracy after %d epochs: %s' % (epoch_num, results[:5]))
                    print('After ' + str(epoch_num) + ' epochs: Precision: ' + str(precision) + ', recall: ' + str(
                        recall) + ', F1: ' + str(f1))
        except tf.errors.OutOfRangeError:
            pass  # The reader iterator is exhausted and have no more batches to produce.

        print('Done training')

        if self.config.MODEL_SAVE_PATH:
            self._save_inner_model(self.sess, self.config.MODEL_SAVE_PATH)
            print('Model saved in file: %s' % self.config.MODEL_SAVE_PATH)

        elapsed = int(time.time() - start_time)
        print("Training time: %sH:%sM:%sS\n" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))

    def evaluate(self):
        eval_start_time = time.time()
        if self.eval_reader is None:
            self.eval_reader = PathContextReader(vocabs=self.vocabs,
                                                 model_input_tensors_former=_TFEvaluateModelInputTensorsFormer(),
                                                 config=self.config, is_evaluating=True)
            input_iterator = self.train_reader.dataset.make_initializable_iterator()
            self.eval_input_iterator_reset_op = input_iterator.initializer
            input_tensors = input_iterator.get_next()

            self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op, _, _, _, _, \
                self.eval_code_vectors = self._build_tf_test_graph(input_tensors)
            self.saver = tf.train.Saver()

        if self.config.MODEL_LOAD_PATH and not self.config.TRAIN_DATA_PATH_PREFIX:
            self._initialize_session_variables()
            self._load_inner_model(self.sess)
            if self.config.RELEASE:
                release_name = self.config.MODEL_LOAD_PATH + '.release'
                print('Releasing model, output model: %s' % release_name)
                self.saver.save(self.sess, release_name)
                return None

        with open('log.txt', 'w') as output_file:
            if self.config.EXPORT_CODE_VECTORS:
                code_vectors_file = open(self.config.TEST_DATA_PATH + '.vectors', 'w')
            num_correct_predictions = np.zeros(self.config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION)
            total_predictions = 0
            total_prediction_batches = 0
            true_positive, false_positive, false_negative = 0, 0, 0
            start_time = time.time()

            self.sess.run(self.eval_input_iterator_reset_op)

            # Run evaluation in a loop until iterator is exhausted.
            # Each iteration = batch. We iterate as long as the tf iterator (reader) yields batches.
            try:
                while True:
                    top_words, top_scores, original_names, code_vectors = self.sess.run(
                        [self.eval_top_words_op, self.eval_top_values_op,
                         self.eval_original_names_op, self.eval_code_vectors],
                    )
                    top_words = common.binary_to_string_matrix(top_words)
                    original_names = common.binary_to_string_matrix(original_names)
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
                        self._write_code_vectors(code_vectors_file, code_vectors)
                    if total_prediction_batches % self.num_batches_to_log == 0:
                        elapsed = time.time() - start_time
                        # start_time = time.time()
                        self.trace_evaluation(output_file, num_correct_predictions, total_predictions, elapsed)
            except tf.errors.OutOfRangeError:
                pass  # reader iterator is exhausted and have no more batches to produce.
            print('Done testing, epoch reached')
            output_file.write(str(num_correct_predictions / total_predictions) + '\n')
        if self.config.EXPORT_CODE_VECTORS:
            code_vectors_file.close()
        
        elapsed = int(time.time() - eval_start_time)
        precision, recall, f1 = self.calculate_results(true_positive, false_positive, false_negative)
        print("Evaluation time: %sH:%sM:%sS" % ((elapsed // 60 // 60), (elapsed // 60) % 60, elapsed % 60))
        return num_correct_predictions / total_predictions, precision, recall, f1

    def _build_tf_training_graph(self, input_tensors):
        # Use `_TFTrainModelInputTensorsFormer` to access input tensors by name.
        input_tensors = _TFTrainModelInputTensorsFormer().from_model_input_form(input_tensors)
        # shape of (batch, 1) for input_tensors.target_index
        # shape of (batch, max_contexts) for others:
        #   input_tensors.path_source_indices, input_tensors.path_indices,
        #   input_tensors.path_target_indices, input_tensors.context_valid_mask

        with tf.variable_scope('model'):
            tokens_vocab = tf.get_variable(
                'TOKENS_VOCAB',
                shape=(self.vocabs.token_vocab.size, self.config.TOKEN_EMBEDDINGS_SIZE), dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))
            targets_vocab = tf.get_variable(
                'TARGETS_VOCAB',
                shape=(self.vocabs.target_vocab.size, self.config.CODE_VECTOR_SIZE), dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))
            attention_param = tf.get_variable(
                'ATTENTION',
                shape=(self.config.CODE_VECTOR_SIZE, 1), dtype=tf.float32)
            paths_vocab = tf.get_variable(
                'PATHS_VOCAB',
                shape=(self.vocabs.path_vocab.size, self.config.PATH_EMBEDDINGS_SIZE), dtype=tf.float32,
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', uniform=True))

            code_vectors, _ = self.calculate_weighted_contexts(
                tokens_vocab, paths_vocab, attention_param, input_tensors.path_source_token_indices,
                input_tensors.path_indices, input_tensors.path_target_token_indices, input_tensors.context_valid_mask)

            logits = tf.matmul(code_vectors, targets_vocab, transpose_b=True)
            batch_size = tf.to_float(tf.shape(input_tensors.target_index)[0])
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(input_tensors.target_index, [-1]),
                logits=logits)) / batch_size

            optimizer = tf.train.AdamOptimizer().minimize(loss)

        return optimizer, loss

    def calculate_weighted_contexts(self, tokens_vocab, paths_vocab, attention_param, source_input, path_input,
                                    target_input, valid_mask, is_evaluating=False):
        source_word_embed = tf.nn.embedding_lookup(params=tokens_vocab, ids=source_input)  # (batch, max_contexts, dim)
        path_embed = tf.nn.embedding_lookup(params=paths_vocab, ids=path_input)  # (batch, max_contexts, dim)
        target_word_embed = tf.nn.embedding_lookup(params=tokens_vocab, ids=target_input)  # (batch, max_contexts, dim)

        context_embed = tf.concat([source_word_embed, path_embed, target_word_embed],
                                  axis=-1)  # (batch, max_contexts, dim * 3)
        if not is_evaluating:
            context_embed = tf.nn.dropout(context_embed, self.config.DROPOUT_KEEP_RATE)

        flat_embed = tf.reshape(context_embed, [-1, self.config.CONTEXT_EMBEDDINGS_SIZE])  # (batch * max_contexts, dim * 3)
        transform_param = tf.get_variable(
            'TRANSFORM', shape=(self.config.CONTEXT_EMBEDDINGS_SIZE, self.config.CODE_VECTOR_SIZE), dtype=tf.float32)

        flat_embed = tf.tanh(tf.matmul(flat_embed, transform_param))  # (batch * max_contexts, dim * 3)

        contexts_weights = tf.matmul(flat_embed, attention_param)  # (batch * max_contexts, 1)
        batched_contexts_weights = tf.reshape(contexts_weights,
                                              [-1, self.config.MAX_CONTEXTS, 1])  # (batch, max_contexts, 1)
        mask = tf.log(valid_mask)  # (batch, max_contexts)
        mask = tf.expand_dims(mask, axis=2)  # (batch, max_contexts, 1)
        batched_contexts_weights += mask  # (batch, max_contexts, 1)
        attention_weights = tf.nn.softmax(batched_contexts_weights, axis=1)  # (batch, max_contexts, 1)

        batched_embed = tf.reshape(flat_embed, shape=[-1, self.config.MAX_CONTEXTS, self.config.CODE_VECTOR_SIZE])
        code_vectors = tf.reduce_sum(tf.multiply(batched_embed, attention_weights), axis=1)  # (batch, dim * 3)

        return code_vectors, attention_weights

    def _build_tf_test_graph(self, input_tensors, normalize_scores=False):
        with tf.variable_scope('model', reuse=self.get_should_reuse_variables()):
            tokens_vocab = tf.get_variable(
                'TOKENS_VOCAB', shape=(self.vocabs.token_vocab.size, self.config.TOKEN_EMBEDDINGS_SIZE),
                dtype=tf.float32, trainable=False)
            targets_vocab = tf.get_variable(
                'TARGETS_VOCAB', shape=(self.vocabs.target_vocab.size, self.config.TARGET_EMBEDDINGS_SIZE),
                dtype=tf.float32, trainable=False)
            attention_param = tf.get_variable(
                'ATTENTION', shape=(self.config.EMBEDDINGS_SIZE * 3, 1),
                dtype=tf.float32, trainable=False)
            paths_vocab = tf.get_variable(
                'PATHS_VOCAB', shape=(self.vocabs.path_vocab.size, self.config.PATH_EMBEDDINGS_SIZE),
                dtype=tf.float32, trainable=False)

            targets_vocab = tf.transpose(targets_vocab)  # (dim * 3, target_word_vocab+1)

            # Use `_TFEvaluateModelInputTensorsFormer` to access input tensors by name.
            input_tensors = _TFEvaluateModelInputTensorsFormer().from_model_input_form(input_tensors)
            # shape of (batch, 1) for input_tensors.target_string
            # shape of (batch, max_contexts) for the other tensors

            code_vectors, attention_weights = self.calculate_weighted_contexts(
                tokens_vocab, paths_vocab, attention_param, input_tensors.path_source_token_indices,
                input_tensors.path_indices, input_tensors.path_target_token_indices,
                input_tensors.context_valid_mask, is_evaluating=True)

        scores = tf.matmul(code_vectors, targets_vocab)  # (batch, target_word_vocab+1)

        topk_candidates = tf.nn.top_k(scores, k=tf.minimum(
            self.config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION, self.vocabs.target_vocab.size))
        top_indices = tf.to_int64(topk_candidates.indices)
        top_words = self.vocabs.target_vocab.lookup_word(top_indices)
        original_words = input_tensors.target_string
        top_scores = topk_candidates.values
        if normalize_scores:
            top_scores = tf.nn.softmax(top_scores)

        return top_words, top_scores, original_words, attention_weights, input_tensors.path_source_token_strings, \
               input_tensors.path_strings, input_tensors.path_target_token_strings, code_vectors

    def predict(self, predict_data_lines):
        if self.predict_reader is None:
            self.predict_reader = PathContextReader(vocabs=self.vocabs,
                                                    model_input_tensors_former=_TFEvaluateModelInputTensorsFormer(),
                                                    config=self.config, is_evaluating=True)

            self.predict_placeholder = tf.placeholder(tf.string)
            reader_output = self.predict_reader.process_from_placeholder(self.predict_placeholder)
            reader_output = tuple(tf.expand_dims(tensor, 0) for tensor in reader_output)

            self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op, \
            self.attention_weights_op, self.predict_source_string, self.predict_path_string, \
            self.predict_path_target_string, self.predict_code_vectors = \
                self._build_tf_test_graph(reader_output, normalize_scores=True)

            self._initialize_session_variables()
            self.saver = tf.train.Saver()
            self._load_inner_model(sess=self.sess)

        code_vectors = []
        results = []
        for line in predict_data_lines:
            top_words, top_scores, original_names, attention_weights, source_strings, path_strings, target_strings, \
                batch_code_vectors = self.sess.run(
                    [self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op,
                     self.attention_weights_op, self.predict_source_string, self.predict_path_string,
                     self.predict_path_target_string, self.predict_code_vectors],
                    feed_dict={self.predict_placeholder: line})
            top_words = common.binary_to_string_matrix(top_words)
            original_names = common.binary_to_string_matrix(original_names)
            # Flatten original names from [[]] to []
            attention_per_path = self.get_attention_per_path(
                source_strings, path_strings, target_strings, attention_weights)
            original_names = [w for l in original_names for w in l]
            results.append((original_names[0], top_words[0], top_scores[0], attention_per_path))
            if self.config.EXPORT_CODE_VECTORS:
                code_vectors.append(batch_code_vectors)
        if len(code_vectors) > 0:
            code_vectors = np.vstack(code_vectors)
        return results, code_vectors

    def _save_inner_model(self, path, sess=None):
        self.saver.save(sess, path)

    def _load_inner_model(self, sess=None):
        if sess is not None:
            print('Loading model weights from: ' + self.config.MODEL_LOAD_PATH)
            self.saver.restore(sess, self.config.MODEL_LOAD_PATH)
            print('Done')

    def _get_vocab_embedding_as_np_array(self, vocab_type: VocabType) -> np.ndarray:
        assert vocab_type in VocabType
        vocab_tf_variable_name = self.vocab_type_to_tf_variable_name_mapping[vocab_type]
        with tf.variable_scope('model', reuse=None):
            embeddings = tf.get_variable(vocab_tf_variable_name)
            self.saver = tf.train.Saver()
            self._load_inner_model(self.sess)
            vocab_embedding_matrix = self.sess.run(embeddings)
            return vocab_embedding_matrix

    def get_should_reuse_variables(self):
        if self.config.TRAIN_DATA_PATH_PREFIX:
            return True
        else:
            return None

    def update_correct_predictions(self, num_correct_predictions, output_file, results):
        for original_name, top_words in results:
            normalized_original_name = common.normalize_word(original_name)
            predicted_something = False
            for i, predicted_word in enumerate(common.filter_impossible_names(top_words)):
                if i == 0:
                    output_file.write('Original: ' + original_name + ', predicted 1st: ' + predicted_word + '\n')
                predicted_something = True
                normalized_suggestion = common.normalize_word(predicted_word)
                if normalized_original_name == normalized_suggestion:
                    output_file.write('\t\t predicted correctly at rank: ' + str(i + 1) + '\n')
                    for j in range(i, self.config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION):
                        num_correct_predictions[j] += 1
                    break
            if not predicted_something:
                output_file.write('No results for predicting: ' + original_name)
        return num_correct_predictions

    @staticmethod
    def calculate_results(true_positive, false_positive, false_negative):
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def update_per_subtoken_statistics(self, results, true_positive, false_positive, false_negative):
        for original_name, top_words in results:
            prediction = common.filter_impossible_names(top_words)[0]
            original_subtokens = common.get_subtokens(original_name)
            predicted_subtokens = common.get_subtokens(prediction)
            for subtok in predicted_subtokens:
                if subtok in original_subtokens:
                    true_positive += 1
                else:
                    false_positive += 1
            for subtok in original_subtokens:
                if not subtok in predicted_subtokens:
                    false_negative += 1
        return true_positive, false_positive, false_negative

    def trace(self, sum_loss, batch_num, multi_batch_start_time):
        multi_batch_elapsed = time.time() - multi_batch_start_time
        avg_loss = sum_loss / (self.num_batches_to_log * self.config.TRAIN_BATCH_SIZE)
        throughput = self.config.TRAIN_BATCH_SIZE * self.num_batches_to_log / \
                     (multi_batch_elapsed if multi_batch_elapsed > 0 else 1)
        print('Average loss at batch %d: %f, \tthroughput: %d samples/sec' % (
            batch_num, avg_loss, throughput))

    @staticmethod
    def trace_evaluation(output_file, correct_predictions, total_predictions, elapsed):
        state_message = 'Evaluated %d examples...' % total_predictions
        throughput_message = "Prediction throughput: %d samples/sec" % int(
            total_predictions / (elapsed if elapsed > 0 else 1))
        print(state_message)
        print(throughput_message)

    def close_session(self):
        self.sess.close()

    def _initialize_session_variables(self):
        self.sess.run(tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))
        print('Initalized variables')
