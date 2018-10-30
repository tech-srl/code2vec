import tensorflow as tf
import common

no_such_word = 'NOSUCH'
no_such_composite = no_such_word + ',' + no_such_word + ',' + no_such_word


class PathContextReader:
    class_word_table = None
    class_target_word_table = None
    class_path_table = None

    def __init__(self, word_to_index, target_word_to_index, path_to_index, config, is_evaluating=False):
        self.file_path = config.TEST_PATH if is_evaluating else (config.TRAIN_PATH + '.train.c2v')
        self.batch_size = config.TEST_BATCH_SIZE if is_evaluating else min(config.BATCH_SIZE, config.NUM_EXAMPLES)
        self.num_epochs = config.NUM_EPOCHS
        self.reading_batch_size = config.READING_BATCH_SIZE if is_evaluating else min(config.READING_BATCH_SIZE, config.NUM_EXAMPLES)
        self.num_batching_threads = config.NUM_BATCHING_THREADS
        self.batch_queue_size = config.BATCH_QUEUE_SIZE
        self.data_num_contexts = config.MAX_CONTEXTS
        self.max_contexts = config.MAX_CONTEXTS
        self.is_evaluating = is_evaluating

        self.word_table = PathContextReader.get_word_table(word_to_index)
        self.target_word_table = PathContextReader.get_target_word_table(target_word_to_index)
        self.path_table = PathContextReader.get_path_table(path_to_index)
        self.filtered_output = self.get_filtered_input()

    @classmethod
    def get_word_table(cls, word_to_index):
        if cls.class_word_table is None:
            cls.class_word_table = cls.initalize_hash_map(word_to_index, 0)
        return cls.class_word_table

    @classmethod
    def get_target_word_table(cls, target_word_to_index):
        if cls.class_target_word_table is None:
            cls.class_target_word_table = cls.initalize_hash_map(target_word_to_index, 0)
        return cls.class_target_word_table

    @classmethod
    def get_path_table(cls, path_to_index):
        if cls.class_path_table is None:
            cls.class_path_table = cls.initalize_hash_map(path_to_index, 0)
        return cls.class_path_table

    @classmethod
    def initalize_hash_map(cls, word_to_index, default_value):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(list(word_to_index.keys()), list(word_to_index.values()),
                                                        key_dtype=tf.string,
                                                        value_dtype=tf.int32), default_value)

    def get_input_placeholder(self):
        return self.input_placeholder

    def start(self, session, data_lines=None):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=session, coord=self.coord)
        return self

    def read_file(self):
        row = self.get_row_input()
        record_defaults = [[no_such_composite]] * (self.data_num_contexts + 1)
        row_parts = tf.decode_csv(row, record_defaults=record_defaults, field_delim=' ')
        word = row_parts[0]  # (batch, )
        contexts = tf.stack(row_parts[1:(self.max_contexts + 1)], axis=1)  # (batch, max_contexts)

        flat_contexts = tf.reshape(contexts, [-1])  # (batch * max_contexts, )
        split_contexts = tf.string_split(flat_contexts, delimiter=',')
        dense_split_contexts = tf.reshape(tf.sparse_tensor_to_dense(split_contexts,
                                                                    default_value=no_such_word),
                                          shape=[-1, self.max_contexts, 3])  # (batch, max_contexts, 3)

        if self.is_evaluating:
            target_word_label = word  # (batch, ) of string
        else:
            target_word_label = self.target_word_table.lookup(word)  # (batch, ) of int

        path_source_strings = tf.slice(dense_split_contexts, [0, 0, 0], [-1, self.max_contexts, 1])
        path_source_indices = self.word_table.lookup(path_source_strings)  # (batch, max_contexts, 1)
        path_strings = tf.slice(dense_split_contexts, [0, 0, 1], [-1, self.max_contexts, 1])
        path_indices = self.path_table.lookup(path_strings)  # (batch, max_contexts, 1)
        path_target_strings = tf.slice(dense_split_contexts, [0, 0, 2], [-1, self.max_contexts, 1])
        path_target_indices = self.word_table.lookup(path_target_strings)  # (batch, max_contexts, 1)

        return target_word_label, path_source_indices, path_target_indices, path_indices, \
               path_source_strings, path_strings, path_target_strings

    def get_row_input(self):
        if self.is_evaluating:  # test, read from queue (small data)
            row = self.input_placeholder = tf.placeholder(tf.string)
        else:  # training, read from file
            filename_queue = tf.train.string_input_producer([self.file_path], num_epochs=self.num_epochs, shuffle=False)
            reader = tf.TextLineReader()
            _, row = reader.read_up_to(filename_queue, num_records=self.reading_batch_size)
        return row

    def input_tensors(self):
        return self.initialize_batch_outputs(self.filtered_output[:-3])

    def get_filtered_batches(self):
        return self.filtered_output

    def initialize_batch_outputs(self, filtered_input):
        return tf.train.shuffle_batch(filtered_input,
                                      batch_size=self.batch_size,
                                      enqueue_many=True,
                                      capacity=self.batch_queue_size,
                                      min_after_dequeue=int(self.batch_queue_size * 0.85),
                                      num_threads=self.num_batching_threads,
                                      allow_smaller_final_batch=True)

    def get_filtered_input(self):
        word_label, path_source_indices, path_target_indices, path_indices, \
        source_strings, path_strings, target_strings = self.read_file()
        any_contexts_is_valid = tf.logical_or(
            tf.greater(tf.squeeze(tf.reduce_max(path_source_indices, 1), axis=1), 0),
            tf.logical_or(
                tf.greater(tf.squeeze(tf.reduce_max(path_target_indices, 1), axis=1), 0),
                tf.greater(tf.squeeze(tf.reduce_max(path_indices, 1), axis=1), 0))
        )  # (batch, )

        if self.is_evaluating:
            cond = tf.where(any_contexts_is_valid)
        else:  # training
            word_is_valid = tf.greater(word_label, 0)  # (batch, )
            cond = tf.where(tf.logical_and(word_is_valid, any_contexts_is_valid))  # (batch, 1)
        valid_mask = tf.to_float(  # (batch, max_contexts, 1)
            tf.logical_or(tf.logical_or(tf.greater(path_source_indices, 0),
                                        tf.greater(path_target_indices, 0)),
                          tf.greater(path_indices, 0))
        )

        filtered = \
            tf.gather(word_label, cond), \
            tf.squeeze(tf.gather(path_source_indices, cond), [1, 3]), \
            tf.squeeze(tf.gather(path_indices, cond), [1, 3]), \
            tf.squeeze(tf.gather(path_target_indices, cond), [1, 3]), \
            tf.squeeze(tf.gather(valid_mask, cond), [1, 3]), \
            tf.squeeze(tf.gather(source_strings, cond)), \
            tf.squeeze(tf.gather(path_strings, cond)), \
            tf.squeeze(tf.gather(target_strings, cond))  # (batch, max_contexts)

        return filtered

    def __enter__(self):
        return self

    def should_stop(self):
        return self.coord.should_stop()

    def __exit__(self, type, value, traceback):
        print('Reader stopping')
        self.coord.request_stop()
        self.coord.join(self.threads)
