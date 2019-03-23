import tensorflow as tf
from collections import namedtuple
from typing import Dict, Tuple, NamedTuple, Union, Optional, Type
from config import Config
from common import common
import abc

NO_SUCH_CONTEXT = ','.join([common.SpecialDictWords.NoSuchWord.name] * 3)


PathContextInputTensors__old = namedtuple('PathContextInputTensors',
    ['target_label', 'path_source_indices', 'path_indices', 'path_target_indices',
     'valid_mask', 'path_source_strings', 'path_strings', 'path_target_strings'])


class ReaderInputTensors(NamedTuple):
    path_source_indices: tf.Tensor
    path_indices: tf.Tensor
    path_target_indices: tf.Tensor
    context_valid_mask: tf.Tensor
    target_index: Optional[tf.Tensor] = None
    target_string: Optional[tf.Tensor] = None


class ModelInputTensorsFormer(abc.ABC):
    @abc.abstractmethod
    def to_model_input_form(self, input_tensors: ReaderInputTensors):
        ...

    @abc.abstractmethod
    def from_model_input_form(self, input_row) -> ReaderInputTensors:
        ...


class PathContextReader:
    class_token_table = None
    class_target_word_table = None
    class_path_table = None

    def __init__(self,
                 token_to_index: Dict[str, int],
                 target_to_index: Dict[str, int],
                 path_to_index: Dict[str, int],
                 config: Config,
                 model_input_tensors_former: ModelInputTensorsFormer,
                 is_evaluating: bool = False):
        self.file_path = config.TEST_PATH if is_evaluating else (config.TRAIN_PATH + '.train.c2v')
        self.batch_size = config.TEST_BATCH_SIZE if is_evaluating else min(config.BATCH_SIZE, config.NUM_EXAMPLES)
        self.config = config
        self.model_input_tensors_former = model_input_tensors_former
        self.is_evaluating = is_evaluating
        self.record_defaults = [[NO_SUCH_CONTEXT]] * (self.config.MAX_CONTEXTS + 1)

        self.token_table = PathContextReader.get_token_table(token_to_index)
        self.target_table = PathContextReader.get_target_word_table(target_to_index)
        self.path_table = PathContextReader.get_path_table(path_to_index)

        self._dataset = self.create_dataset_pipeline()

    @classmethod
    def get_token_table(cls, token_to_index: Dict[str, int]):
        if cls.class_token_table is None:
            cls.class_token_table = cls.initalize_hash_map(
                token_to_index, default_value=common.SpecialDictWords.NoSuchWord.value)
        return cls.class_token_table

    @classmethod
    def get_target_word_table(cls, target_to_index: Dict[str, int]):
        if cls.class_target_word_table is None:
            cls.class_target_word_table = cls.initalize_hash_map(
                target_to_index, default_value=common.SpecialDictWords.NoSuchWord.value)
        return cls.class_target_word_table

    @classmethod
    def get_path_table(cls, path_to_index: Dict[str, int]):
        if cls.class_path_table is None:
            cls.class_path_table = cls.initalize_hash_map(
                path_to_index, default_value=common.SpecialDictWords.NoSuchWord.value)
        return cls.class_path_table

    @classmethod
    def initalize_hash_map(cls, word_to_index: Dict[str, int], default_value: int):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(
                list(word_to_index.keys()), list(word_to_index.values()), key_dtype=tf.string, value_dtype=tf.int32),
            default_value=tf.constant(default_value, dtype=tf.int32))

    def process_from_placeholder(self, row):
        parts = tf.io.decode_csv(row, record_defaults=self.record_defaults, field_delim=' ', use_quote_delim=False)
        return self.process_dataset(*parts)

    @property
    def dataset(self):
        return self._dataset

    def create_dataset_pipeline(self) -> tf.data.Dataset:
        dataset = tf.data.experimental.CsvDataset(
            self.file_path, record_defaults=self.record_defaults, field_delim=' ',
            use_quote_delim=False, buffer_size=self.config.CSV_BUFFER_SIZE)

        if not self.is_evaluating:
            if self.config.NUM_EPOCHS > 1:
                dataset = dataset.repeat(self.config.NUM_EPOCHS)
            dataset = dataset.shuffle(self.config.SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)

        dataset = dataset.map(self.process_dataset, num_parallel_calls=self.config.READER_NUM_PARALLEL_BATCHES)
        dataset = dataset.filter(self.filter_dataset)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset

    def filter_dataset(self, *row_parts) -> tf.bool:
        row_parts = self.model_input_tensors_former.from_model_input_form(row_parts)

        assert all(tensor.shape == (self.config.MAX_CONTEXTS,) for tensor in
                   {row_parts.path_source_indices, row_parts.path_indices,
                    row_parts.path_target_indices, row_parts.context_valid_mask})

        any_contexts_is_valid = tf.logical_or(
            tf.greater(tf.reduce_max(row_parts.path_source_indices, axis=0), 0),
            tf.logical_or(
                tf.greater(tf.reduce_max(row_parts.path_target_indices, axis=0), 0),
                tf.greater(tf.reduce_max(row_parts.path_indices, axis=0), 0))
        )  # scalar

        if self.is_evaluating:
            cond = any_contexts_is_valid  # scalar
        else:  # training
            word_is_valid = tf.greater(row_parts.target_index, common.SpecialDictWords.NoSuchWord.value)  # scalar
            cond = tf.logical_and(word_is_valid, any_contexts_is_valid)  # scalar

        return cond  # scalar

    def process_dataset(self, *row_parts) -> Tuple[Union[tf.Tensor, Tuple[tf.Tensor, ...], Dict[str, tf.Tensor]], ...]:
        row_parts = list(row_parts)
        target_str = row_parts[0]
        target_index = self.target_table.lookup(target_str)
        # target_word_label = target_str if self.is_evaluating else target_index

        contexts_str = tf.stack(row_parts[1:(self.config.MAX_CONTEXTS + 1)], axis=0)
        split_contexts = tf.string_split(contexts_str, delimiter=',')
        dense_split_contexts = tf.sparse_tensor_to_dense(split_contexts,
                                                         default_value=common.SpecialDictWords.NoSuchWord.name)

        path_source_strings = tf.squeeze(tf.slice(dense_split_contexts, [0, 0], [self.config.MAX_CONTEXTS, 1]), axis=1)
        path_source_indices = self.token_table.lookup(path_source_strings)  # (max_contexts, )
        path_strings = tf.squeeze(tf.slice(dense_split_contexts, [0, 1], [self.config.MAX_CONTEXTS, 1]), axis=1)
        path_indices = self.path_table.lookup(path_strings)  # (max_contexts, )
        path_target_strings = tf.squeeze(tf.slice(dense_split_contexts, [0, 2], [self.config.MAX_CONTEXTS, 1]), axis=1)
        path_target_indices = self.token_table.lookup(path_target_strings)  # (max_contexts, )

        context_valid_mask = tf.cast(
            tf.logical_or(tf.logical_or(tf.greater(path_source_indices, 0),
                                        tf.greater(path_target_indices, 0)),
                          tf.greater(path_indices, 0)),
            dtype=tf.float32
        )  # (max_contexts, )

        assert all(tensor.shape == (self.config.MAX_CONTEXTS,) for tensor in
                   {path_source_indices, path_indices, path_target_indices, context_valid_mask})

        tensors = ReaderInputTensors(
            path_source_indices=path_source_indices,
            path_indices=path_indices,
            path_target_indices=path_target_indices,
            context_valid_mask=context_valid_mask,
            target_index=target_index,
            target_string=target_str
        )

        return self.model_input_tensors_former.to_model_input_form(tensors)
