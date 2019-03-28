import tensorflow as tf
from typing import Dict, Tuple, NamedTuple, Union, Optional
from config import Config
from common import common, SpecialVocabWords
from vocabularies import Code2VecVocabs
import abc
from functools import reduce


class ReaderInputTensors(NamedTuple):
    """Used mostly for convenient-and-clear access to input parts."""
    path_source_token_indices: tf.Tensor
    path_indices: tf.Tensor
    path_target_token_indices: tf.Tensor
    context_valid_mask: tf.Tensor
    target_index: Optional[tf.Tensor] = None
    target_string: Optional[tf.Tensor] = None
    path_source_token_strings: Optional[tf.Tensor] = None
    path_strings: Optional[tf.Tensor] = None
    path_target_token_strings: Optional[tf.Tensor] = None


class ModelInputTensorsFormer(abc.ABC):
    """Inherited by the model to set the wanted input parts and its form (as expected by the model input)."""
    @abc.abstractmethod
    def to_model_input_form(self, input_tensors: ReaderInputTensors):
        ...

    @abc.abstractmethod
    def from_model_input_form(self, input_row) -> ReaderInputTensors:
        ...


class PathContextReader:
    CONTEXT_PADDING = ','.join([SpecialVocabWords.PAD] * 3)

    def __init__(self,
                 vocabs: Code2VecVocabs,
                 config: Config,
                 model_input_tensors_former: ModelInputTensorsFormer,
                 is_evaluating: bool = False):
        self.vocabs = vocabs
        self.config = config
        self.model_input_tensors_former = model_input_tensors_former
        self.is_evaluating = is_evaluating
        self.csv_record_defaults = [[SpecialVocabWords.OOV]] + ([[self.CONTEXT_PADDING]] * self.config.MAX_CONTEXTS)

        # initialize the needed lookup tables (if not already initialized).
        self.vocabs.token_vocab.get_word_to_index_lookup_table()
        self.vocabs.path_vocab.get_word_to_index_lookup_table()
        self.vocabs.target_vocab.get_word_to_index_lookup_table()

        self._dataset = self._create_dataset_pipeline()

    def process_from_placeholder(self, row):
        parts = tf.io.decode_csv(row, record_defaults=self.csv_record_defaults, field_delim=' ', use_quote_delim=False)
        # TODO: apply the filter `_filter_input_rows()` here.
        return self._map_raw_dataset_row_to_expected_model_input_form(*parts)

    @property
    def dataset(self):
        return self._dataset

    def _create_dataset_pipeline(self) -> tf.data.Dataset:
        dataset = tf.data.experimental.CsvDataset(
            self.config.data_path(is_evaluating=self.is_evaluating), record_defaults=self.csv_record_defaults,
            field_delim=' ', use_quote_delim=False, buffer_size=self.config.CSV_BUFFER_SIZE)

        if not self.is_evaluating:
            if self.config.NUM_EPOCHS > 1:
                dataset = dataset.repeat(self.config.NUM_EPOCHS)
            dataset = dataset.shuffle(self.config.SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)

        dataset = dataset.map(self._map_raw_dataset_row_to_expected_model_input_form,
                              num_parallel_calls=self.config.READER_NUM_PARALLEL_BATCHES)
        dataset = dataset.filter(self._filter_input_rows)
        dataset = dataset.batch(self.config.batch_size(is_evaluating=self.is_evaluating))
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        return dataset

    def _filter_input_rows(self, *row_parts) -> tf.bool:
        row_parts = self.model_input_tensors_former.from_model_input_form(row_parts)

        assert all(tensor.shape == (self.config.MAX_CONTEXTS,) for tensor in
                   {row_parts.path_source_token_indices, row_parts.path_indices,
                    row_parts.path_target_token_indices, row_parts.context_valid_mask})

        # FIXME: Does "valid" here mean just "no padding" or "neither padding nor OOV"? I assumed just "no padding".
        any_word_valid_mask_per_context_part = [
            tf.not_equal(tf.reduce_max(row_parts.path_source_token_indices, axis=0),
                         self.vocabs.token_vocab.word_to_index[SpecialVocabWords.PAD]),
            tf.not_equal(tf.reduce_max(row_parts.path_target_token_indices, axis=0),
                         self.vocabs.token_vocab.word_to_index[SpecialVocabWords.PAD]),
            tf.not_equal(tf.reduce_max(row_parts.path_indices, axis=0),
                         self.vocabs.path_vocab.word_to_index[SpecialVocabWords.PAD])]
        any_contexts_is_valid = reduce(tf.logical_or, any_word_valid_mask_per_context_part)  # scalar

        if self.is_evaluating:
            cond = any_contexts_is_valid  # scalar
        else:  # training
            word_is_valid = tf.greater(
                row_parts.target_index, self.vocabs.target_vocab.word_to_index[SpecialVocabWords.OOV])  # scalar
            cond = tf.logical_and(word_is_valid, any_contexts_is_valid)  # scalar

        return cond  # scalar

    def _map_raw_dataset_row_to_expected_model_input_form(self, *row_parts) -> \
            Tuple[Union[tf.Tensor, Tuple[tf.Tensor, ...], Dict[str, tf.Tensor]], ...]:
        row_parts = list(row_parts)
        target_str = row_parts[0]
        target_index = self.vocabs.target_vocab.lookup_index(target_str)

        contexts_str = tf.stack(row_parts[1:(self.config.MAX_CONTEXTS + 1)], axis=0)
        split_contexts = tf.string_split(contexts_str, delimiter=',', skip_empty=False)
        # dense_split_contexts = tf.sparse_tensor_to_dense(split_contexts, default_value=SpecialVocabWords.PAD)
        sparse_split_contexts = tf.sparse.SparseTensor(
            indices=split_contexts.indices, values=split_contexts.values, dense_shape=[self.config.MAX_CONTEXTS, 3])
        dense_split_contexts = tf.reshape(
            tf.sparse.to_dense(sp_input=sparse_split_contexts, default_value=SpecialVocabWords.PAD),
            shape=[self.config.MAX_CONTEXTS, 3])  # (max_contexts, 3)

        path_source_token_strings = tf.squeeze(
            tf.slice(dense_split_contexts, begin=[0, 0], size=[self.config.MAX_CONTEXTS, 1]), axis=1)  # (max_contexts,)
        path_strings = tf.squeeze(
            tf.slice(dense_split_contexts, begin=[0, 1], size=[self.config.MAX_CONTEXTS, 1]), axis=1)  # (max_contexts,)
        path_target_token_strings = tf.squeeze(
            tf.slice(dense_split_contexts, begin=[0, 2], size=[self.config.MAX_CONTEXTS, 1]), axis=1)  # (max_contexts,)

        path_source_token_indices = self.vocabs.token_vocab.lookup_index(path_source_token_strings)  # (max_contexts, )
        path_indices = self.vocabs.path_vocab.lookup_index(path_strings)  # (max_contexts, )
        path_target_token_indices = self.vocabs.token_vocab.lookup_index(path_target_token_strings)  # (max_contexts, )

        # FIXME: Does "valid" here mean just "no padding" or "neither padding nor OOV"? I assumed just "no padding".
        valid_word_mask_per_context_part = [
            tf.not_equal(path_source_token_indices, self.vocabs.token_vocab.word_to_index[SpecialVocabWords.PAD]),
            tf.not_equal(path_target_token_indices, self.vocabs.token_vocab.word_to_index[SpecialVocabWords.PAD]),
            tf.not_equal(path_indices, self.vocabs.path_vocab.word_to_index[SpecialVocabWords.PAD])]  # [(max_contexts, )]
        context_valid_mask = tf.cast(reduce(tf.logical_or, valid_word_mask_per_context_part), dtype=tf.float32)  # (max_contexts, )

        assert all(tensor.shape == (self.config.MAX_CONTEXTS,) for tensor in
                   {path_source_token_indices, path_indices, path_target_token_indices, context_valid_mask})

        tensors = ReaderInputTensors(
            path_source_token_indices=path_source_token_indices,
            path_indices=path_indices,
            path_target_token_indices=path_target_token_indices,
            context_valid_mask=context_valid_mask,
            target_index=target_index,
            target_string=target_str,
            path_source_token_strings=path_source_token_strings,
            path_strings=path_strings,
            path_target_token_strings=path_target_token_strings
        )

        return self.model_input_tensors_former.to_model_input_form(tensors)
