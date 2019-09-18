from itertools import chain
from typing import Optional, Dict, Iterable, Set, NamedTuple
import pickle
import os
from enum import Enum
from config import Config
import tensorflow as tf
from argparse import Namespace

from common import common


class VocabType(Enum):
    Token = 1
    Target = 2
    Path = 3


SpecialVocabWordsType = Namespace


_SpecialVocabWords_OnlyOov = Namespace(
    OOV='<OOV>'
)

_SpecialVocabWords_SeparateOovPad = Namespace(
    PAD='<PAD>',
    OOV='<OOV>'
)

_SpecialVocabWords_JoinedOovPad = Namespace(
    PAD_OR_OOV='<PAD_OR_OOV>',
    PAD='<PAD_OR_OOV>',
    OOV='<PAD_OR_OOV>'
)


class Vocab:
    def __init__(self, vocab_type: VocabType, words: Iterable[str],
                 special_words: Optional[SpecialVocabWordsType] = None):
        if special_words is None:
            special_words = Namespace()

        self.vocab_type = vocab_type
        self.word_to_index: Dict[str, int] = {}
        self.index_to_word: Dict[int, str] = {}
        self._word_to_index_lookup_table = None
        self._index_to_word_lookup_table = None
        self.special_words: SpecialVocabWordsType = special_words

        for index, word in enumerate(chain(common.get_unique_list(special_words.__dict__.values()), words)):
            self.word_to_index[word] = index
            self.index_to_word[index] = word

        self.size = len(self.word_to_index)

    def save_to_file(self, file):
        # Notice: From historical reasons, a saved vocab doesn't include special words.
        special_words_as_unique_list = common.get_unique_list(self.special_words.__dict__.values())
        nr_special_words = len(special_words_as_unique_list)
        word_to_index_wo_specials = {word: idx for word, idx in self.word_to_index.items() if idx >= nr_special_words}
        index_to_word_wo_specials = {idx: word for idx, word in self.index_to_word.items() if idx >= nr_special_words}
        size_wo_specials = self.size - nr_special_words
        pickle.dump(word_to_index_wo_specials, file)
        pickle.dump(index_to_word_wo_specials, file)
        pickle.dump(size_wo_specials, file)

    @classmethod
    def load_from_file(cls, vocab_type: VocabType, file, special_words: SpecialVocabWordsType) -> 'Vocab':
        special_words_as_unique_list = common.get_unique_list(special_words.__dict__.values())

        # Notice: From historical reasons, a saved vocab doesn't include special words,
        #         so they should be added upon loading.

        word_to_index_wo_specials = pickle.load(file)
        index_to_word_wo_specials = pickle.load(file)
        size_wo_specials = pickle.load(file)
        assert len(index_to_word_wo_specials) == len(word_to_index_wo_specials) == size_wo_specials
        min_word_idx_wo_specials = min(index_to_word_wo_specials.keys())

        if min_word_idx_wo_specials != len(special_words_as_unique_list):
            raise ValueError(
                "Error while attempting to load vocabulary `{vocab_type}` from file `{file_path}`. "
                "The stored vocabulary has minimum word index {min_word_idx}, "
                "while expecting minimum word index to be {nr_special_words} "
                "because having to use {nr_special_words} special words, which are: {special_words}. "
                "Please check the parameter `config.SEPARATE_OOV_AND_PAD`.".format(
                    vocab_type=vocab_type, file_path=file.name, min_word_idx=min_word_idx_wo_specials,
                    nr_special_words=len(special_words_as_unique_list), special_words=special_words))

        vocab = cls(vocab_type, [], special_words)
        vocab.word_to_index = {**word_to_index_wo_specials,
                               **{word: idx for idx, word in enumerate(special_words_as_unique_list)}}
        vocab.index_to_word = {**index_to_word_wo_specials,
                               **{idx: word for idx, word in enumerate(special_words_as_unique_list)}}
        vocab.size = size_wo_specials + len(special_words_as_unique_list)
        return vocab

    @classmethod
    def create_from_freq_dict(cls, vocab_type: VocabType, word_to_count: Dict[str, int], max_size: int,
                              special_words: Optional[SpecialVocabWordsType] = None):
        if special_words is None:
            special_words = Namespace()
        words_sorted_by_counts = sorted(word_to_count, key=word_to_count.get, reverse=True)
        words_sorted_by_counts_and_limited = words_sorted_by_counts[:max_size]
        return cls(vocab_type, words_sorted_by_counts_and_limited, special_words)

    @staticmethod
    def _create_word_to_index_lookup_table(word_to_index: Dict[str, int], default_value: int):
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                list(word_to_index.keys()), list(word_to_index.values()), key_dtype=tf.string, value_dtype=tf.int32),
            default_value=tf.constant(default_value, dtype=tf.int32))

    @staticmethod
    def _create_index_to_word_lookup_table(index_to_word: Dict[int, str], default_value: str) \
            -> tf.lookup.StaticHashTable:
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                list(index_to_word.keys()), list(index_to_word.values()), key_dtype=tf.int32, value_dtype=tf.string),
            default_value=tf.constant(default_value, dtype=tf.string))

    def get_word_to_index_lookup_table(self) -> tf.lookup.StaticHashTable:
        if self._word_to_index_lookup_table is None:
            self._word_to_index_lookup_table = self._create_word_to_index_lookup_table(
                self.word_to_index, default_value=self.word_to_index[self.special_words.OOV])
        return self._word_to_index_lookup_table

    def get_index_to_word_lookup_table(self) -> tf.lookup.StaticHashTable:
        if self._index_to_word_lookup_table is None:
            self._index_to_word_lookup_table = self._create_index_to_word_lookup_table(
                self.index_to_word, default_value=self.special_words.OOV)
        return self._index_to_word_lookup_table

    def lookup_index(self, word: tf.Tensor) -> tf.Tensor:
        return self.get_word_to_index_lookup_table().lookup(word)

    def lookup_word(self, index: tf.Tensor) -> tf.Tensor:
        return self.get_index_to_word_lookup_table().lookup(index)


WordFreqDictType = Dict[str, int]


class Code2VecWordFreqDicts(NamedTuple):
    token_to_count: WordFreqDictType
    path_to_count: WordFreqDictType
    target_to_count: WordFreqDictType


class Code2VecVocabs:
    def __init__(self, config: Config):
        self.config = config
        self.token_vocab: Optional[Vocab] = None
        self.path_vocab: Optional[Vocab] = None
        self.target_vocab: Optional[Vocab] = None

        # Used to avoid re-saving a non-modified vocabulary to a path it is already saved in.
        self._already_saved_in_paths: Set[str] = set()

        self._load_or_create()

    def _load_or_create(self):
        assert self.config.is_training or self.config.is_loading
        if self.config.is_loading:
            vocabularies_load_path = self.config.get_vocabularies_path_from_model_path(self.config.MODEL_LOAD_PATH)
            if not os.path.isfile(vocabularies_load_path):
                raise ValueError(
                    "Model dictionaries file is not found in model load dir. "
                    "Expecting file `{vocabularies_load_path}`.".format(vocabularies_load_path=vocabularies_load_path))
            self._load_from_path(vocabularies_load_path)
        else:
            self._create_from_word_freq_dict()

    def _load_from_path(self, vocabularies_load_path: str):
        assert os.path.exists(vocabularies_load_path)
        self.config.log('Loading model vocabularies from: `%s` ... ' % vocabularies_load_path)
        with open(vocabularies_load_path, 'rb') as file:
            self.token_vocab = Vocab.load_from_file(
                VocabType.Token, file, self._get_special_words_by_vocab_type(VocabType.Token))
            self.target_vocab = Vocab.load_from_file(
                VocabType.Target, file, self._get_special_words_by_vocab_type(VocabType.Target))
            self.path_vocab = Vocab.load_from_file(
                VocabType.Path, file, self._get_special_words_by_vocab_type(VocabType.Path))
        self.config.log('Done loading model vocabularies.')
        self._already_saved_in_paths.add(vocabularies_load_path)

    def _create_from_word_freq_dict(self):
        word_freq_dict = self._load_word_freq_dict()
        self.config.log('Word frequencies dictionaries loaded. Now creating vocabularies.')
        self.token_vocab = Vocab.create_from_freq_dict(
            VocabType.Token, word_freq_dict.token_to_count, self.config.MAX_TOKEN_VOCAB_SIZE,
            special_words=self._get_special_words_by_vocab_type(VocabType.Token))
        self.config.log('Created token vocab. size: %d' % self.token_vocab.size)
        self.path_vocab = Vocab.create_from_freq_dict(
            VocabType.Path, word_freq_dict.path_to_count, self.config.MAX_PATH_VOCAB_SIZE,
            special_words=self._get_special_words_by_vocab_type(VocabType.Path))
        self.config.log('Created path vocab. size: %d' % self.path_vocab.size)
        self.target_vocab = Vocab.create_from_freq_dict(
            VocabType.Target, word_freq_dict.target_to_count, self.config.MAX_TARGET_VOCAB_SIZE,
            special_words=self._get_special_words_by_vocab_type(VocabType.Target))
        self.config.log('Created target vocab. size: %d' % self.target_vocab.size)

    def _get_special_words_by_vocab_type(self, vocab_type: VocabType) -> SpecialVocabWordsType:
        if not self.config.SEPARATE_OOV_AND_PAD:
            return _SpecialVocabWords_JoinedOovPad
        if vocab_type == VocabType.Target:
            return _SpecialVocabWords_OnlyOov
        return _SpecialVocabWords_SeparateOovPad

    def save(self, vocabularies_save_path: str):
        if vocabularies_save_path in self._already_saved_in_paths:
            return
        with open(vocabularies_save_path, 'wb') as file:
            self.token_vocab.save_to_file(file)
            self.target_vocab.save_to_file(file)
            self.path_vocab.save_to_file(file)
        self._already_saved_in_paths.add(vocabularies_save_path)

    def _load_word_freq_dict(self) -> Code2VecWordFreqDicts:
        assert self.config.is_training
        self.config.log('Loading word frequencies dictionaries from: %s ... ' % self.config.word_freq_dict_path)
        with open(self.config.word_freq_dict_path, 'rb') as file:
            token_to_count = pickle.load(file)
            path_to_count = pickle.load(file)
            target_to_count = pickle.load(file)
        self.config.log('Done loading word frequencies dictionaries.')
        # assert all(isinstance(item, WordFreqDictType) for item in {token_to_count, path_to_count, target_to_count})
        return Code2VecWordFreqDicts(
            token_to_count=token_to_count, path_to_count=path_to_count, target_to_count=target_to_count)

    def get(self, vocab_type: VocabType) -> Vocab:
        if not isinstance(vocab_type, VocabType):
            raise ValueError('`vocab_type` should be `VocabType.Token`, `VocabType.Target` or `VocabType.Path`.')
        if vocab_type == VocabType.Token:
            return self.token_vocab
        if vocab_type == VocabType.Target:
            return self.target_vocab
        if vocab_type == VocabType.Path:
            return self.path_vocab
