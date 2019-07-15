from itertools import chain
from typing import Optional, Dict, Iterable, Set, NamedTuple, Type
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
    def __init__(self, words: Iterable[str], special_words: SpecialVocabWordsType):
        if special_words is None:
            special_words = Namespace()

        self.word_to_index: Dict[str, int] = {}
        self.index_to_word: Dict[int, str] = {}
        self._word_to_index_lookup_table = None
        self._index_to_word_lookup_table = None
        self.special_words: SpecialVocabWordsType = special_words

        for index, word in enumerate(words):
            self.word_to_index[word] = index
            self.index_to_word[index] = word

        self.size = len(self.word_to_index)

    def save_to_file(self, file):
        pickle.dump((self.word_to_index, self.index_to_word), file)

    @classmethod
    def load_from_file(cls, file, special_words: SpecialVocabWordsType) -> 'Vocab':
        word_to_index, index_to_word = pickle.load(file)
        vocab = cls([], special_words)
        vocab.word_to_index = word_to_index
        vocab.index_to_word = index_to_word
        vocab.size = len(word_to_index)
        return vocab

    @classmethod
    def create_from_freq_dict(cls, word_to_count: Dict[str, int], max_size: int,
                              special_words: Optional[SpecialVocabWordsType] = None):
        if special_words is None:
            special_words = Namespace()
        sorted_counts = sorted(word_to_count, key=word_to_count.get, reverse=True)
        limited_sorted = sorted_counts[:max_size]
        all_words = chain(common.get_unique_list(special_words.__dict__.values()), limited_sorted)
        return cls(all_words, special_words)

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
        self._already_saved_in_paths: Set[str] = set()
        self._load_or_create()

    def _load_or_create(self):
        vocabularies_load_path = None
        if not self.config.is_training or self.config.is_loading:
            vocabularies_load_path = self.config.get_vocabularies_path_from_model_path(self.config.MODEL_LOAD_PATH)
            if not os.path.isfile(vocabularies_load_path):
                vocabularies_load_path = None
        if vocabularies_load_path is None:
            self._create_from_word_freq_dict()
        else:
            self._load_from_path(vocabularies_load_path)

    def _load_from_path(self, vocabularies_load_path: str):
        assert os.path.exists(vocabularies_load_path)
        self.config.log('Loading model vocabularies from: `%s` ... ' % vocabularies_load_path)
        with open(vocabularies_load_path, 'rb') as file:
            self.token_vocab = Vocab.load_from_file(file, self._get_special_words_by_vocab_type(VocabType.Token))
            self.path_vocab = Vocab.load_from_file(file, self._get_special_words_by_vocab_type(VocabType.Path))
            self.target_vocab = Vocab.load_from_file(file, self._get_special_words_by_vocab_type(VocabType.Target))
        self.config.log('Done loading model vocabularies.')
        self._already_saved_in_paths.add(vocabularies_load_path)

    # @classmethod
    # def __load_or_create(cls, config: Config) -> 'Code2VecVocabs':
    #     format_type = 'new'
    #     vocabularies_load_path = None
    #     if not config.is_training or config.is_loading:
    #         vocabularies_load_path = config.get_vocabularies_path_from_model_path(config.MODEL_LOAD_PATH)
    #         if not os.path.isfile(vocabularies_load_path):
    #             vocabularies_load_path = vocabularies_load_path.replace('vocabularies', 'dictionaries')
    #             if os.path.isfile(vocabularies_load_path):
    #                 format_type = 'old'
    #             else:
    #                 vocabularies_load_path = None
    #     if vocabularies_load_path is None:
    #         return cls.create(config)
    #     elif format_type == 'new':
    #         return cls.load(vocabularies_load_path)
    #     elif format_type == 'old':
    #         return cls.__load_old_format(vocabularies_load_path)
    #     assert False
    #
    # @classmethod
    # def __load_old_format(cls, vocabularies_load_path: str) -> 'Code2VecVocabs':
    #     with open(vocabularies_load_path, 'rb') as file:
    #         print('Loading model vocabularies from: `%s` ... ' % vocabularies_load_path, end='')
    #         token_vocab = Vocab([])
    #         path_vocab = Vocab([])
    #         target_vocab = Vocab([])
    #
    #         for vocab, vocab_type in ((token_vocab, VocabType.Token), (target_vocab, VocabType.Target), (path_vocab, VocabType.Path)):
    #             vocab.word_to_index = pickle.load(file)
    #             vocab.index_to_word = pickle.load(file)
    #             _ = pickle.load(file)
    #             assert SpecialVocabWords.OOV not in vocab.word_to_index
    #             assert 0 not in vocab.index_to_word
    #             vocab.word_to_index[SpecialVocabWords.OOV] = 0
    #             vocab.index_to_word[0] = SpecialVocabWords.OOV
    #             vocab.size = len(vocab.word_to_index)
    #             vocab.special_words = self._get_special_words_by_vocab_type(vocab_type)
    #
    #     print('Done')
    #     vocabs = cls(token_vocab, path_vocab, target_vocab)
    #     vocabs._already_saved_in_paths.add(vocabularies_load_path)
    #     return vocabs

    def _create_from_word_freq_dict(self):
        word_freq_dict = self._load_word_freq_dict()
        self.config.log('Word frequencies dictionaries loaded. Now creating vocabularies.')
        self.token_vocab = Vocab.create_from_freq_dict(
            word_freq_dict.token_to_count, self.config.MAX_TOKEN_VOCAB_SIZE,
            special_words=self._get_special_words_by_vocab_type(VocabType.Token))
        self.config.log('Created token vocab. size: %d' % self.token_vocab.size)
        self.path_vocab = Vocab.create_from_freq_dict(
            word_freq_dict.path_to_count, self.config.MAX_PATH_VOCAB_SIZE,
            special_words=self._get_special_words_by_vocab_type(VocabType.Path))
        self.config.log('Created path vocab. size: %d' % self.path_vocab.size)
        self.target_vocab = Vocab.create_from_freq_dict(
            word_freq_dict.target_to_count, self.config.MAX_TARGET_VOCAB_SIZE,
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
            self.path_vocab.save_to_file(file)
            self.target_vocab.save_to_file(file)
        self._already_saved_in_paths.add(vocabularies_save_path)

    def _load_word_freq_dict(self) -> Code2VecWordFreqDicts:
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
