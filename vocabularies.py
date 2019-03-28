from itertools import chain
from typing import Optional, Dict, Iterable, Tuple
import pickle
from config import Config
import tensorflow as tf


class SpecialVocabWords:
    PAD = '<PAD>'  # padding
    OOV = '<OOV>'  # out-of-vocabulary


class Vocab:
    def __init__(self, words: Optional[Iterable[str]] = None):
        self.word_to_index: Dict[str, int] = {}
        self.index_to_word: Dict[int, str] = {}
        self._word_to_index_lookup_table = None
        self._index_to_word_lookup_table = None

        for index, word in enumerate(words):
            self.word_to_index[word] = index
            self.index_to_word[index] = word

        self.size = len(self.word_to_index)

    @classmethod
    def create_from_freq_dict(cls, word_to_count: Dict[str, int], max_size: int,
                              special_words: Optional[Iterable[str]] = None):
        sorted_counts = sorted(word_to_count, key=word_to_count.get, reverse=True)
        limited_sorted = sorted_counts[:max_size]
        all_words = chain(special_words, limited_sorted)
        return cls(all_words)

    @staticmethod
    def _create_word_to_index_lookup_table(word_to_index: Dict[str, int], default_value: int):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(
                list(word_to_index.keys()), list(word_to_index.values()), key_dtype=tf.string, value_dtype=tf.int32),
            default_value=tf.constant(default_value, dtype=tf.int32))

    @staticmethod
    def _create_index_to_word_lookup_table(index_to_word: Dict[int, str], default_value: str) \
            -> tf.contrib.lookup.HashTable:
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(
                list(index_to_word.keys()), list(index_to_word.values()), key_dtype=tf.int32, value_dtype=tf.string),
            default_value=tf.constant(default_value, dtype=tf.string))

    def get_word_to_index_lookup_table(self) -> tf.contrib.lookup.HashTable:
        if self._word_to_index_lookup_table is None:
            self._word_to_index_lookup_table = self._create_word_to_index_lookup_table(
                self.word_to_index, default_value=self.word_to_index[SpecialVocabWords.OOV])
        return self._word_to_index_lookup_table

    def get_index_to_word_lookup_table(self) -> tf.contrib.lookup.HashTable:
        if self._index_to_word_lookup_table is None:
            self._index_to_word_lookup_table = self._create_index_to_word_lookup_table(
                self.index_to_word, default_value=SpecialVocabWords.OOV)
        return self._index_to_word_lookup_table

    def lookup_index(self, word: tf.Tensor) -> tf.Tensor:
        return self.get_word_to_index_lookup_table().lookup(word)

    def lookup_word(self, index: tf.Tensor) -> tf.Tensor:
        return self.get_index_to_word_lookup_table().lookup(index)


WordFreqDictType = Dict[str, int]


class Code2VecVocabs:
    def __init__(self, token_vocab: Vocab, path_vocab: Vocab, target_vocab: Vocab):
        self.token_vocab: Vocab = token_vocab
        self.path_vocab: Vocab = path_vocab
        self.target_vocab: Vocab = target_vocab

    @classmethod
    def load_or_create(cls, config: Config) -> 'Code2VecVocabs':
        if config.TRAIN_DATA_PATH and not config.MODEL_LOAD_PATH:
            return cls.create(config)
        else:
            return cls.load(config.get_vocabularies_path_from_model_path(config.MODEL_LOAD_PATH))

    @classmethod
    def load(cls, vocabularies_load_path: str) -> 'Code2VecVocabs':
        with open(vocabularies_load_path, 'rb') as file:
            print('Loading model vocabularies from: %s ... ' % vocabularies_load_path, end='')
            token_vocab = pickle.load(file)
            path_vocab = pickle.load(file)
            target_vocab = pickle.load(file)
            print('Done')
            return cls(token_vocab, path_vocab, target_vocab)

    @classmethod
    def create(cls, config: Config) -> 'Code2VecVocabs':
        token_to_count, path_to_count, target_to_count = cls._load_word_freq_dict(config.word_freq_dict_path)
        print('Word frequencies dictionaries loaded. Now creating vocabularies.')
        token_vocab = Vocab.create_from_freq_dict(
            token_to_count, config.MAX_TOKEN_VOCAB_SIZE, special_words=[SpecialVocabWords.PAD, SpecialVocabWords.OOV])
        print('Created token vocab. size: %d' % token_vocab.size)
        path_vocab = Vocab.create_from_freq_dict(
            path_to_count, config.MAX_PATH_VOCAB_SIZE, special_words=[SpecialVocabWords.PAD, SpecialVocabWords.OOV])
        print('Created path vocab. size: %d' % path_vocab.size)
        target_vocab = Vocab.create_from_freq_dict(
            target_to_count, config.MAX_TARGET_VOCAB_SIZE, special_words=[SpecialVocabWords.OOV])
        print('Created target vocab. size: %d' % target_vocab.size)
        return cls(token_vocab, path_vocab, target_vocab)

    def save(self, vocabularies_save_path: str):
        with open(vocabularies_save_path, 'wb') as file:
            pickle.dump(self.token_vocab, file)
            pickle.dump(self.path_vocab, file)
            pickle.dump(self.target_vocab, file)

    @staticmethod
    def _load_word_freq_dict(path: str) -> Tuple[WordFreqDictType, WordFreqDictType, WordFreqDictType]:
        print('Loading word frequencies dictionaries from: %s ... ' % path, end='')
        with open(path, 'rb') as file:
            token_to_count = pickle.load(file)
            path_to_count = pickle.load(file)
            target_to_count = pickle.load(file)
        print('Done.')
        # assert all(isinstance(item, WordFreqDictType) for item in {token_to_count, path_to_count, target_to_count})
        return token_to_count, path_to_count, target_to_count
