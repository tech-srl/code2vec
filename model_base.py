import numpy as np
import abc
import os
from typing import NamedTuple, Optional, List, Dict, Tuple, Iterable

from common import common
from vocabularies import Code2VecVocabs, VocabType
from config import Config


class ModelEvaluationResults(NamedTuple):
    topk_acc: float
    subtoken_precision: float
    subtoken_recall: float
    subtoken_f1: float
    loss: Optional[float] = None

    def __str__(self):
        res_str = 'topk_acc: {topk_acc}, precision: {precision}, recall: {recall}, F1: {f1}'.format(
            topk_acc=self.topk_acc,
            precision=self.subtoken_precision,
            recall=self.subtoken_recall,
            f1=self.subtoken_f1)
        if self.loss is not None:
            res_str = ('loss: {}, '.format(self.loss)) + res_str
        return res_str


class ModelPredictionResults(NamedTuple):
    original_name: str
    topk_predicted_words: np.ndarray
    topk_predicted_words_scores: np.ndarray
    attention_per_context: Dict[Tuple[str, str, str], float]
    code_vector: Optional[np.ndarray] = None


class Code2VecModelBase(abc.ABC):
    def __init__(self, config: Config):
        self.config = config
        self.config.verify()

        self._log_creating_model()

        if not config.RELEASE:
            self._init_num_of_examples()
        self._log_model_configuration()
        self.vocabs = Code2VecVocabs(config)
        self.vocabs.target_vocab.get_index_to_word_lookup_table()  # just to initialize it (if not already initialized)
        self._load_or_create_inner_model()
        self._initialize()

    def _log_creating_model(self):
        self.log('')
        self.log('')
        self.log('---------------------------------------------------------------------')
        self.log('---------------------------------------------------------------------')
        self.log('---------------------- Creating code2vec model ----------------------')
        self.log('---------------------------------------------------------------------')
        self.log('---------------------------------------------------------------------')

    def _log_model_configuration(self):
        self.log('---------------------------------------------------------------------')
        self.log('----------------- Configuration - Hyper Parameters ------------------')
        longest_param_name_len = max(len(param_name) for param_name, _ in self.config)
        for param_name, param_val in self.config:
            self.log('{name: <{name_len}}{val}'.format(
                name=param_name, val=param_val, name_len=longest_param_name_len+2))
        self.log('---------------------------------------------------------------------')

    @property
    def logger(self):
        return self.config.get_logger()

    def log(self, msg):
        self.logger.info(msg)

    def _init_num_of_examples(self):
        self.log('Checking number of examples ...')
        if self.config.is_training:
            self.config.NUM_TRAIN_EXAMPLES = self._get_num_of_examples_for_dataset(self.config.train_data_path)
            self.log('    Number of train examples: {}'.format(self.config.NUM_TRAIN_EXAMPLES))
        if self.config.is_testing:
            self.config.NUM_TEST_EXAMPLES = self._get_num_of_examples_for_dataset(self.config.TEST_DATA_PATH)
            self.log('    Number of test examples: {}'.format(self.config.NUM_TEST_EXAMPLES))

    @staticmethod
    def _get_num_of_examples_for_dataset(dataset_path: str) -> int:
        dataset_num_examples_file_path = dataset_path + '.num_examples'
        if os.path.isfile(dataset_num_examples_file_path):
            with open(dataset_num_examples_file_path, 'r') as file:
                num_examples_in_dataset = int(file.readline())
        else:
            num_examples_in_dataset = common.count_lines_in_file(dataset_path)
            with open(dataset_num_examples_file_path, 'w') as file:
                file.write(str(num_examples_in_dataset))
        return num_examples_in_dataset

    def load_or_build(self):
        self.vocabs = Code2VecVocabs(self.config)
        self._load_or_create_inner_model()

    def save(self, model_save_path=None):
        if model_save_path is None:
            model_save_path = self.config.MODEL_SAVE_PATH
        model_save_dir = '/'.join(model_save_path.split('/')[:-1])
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir, exist_ok=True)
        self.vocabs.save(self.config.get_vocabularies_path_from_model_path(model_save_path))
        self._save_inner_model(model_save_path)

    def _write_code_vectors(self, file, code_vectors):
        for vec in code_vectors:
            file.write(' '.join(map(str, vec)) + '\n')

    def _get_attention_weight_per_context(
            self, path_source_strings: Iterable[str], path_strings: Iterable[str], path_target_strings: Iterable[str],
            attention_weights: Iterable[float]) -> Dict[Tuple[str, str, str], float]:
        attention_weights = np.squeeze(attention_weights, axis=-1)  # (max_contexts, )
        attention_per_context: Dict[Tuple[str, str, str], float] = {}
        # shape of path_source_strings, path_strings, path_target_strings, attention_weights is (max_contexts, )

        # iterate over contexts
        for path_source, path, path_target, weight in \
                zip(path_source_strings, path_strings, path_target_strings, attention_weights):
            string_context_triplet = (common.binary_to_string(path_source),
                                      common.binary_to_string(path),
                                      common.binary_to_string(path_target))
            attention_per_context[string_context_triplet] = weight
        return attention_per_context

    def close_session(self):
        # can be overridden by the implementation model class.
        # default implementation just does nothing.
        pass

    @abc.abstractmethod
    def train(self):
        ...

    @abc.abstractmethod
    def evaluate(self) -> Optional[ModelEvaluationResults]:
        ...

    @abc.abstractmethod
    def predict(self, predict_data_lines: Iterable[str]) -> List[ModelPredictionResults]:
        ...

    @abc.abstractmethod
    def _save_inner_model(self, path):
        ...

    def _load_or_create_inner_model(self):
        if self.config.is_loading:
            self._load_inner_model()
        else:
            self._create_inner_model()

    @abc.abstractmethod
    def _load_inner_model(self):
        ...

    def _create_inner_model(self):
        # can be overridden by the implementation model class.
        # default implementation just does nothing.
        pass

    def _initialize(self):
        # can be overridden by the implementation model class.
        # default implementation just does nothing.
        pass

    @abc.abstractmethod
    def _get_vocab_embedding_as_np_array(self, vocab_type: VocabType) -> np.ndarray:
        ...

    def save_word2vec_format(self, dest_save_path: str, vocab_type: VocabType):
        if vocab_type not in VocabType:
            raise ValueError('`vocab_type` should be `VocabType.Token`, `VocabType.Target` or `VocabType.Path`.')
        vocab_embedding_matrix = self._get_vocab_embedding_as_np_array(vocab_type)
        index_to_word = self.vocabs.get(vocab_type).index_to_word
        with open(dest_save_path, 'w') as words_file:
            common.save_word2vec_file(words_file, index_to_word, vocab_embedding_matrix)
