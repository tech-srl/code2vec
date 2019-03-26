from math import ceil


class Config:
    @staticmethod
    def get_default_config(args):
        config = Config()
        config.DL_FRAMEWORK = 'keras'
        config.NUM_EPOCHS = 20
        config.SAVE_EVERY_EPOCHS = 1
        config.TRAIN_BATCH_SIZE = 512
        config.TEST_BATCH_SIZE = config.TRAIN_BATCH_SIZE
        config.READING_BATCH_SIZE = 1300 * 4  # not used by the new reader
        config.NUM_BATCHING_THREADS = 2  # not used by the new reader
        config.BATCH_QUEUE_SIZE = 300000  # not used by the new reader
        config.MAX_CONTEXTS = 200
        config.MAX_TOKEN_VOCAB_SIZE = 1301136
        config.MAX_TARGET_VOCAB_SIZE = 261245
        config.MAX_PATH_VOCAB_SIZE = 911417
        config.EMBEDDINGS_SIZE = 128
        config.MAX_TO_KEEP = 10
        config.DROPOUT_KEEP_RATE = 0.75
        config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION = 10

        config.READER_NUM_PARALLEL_BATCHES = 6  # cpu cores [for tf.contrib.data.map_and_batch()]
        config.SHUFFLE_BUFFER_SIZE = 10000
        config.CSV_BUFFER_SIZE = 100 * 1024 * 1024  # 100 MB

        # Automatically filled, do not edit:
        config.TRAIN_DATA_PATH = args.data_path
        config.TEST_DATA_PATH = args.test_path
        config.MODEL_SAVE_PATH = args.save_path
        config.MODEL_LOAD_PATH = args.load_path
        config.RELEASE = args.release
        config.EXPORT_CODE_VECTORS = args.export_code_vectors
        return config

    def __init__(self):
        self.DL_FRAMEWORK: str = ''  # in {'keras', 'tensorflow'}
        self.NUM_EPOCHS: int = 0
        self.SAVE_EVERY_EPOCHS: int = 0
        self.TRAIN_BATCH_SIZE: int = 0
        self.TEST_BATCH_SIZE: int = 0
        self.READING_BATCH_SIZE: int = 0  # not used by the new reader
        self.NUM_BATCHING_THREADS: int = 0  # not used by the new reader
        self.BATCH_QUEUE_SIZE: int = 0  # not used by the new reader
        self.MAX_CONTEXTS: int = 0
        self.MAX_TOKEN_VOCAB_SIZE: int = 0
        self.MAX_TARGET_VOCAB_SIZE: int = 0
        self.MAX_PATH_VOCAB_SIZE: int = 0
        self.EMBEDDINGS_SIZE: int = 0
        self.MAX_TO_KEEP: int = 0
        self.DROPOUT_KEEP_RATE: float = 0
        self.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION: int = 0

        self.READER_NUM_PARALLEL_BATCHES: int = 0
        self.SHUFFLE_BUFFER_SIZE: int = 0
        self.CSV_BUFFER_SIZE: int = 0

        # Automatically filled by `args`.
        self.MODEL_SAVE_PATH: str = ''
        self.MODEL_LOAD_PATH: str = ''
        self.TRAIN_DATA_PATH: str = ''
        self.TEST_DATA_PATH: str = ''
        self.RELEASE: bool = False
        self.EXPORT_CODE_VECTORS: bool = False

        # Automatically filled by `ModelBase.__init__()`.
        self.NUM_TRAIN_EXAMPLES: int = 0
        self.NUM_TEST_EXAMPLES: int = 0   # TODO: fix its calculation in `ModelBase.__init__()`!

    @property
    def train_steps_per_epoch(self) -> int:
        return ceil(self.NUM_TRAIN_EXAMPLES / self.TRAIN_BATCH_SIZE)

    @property
    def test_steps_per_epoch(self) -> int:
        return ceil(self.NUM_TEST_EXAMPLES / self.TEST_BATCH_SIZE)

    def data_path(self, is_evaluating: bool = False):
        return self.TEST_DATA_PATH if is_evaluating else (self.TRAIN_DATA_PATH + '.train.c2v')

    def batch_size(self, is_evaluating: bool = False):
        return self.TEST_BATCH_SIZE if is_evaluating else self.TRAIN_BATCH_SIZE  # take min with NUM_TRAIN_EXAMPLES?
