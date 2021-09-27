from vocabularies import VocabType
from config import Config
from interactive_predict import InteractivePredictor
from model_base import Code2VecModelBase
from pathlib import Path


def load_model_dynamically(config: Config) -> Code2VecModelBase:
    assert config.DL_FRAMEWORK in {'tensorflow', 'keras'}
    if config.DL_FRAMEWORK == 'tensorflow':
        from tensorflow_model import Code2VecModel
    elif config.DL_FRAMEWORK == 'keras':
        from keras_model import Code2VecModel
    return Code2VecModel(config)


if __name__ == '__main__':
    config = Config(set_defaults=True, load_from_args=True, verify=True)

    model = load_model_dynamically(config)
    config.log('Done creating code2vec model')

    predictor = InteractivePredictor(config, model)
    folder = Path('dataset')
    predictor.predict(folder)
    model.close_session()
