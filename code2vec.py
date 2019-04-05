from vocabularies import VocabType
from config import Config
from argparse import ArgumentParser
from interactive_predict import InteractivePredictor
from model_base import Code2VecModelBase
import sys


def load_model_dynamically(config: Config) -> Code2VecModelBase:
    if config.DL_FRAMEWORK == 'tensorflow':
        from tensorflow_model import Code2VecModel
    elif config.DL_FRAMEWORK == 'keras':
        from keras_model import Code2VecModel
    else:
        raise ValueError("config.DL_FRAMEWORK must be in {'tensorflow', 'keras'}.")
    return Code2VecModel(config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed dataset", required=False)
    parser.add_argument("-te", "--test", dest="test_path",
                        help="path to test file", metavar="FILE", required=False)

    is_training = '--train' in sys.argv or '-tr' in sys.argv
    parser.add_argument("-s", "--save", dest="save_path",
                        help="path to save the model file", metavar="FILE", required=False)
    parser.add_argument("-w2v", "--save_word2v", dest="save_w2v",
                        help="path to save the tokens embeddings file", metavar="FILE", required=False)
    parser.add_argument("-t2v", "--save_target2v", dest="save_t2v",
                        help="path to save the targets embeddings file", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to load the model from", metavar="FILE", required=False)
    parser.add_argument('--save_w2v', dest='save_w2v', required=False,
                        help="save word (token) vectors in word2vec format")
    parser.add_argument('--save_t2v', dest='save_t2v', required=False,
                        help="save target vectors in word2vec format")
    parser.add_argument('--export_code_vectors', action='store_true', required=False,
                        help="export code vectors for the given examples")
    parser.add_argument('--release', action='store_true',
                        help='if specified and loading a trained model, release the loaded model for a lower model '
                             'size.')
    parser.add_argument('--predict', action='store_true',
                        help='execute the interactive prediction shell')
    parser.add_argument("-fw", "--framework", dest="dl_framework", choices=['keras', 'tensorflow'],
                        default='tensorflow', help="deep learning framework to use.")
    args = parser.parse_args()

    config = Config.get_default_config(args)

    model = load_model_dynamically(config)
    print('Created model')
    if config.TRAIN_DATA_PATH_PREFIX:
        model.train()
    if args.save_w2v is not None:
        model.save_word2vec_format(args.save_w2v, VocabType.Token)
        print('Origin word vectors saved in word2vec text format in: %s' % args.save_w2v)
    if args.save_t2v is not None:
        model.save_word2vec_format(args.save_t2v, VocabType.Target)
        print('Target word vectors saved in word2vec text format in: %s' % args.save_t2v)
    if config.TEST_DATA_PATH and not args.data_path:
        eval_results = model.evaluate()
        if eval_results is not None:
            print(str(eval_results).replace('topk', 'top{}'.format(config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION)))
    if args.predict:
        predictor = InteractivePredictor(config, model)
        predictor.predict()
    model.close_session()
