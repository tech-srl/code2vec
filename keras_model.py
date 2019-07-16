import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Concatenate, Dropout, TimeDistributed, Dense
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import sparse_top_k_categorical_accuracy

from path_context_reader import PathContextReader, ModelInputTensorsFormer, ReaderInputTensors, EstimatorAction
import os
import numpy as np
from functools import partial
from typing import List, Optional, Iterable, Union, Callable, Dict
from collections import namedtuple
import time
import datetime
from vocabularies import VocabType
from keras_attention_layer import AttentionLayer
from keras_topk_word_predictions_layer import TopKWordPredictionsLayer
from keras_words_subtoken_metrics import WordsSubtokenPrecisionMetric, WordsSubtokenRecallMetric, WordsSubtokenF1Metric
from config import Config
from common import common
from model_base import Code2VecModelBase, ModelEvaluationResults, ModelPredictionResults
from keras_checkpoint_saver_callback import ModelTrainingStatus, ModelTrainingStatusTrackerCallback,\
    ModelCheckpointSaverCallback, MultiBatchCallback, ModelTrainingProgressLoggerCallback


class Code2VecModel(Code2VecModelBase):
    def __init__(self, config: Config):
        self.keras_train_model: Optional[keras.Model] = None
        self.keras_eval_model: Optional[keras.Model] = None
        self.keras_model_predict_function: Optional[K.GraphExecutionFunction] = None
        self.training_status: ModelTrainingStatus = ModelTrainingStatus()
        self._checkpoint: Optional[tf.train.Checkpoint] = None
        self._checkpoint_manager: Optional[tf.train.CheckpointManager] = None
        super(Code2VecModel, self).__init__(config)

    def _create_keras_model(self):
        # Each input sample consists of a bag of x`MAX_CONTEXTS` tuples (source_terminal, path, target_terminal).
        # The valid mask indicates for each context whether it actually exists or it is just a padding.
        path_source_token_input = Input((self.config.MAX_CONTEXTS,), dtype=tf.int32)
        path_input = Input((self.config.MAX_CONTEXTS,), dtype=tf.int32)
        path_target_token_input = Input((self.config.MAX_CONTEXTS,), dtype=tf.int32)
        context_valid_mask = Input((self.config.MAX_CONTEXTS,))

        # Input paths are indexes, we embed these here.
        paths_embedded = Embedding(
            self.vocabs.path_vocab.size, self.config.PATH_EMBEDDINGS_SIZE, name='path_embedding')(path_input)

        # Input terminals are indexes, we embed these here.
        token_embedding_shared_layer = Embedding(
            self.vocabs.token_vocab.size, self.config.TOKEN_EMBEDDINGS_SIZE, name='token_embedding')
        path_source_token_embedded = token_embedding_shared_layer(path_source_token_input)
        path_target_token_embedded = token_embedding_shared_layer(path_target_token_input)

        # `Context` is a concatenation of the 2 terminals & path embedding.
        # Each context is a vector of size 3 * EMBEDDINGS_SIZE.
        context_embedded = Concatenate()([path_source_token_embedded, paths_embedded, path_target_token_embedded])
        context_embedded = Dropout(1 - self.config.DROPOUT_KEEP_RATE)(context_embedded)

        # Lets get dense: Apply a dense layer for each context vector (using same weights for all of the context).
        context_after_dense = TimeDistributed(
            Dense(self.config.CODE_VECTOR_SIZE, use_bias=False, activation='tanh'))(context_embedded)

        # The final code vectors are received by applying attention to the "densed" context vectors.
        code_vectors, attention_weights = AttentionLayer(name='attention')(
            [context_after_dense, context_valid_mask])

        # "Decode": Now we use another dense layer to get the target word embedding from each code vector.
        target_index = Dense(
            self.vocabs.target_vocab.size, use_bias=False, activation='softmax', name='target_index')(code_vectors)

        # Wrap the layers into a Keras model, using our subtoken-metrics and the CE loss.
        inputs = [path_source_token_input, path_input, path_target_token_input, context_valid_mask]
        self.keras_train_model = keras.Model(inputs=inputs, outputs=target_index)

        # Actual target word predictions (as strings). Used as a second output layer.
        # Used for predict() and for the evaluation metrics calculations.
        topk_predicted_words, topk_predicted_words_scores = TopKWordPredictionsLayer(
            self.config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION,
            self.vocabs.target_vocab.get_index_to_word_lookup_table(),
            name='target_string')(target_index)

        # We use another dedicated Keras model for evaluation.
        # The evaluation model outputs the `topk_predicted_words` as a 2nd output.
        # The separation between train and eval models is for efficiency.
        self.keras_eval_model = keras.Model(
            inputs=inputs, outputs=[target_index, topk_predicted_words], name="code2vec-keras-model")

        # We use another dedicated Keras function to produce predictions.
        # It have additional outputs than the original model.
        # It is based on the trained layers of the original model and uses their weights.
        predict_outputs = tuple(KerasPredictionModelOutput(
            target_index=target_index, code_vectors=code_vectors, attention_weights=attention_weights,
            topk_predicted_words=topk_predicted_words, topk_predicted_words_scores=topk_predicted_words_scores))
        self.keras_model_predict_function = K.function(inputs=inputs, outputs=predict_outputs)

    def _create_metrics_for_keras_eval_model(self) -> Dict[str, List[Union[Callable, keras.metrics.Metric]]]:
        top_k_acc_metrics = []
        for k in range(1, self.config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION + 1):
            top_k_acc_metric = partial(
                sparse_top_k_categorical_accuracy, k=k)
            top_k_acc_metric.__name__ = 'top{k}_acc'.format(k=k)
            top_k_acc_metrics.append(top_k_acc_metric)
        predicted_words_filters = [
            lambda word_strings: tf.not_equal(word_strings, self.vocabs.target_vocab.special_words.OOV),
            lambda word_strings: tf.strings.regex_full_match(word_strings, r'^[a-zA-Z\|]+$')
        ]
        words_subtokens_metrics = [
            WordsSubtokenPrecisionMetric(predicted_words_filters=predicted_words_filters, name='subtoken_precision'),
            WordsSubtokenRecallMetric(predicted_words_filters=predicted_words_filters, name='subtoken_recall'),
            WordsSubtokenF1Metric(predicted_words_filters=predicted_words_filters, name='subtoken_f1')
        ]
        return {'target_index': top_k_acc_metrics, 'target_string': words_subtokens_metrics}

    @classmethod
    def _create_optimizer(cls):
        return tf.optimizers.Adam()

    def _compile_keras_model(self, optimizer=None):
        if optimizer is None:
            optimizer = self.keras_train_model.optimizer
            if optimizer is None:
                optimizer = self._create_optimizer()

        def zero_loss(true_word, topk_predictions):
            return tf.constant(0.0, shape=(), dtype=tf.float32)

        self.keras_train_model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer)

        self.keras_eval_model.compile(
            loss={'target_index': 'sparse_categorical_crossentropy', 'target_string': zero_loss},
            optimizer=optimizer,
            metrics=self._create_metrics_for_keras_eval_model())

    def _create_data_reader(self, estimator_action: EstimatorAction, repeat_endlessly: bool = False):
        return PathContextReader(
            vocabs=self.vocabs,
            config=self.config,
            model_input_tensors_former=_KerasModelInputTensorsFormer(estimator_action=estimator_action),
            estimator_action=estimator_action,
            repeat_endlessly=repeat_endlessly)

    def _create_train_callbacks(self) -> List[Callback]:
        # TODO: do we want to use early stopping? if so, use the right chechpoint manager and set the correct
        #       `monitor` quantity (example: monitor='val_acc', mode='max')

        keras_callbacks = [
            ModelTrainingStatusTrackerCallback(self.training_status),
            ModelTrainingProgressLoggerCallback(self.config, self.training_status),
        ]
        if self.config.is_saving:
            keras_callbacks.append(ModelCheckpointSaverCallback(
                self, self.config.SAVE_EVERY_EPOCHS, self.logger))
        if self.config.is_testing:
            keras_callbacks.append(ModelEvaluationCallback(self))
        if self.config.USE_TENSORBOARD:
            log_dir = "logs/scalars/train_" + common.now_str()
            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=log_dir,
                update_freq=self.config.NUM_BATCHES_TO_LOG_PROGRESS * self.config.TRAIN_BATCH_SIZE)
            keras_callbacks.append(tensorboard_callback)
        return keras_callbacks

    def train(self):
        # initialize the input pipeline reader
        train_data_input_reader = self._create_data_reader(estimator_action=EstimatorAction.Train)

        training_history = self.keras_train_model.fit(
            train_data_input_reader.get_dataset(),
            steps_per_epoch=self.config.train_steps_per_epoch,
            epochs=self.config.NUM_TRAIN_EPOCHS,
            initial_epoch=self.training_status.nr_epochs_trained,
            verbose=self.config.VERBOSE_MODE,
            callbacks=self._create_train_callbacks())

        self.log(training_history)

    def evaluate(self) -> Optional[ModelEvaluationResults]:
        val_data_input_reader = self._create_data_reader(estimator_action=EstimatorAction.Evaluate)
        eval_res = self.keras_eval_model.evaluate(
            val_data_input_reader.get_dataset(),
            steps=self.config.test_steps,
            verbose=self.config.VERBOSE_MODE)
        k = self.config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION
        return ModelEvaluationResults(
            topk_acc=eval_res[3:k+3],
            subtoken_precision=eval_res[k+3],
            subtoken_recall=eval_res[k+4],
            subtoken_f1=eval_res[k+5],
            loss=eval_res[1]
        )

    def predict(self, predict_data_rows: Iterable[str]) -> List[ModelPredictionResults]:
        predict_input_reader = self._create_data_reader(estimator_action=EstimatorAction.Predict)
        input_iterator = predict_input_reader.process_and_iterate_input_from_data_lines(predict_data_rows)
        all_model_prediction_results = []
        for input_row in input_iterator:
            # perform the actual prediction and get raw results.
            input_for_predict = input_row[0][:4]  # we want only the relevant input vectors (w.o. the targets).
            prediction_results = self.keras_model_predict_function(input_for_predict)

            # make `input_row` and `prediction_results` easy to read (by accessing named fields).
            prediction_results = KerasPredictionModelOutput(
                *common.squeeze_single_batch_dimension_for_np_arrays(prediction_results))
            input_row = _KerasModelInputTensorsFormer(
                estimator_action=EstimatorAction.Predict).from_model_input_form(input_row)
            input_row = ReaderInputTensors(*common.squeeze_single_batch_dimension_for_np_arrays(input_row))

            # calculate the attention weight for each context
            attention_per_context = self._get_attention_weight_per_context(
                path_source_strings=input_row.path_source_token_strings,
                path_strings=input_row.path_strings,
                path_target_strings=input_row.path_target_token_strings,
                attention_weights=prediction_results.attention_weights
            )

            # store the calculated prediction results in the wanted format.
            model_prediction_results = ModelPredictionResults(
                original_name=common.binary_to_string(input_row.target_string.item()),
                topk_predicted_words=common.binary_to_string_list(prediction_results.topk_predicted_words),
                topk_predicted_words_scores=prediction_results.topk_predicted_words_scores,
                attention_per_context=attention_per_context,
                code_vector=prediction_results.code_vectors)
            all_model_prediction_results.append(model_prediction_results)

        return all_model_prediction_results

    def _save_inner_model(self, path):
        if self.config.RELEASE:
            self.keras_train_model.save_weights(self.config.get_model_weights_path(path))
        else:
            self._get_checkpoint_manager().save(checkpoint_number=self.training_status.nr_epochs_trained)

    def _create_inner_model(self):
        self._create_keras_model()
        self._compile_keras_model()
        self.keras_train_model.summary(print_fn=self.log)

    def _load_inner_model(self):
        self._create_keras_model()
        self._compile_keras_model()

        # when loading the model for further training, we must use the full saved model file (not just weights).
        # we load the entire model if we must to or if there is no model weights file to load.
        must_use_entire_model = self.config.is_training
        entire_model_exists = os.path.exists(self.config.entire_model_load_path)
        model_weights_exist = os.path.exists(self.config.model_weights_load_path)
        use_full_model = must_use_entire_model or not model_weights_exist

        if must_use_entire_model and not entire_model_exists:
            raise ValueError(
                "There is no model at path `{model_file_path}`. When loading the model for further training, "
                "we must use an entire saved model file (not just weights).".format(
                    model_file_path=self.config.entire_model_load_path))
        if not entire_model_exists and not model_weights_exist:
            raise ValueError(
                "There is no entire model to load at path `{entire_model_path}`, "
                "and there is no model weights file to load at path `{model_weights_path}`.".format(
                    entire_model_path=self.config.entire_model_load_path,
                    model_weights_path=self.config.model_weights_load_path))

        if use_full_model:
            self.log('Loading entire model from path `{}`.'.format(self.config.entire_model_load_path))
            latest_checkpoint = tf.train.latest_checkpoint(self.config.entire_model_load_path)
            if latest_checkpoint is None:
                raise ValueError("Failed to load model: Model latest checkpoint is not found.")
            self.log('Loading latest checkpoint `{}`.'.format(latest_checkpoint))
            status = self._get_checkpoint().restore(latest_checkpoint)
            status.initialize_or_restore()
            # FIXME: are we sure we have to re-compile here? I turned it off to save the optimizer state
            # self._compile_keras_model()  # We have to re-compile because we also recovered the `tf.train.AdamOptimizer`.
            self.training_status.nr_epochs_trained = int(latest_checkpoint.split('-')[-1])
        else:
            # load the "released" model (only the weights).
            self.log('Loading model weights from path `{}`.'.format(self.config.model_weights_load_path))
            self.keras_train_model.load_weights(self.config.model_weights_load_path)

        self.keras_train_model.summary(print_fn=self.log)

    def _get_checkpoint(self):
        assert self.keras_train_model is not None and self.keras_train_model.optimizer is not None
        if self._checkpoint is None:
            # TODO: we would like to save (& restore) the `nr_epochs_trained`.
            self._checkpoint = tf.train.Checkpoint(
                # nr_epochs_trained=tf.Variable(self.training_status.nr_epochs_trained, name='nr_epochs_trained'),
                optimizer=self.keras_train_model.optimizer, model=self.keras_train_model)
        return self._checkpoint

    def _get_checkpoint_manager(self):
        if self._checkpoint_manager is None:
            self._checkpoint_manager = tf.train.CheckpointManager(
                self._get_checkpoint(), self.config.entire_model_save_path,
                max_to_keep=self.config.MAX_TO_KEEP)
        return self._checkpoint_manager

    def _get_vocab_embedding_as_np_array(self, vocab_type: VocabType) -> np.ndarray:
        assert vocab_type in VocabType

        vocab_type_to_embedding_layer_mapping = {
            VocabType.Target: 'target_index',
            VocabType.Token: 'token_embedding',
            VocabType.Path: 'path_embedding'
        }
        embedding_layer_name = vocab_type_to_embedding_layer_mapping[vocab_type]
        weight = np.array(self.keras_train_model.get_layer(embedding_layer_name).get_weights()[0])
        assert len(weight.shape) == 2

        # token, path have an actual `Embedding` layers, but target have just a `Dense` layer.
        # hence, transpose the weight when necessary.
        assert self.vocabs.get(vocab_type).size in weight.shape
        if self.vocabs.get(vocab_type).size != weight.shape[0]:
            weight = np.transpose(weight)

        return weight

    def _create_lookup_tables(self):
        PathContextReader.create_needed_vocabs_lookup_tables(self.vocabs)
        self.log('Lookup tables created.')

    def _initialize(self):
        self._create_lookup_tables()


class ModelEvaluationCallback(MultiBatchCallback):
    """
    This callback is passed to the `model.fit()` call.
    It is responsible to trigger model evaluation during the training.
    The reason we use a callback and not just passing validation data to `model.fit()` is because:
        (i)   the training model is different than the evaluation model for efficiency considerations;
        (ii)  we want to control the logging format;
        (iii) we want the evaluation to occur once per 1K batches (rather than only once per epoch).
    """

    def __init__(self, code2vec_model: 'Code2VecModel'):
        self.code2vec_model = code2vec_model
        self.avg_eval_duration: Optional[int] = None
        super(ModelEvaluationCallback, self).__init__(self.code2vec_model.config.NUM_TRAIN_BATCHES_TO_EVALUATE)

    def on_epoch_end(self, epoch, logs=None):
        self.perform_evaluation()

    def on_multi_batch_end(self, batch, logs, multi_batch_elapsed):
        self.perform_evaluation()

    def perform_evaluation(self):
        if self.avg_eval_duration is None:
            self.code2vec_model.log('Evaluating...')
        else:
            self.code2vec_model.log('Evaluating... (takes ~{})'.format(
                str(datetime.timedelta(seconds=int(self.avg_eval_duration)))))
        eval_start_time = time.time()
        evaluation_results = self.code2vec_model.evaluate()
        eval_duration = time.time() - eval_start_time
        if self.avg_eval_duration is None:
            self.avg_eval_duration = eval_duration
        else:
            self.avg_eval_duration = eval_duration * 0.5 + self.avg_eval_duration * 0.5
        self.code2vec_model.log('Done evaluating (took {}). Evaluation results:'.format(
            str(datetime.timedelta(seconds=int(eval_duration)))))

        self.code2vec_model.log(
            '    loss: {loss:.4f}, f1: {f1:.4f}, recall: {recall:.4f}, precision: {precision:.4f}'.format(
                loss=evaluation_results.loss, f1=evaluation_results.subtoken_f1,
                recall=evaluation_results.subtoken_recall, precision=evaluation_results.subtoken_precision))
        top_k_acc_formated = ['top{}: {:.4f}'.format(i, acc) for i, acc in enumerate(evaluation_results.topk_acc, start=1)]
        for top_k_acc_chunk in common.chunks(top_k_acc_formated, 5):
            self.code2vec_model.log('    ' + (', '.join(top_k_acc_chunk)))


class _KerasModelInputTensorsFormer(ModelInputTensorsFormer):
    """
    An instance of this class is passed to the reader in order to help the reader to construct the input
        in the form that the model expects to receive it.
    This class also enables conveniently & clearly access input parts by their field names.
        eg: 'tensors.path_indices' instead if 'tensors[1]'.
    This allows the input tensors to be passed as pure tuples along the computation graph, while the
        python functions that construct the graph can easily (and clearly) access tensors.
    """

    def __init__(self, estimator_action: EstimatorAction):
        self.estimator_action = estimator_action

    def to_model_input_form(self, input_tensors: ReaderInputTensors):
        inputs = (input_tensors.path_source_token_indices, input_tensors.path_indices,
                  input_tensors.path_target_token_indices, input_tensors.context_valid_mask)
        if self.estimator_action.is_train:
            targets = input_tensors.target_index
        else:
            targets = {'target_index': input_tensors.target_index, 'target_string': input_tensors.target_string}
        if self.estimator_action.is_predict:
            inputs += (input_tensors.path_source_token_strings, input_tensors.path_strings,
                       input_tensors.path_target_token_strings)
        return inputs, targets

    def from_model_input_form(self, input_row) -> ReaderInputTensors:
        inputs, targets = input_row
        return ReaderInputTensors(
            path_source_token_indices=inputs[0],
            path_indices=inputs[1],
            path_target_token_indices=inputs[2],
            context_valid_mask=inputs[3],
            target_index=targets if self.estimator_action.is_train else targets['target_index'],
            target_string=targets['target_string'] if not self.estimator_action.is_train else None,
            path_source_token_strings=inputs[4] if self.estimator_action.is_predict else None,
            path_strings=inputs[5] if self.estimator_action.is_predict else None,
            path_target_token_strings=inputs[6] if self.estimator_action.is_predict else None
        )


"""Used for convenient-and-clear access to raw prediction result parts (by their names)."""
KerasPredictionModelOutput = namedtuple(
    'KerasModelOutput', ['target_index', 'code_vectors', 'attention_weights',
                         'topk_predicted_words', 'topk_predicted_words_scores'])
