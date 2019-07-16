import time
import datetime
import logging
from typing import Optional, Dict
from collections import defaultdict
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.callbacks import Callback

from config import Config


class ModelTrainingStatus:
    def __init__(self):
        self.nr_epochs_trained: int = 0
        self.trained_full_last_epoch: bool = False


class ModelTrainingStatusTrackerCallback(Callback):
    def __init__(self, training_status: ModelTrainingStatus):
        self.training_status: ModelTrainingStatus = training_status
        super(ModelTrainingStatusTrackerCallback, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.training_status.trained_full_last_epoch = False

    def on_epoch_end(self, epoch, logs=None):
        assert self.training_status.nr_epochs_trained == epoch
        self.training_status.nr_epochs_trained += 1
        self.training_status.trained_full_last_epoch = True


class ModelCheckpointSaverCallback(Callback):
    """
    @model_wrapper should have a `.save()` method.
    """
    def __init__(self, model_wrapper, nr_epochs_to_save: int = 1,
                 logger: logging.Logger = None):
        self.model_wrapper = model_wrapper
        self.nr_epochs_to_save: int = nr_epochs_to_save
        self.logger = logger if logger is not None else logging.getLogger()

        self.last_saved_epoch: Optional[int] = None
        super(ModelCheckpointSaverCallback, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        if self.last_saved_epoch is None:
            self.last_saved_epoch = (epoch + 1) - 1

    def on_epoch_end(self, epoch, logs=None):
        nr_epochs_trained = epoch + 1
        nr_non_saved_epochs = nr_epochs_trained - self.last_saved_epoch
        if nr_non_saved_epochs >= self.nr_epochs_to_save:
            self.logger.info('Saving model after {} epochs.'.format(nr_epochs_trained))
            self.model_wrapper.save()
            self.logger.info('Done saving model.')
            self.last_saved_epoch = nr_epochs_trained


class MultiBatchCallback(Callback):
    def __init__(self, multi_batch_size: int, average_logs: bool = False):
        self.multi_batch_size = multi_batch_size
        self.average_logs = average_logs
        self._multi_batch_start_time: int = 0
        self._multi_batch_logs_sum: Dict[str, float] = defaultdict(float)
        super(MultiBatchCallback, self).__init__()

    def on_batch_begin(self, batch, logs=None):
        if self.multi_batch_size == 1 or (batch + 1) % self.multi_batch_size == 1:
            self._multi_batch_start_time = time.time()
            if self.average_logs:
                self._multi_batch_logs_sum = defaultdict(float)

    def on_batch_end(self, batch, logs=None):
        if self.average_logs:
            assert isinstance(logs, dict)
            for log_key, log_value in logs.items():
                self._multi_batch_logs_sum[log_key] += log_value
        if self.multi_batch_size == 1 or (batch + 1) % self.multi_batch_size == 0:
            multi_batch_elapsed = time.time() - self._multi_batch_start_time
            if self.average_logs:
                multi_batch_logs = {log_key: log_value / self.multi_batch_size
                                    for log_key, log_value in self._multi_batch_logs_sum.items()}
            else:
                multi_batch_logs = logs
            self.on_multi_batch_end(batch, multi_batch_logs, multi_batch_elapsed)

    def on_multi_batch_end(self, batch, logs, multi_batch_elapsed):
        pass


class ModelTrainingProgressLoggerCallback(MultiBatchCallback):
    def __init__(self, config: Config, training_status: ModelTrainingStatus):
        self.config = config
        self.training_status = training_status
        self.avg_throughput: Optional[float] = None
        super(ModelTrainingProgressLoggerCallback, self).__init__(
            self.config.NUM_BATCHES_TO_LOG_PROGRESS, average_logs=True)

    def on_train_begin(self, logs=None):
        self.config.log('Starting training...')

    def on_epoch_end(self, epoch, logs=None):
        self.config.log('Completed epoch #{}: {}'.format(epoch + 1, logs))

    def on_multi_batch_end(self, batch, logs, multi_batch_elapsed):
        nr_samples_in_multi_batch = self.config.TRAIN_BATCH_SIZE * \
                                    self.config.NUM_BATCHES_TO_LOG_PROGRESS
        throughput = nr_samples_in_multi_batch / multi_batch_elapsed
        if self.avg_throughput is None:
            self.avg_throughput = throughput
        else:
            self.avg_throughput = 0.5 * throughput + 0.5 * self.avg_throughput
        remained_batches = self.config.train_steps_per_epoch - (batch + 1)
        remained_samples = remained_batches * self.config.TRAIN_BATCH_SIZE
        remained_time_sec = remained_samples / self.avg_throughput

        self.config.log(
            'Train: during epoch #{epoch} batch {batch}/{tot_batches} ({batch_precision}%) -- '
            'throughput (#samples/sec): {throughput} -- epoch ETA: {epoch_ETA} -- loss: {loss:.4f}'.format(
                epoch=self.training_status.nr_epochs_trained + 1,
                batch=batch + 1,
                batch_precision=int(((batch + 1) / self.config.train_steps_per_epoch) * 100),
                tot_batches=self.config.train_steps_per_epoch,
                throughput=int(throughput),
                epoch_ETA=str(datetime.timedelta(seconds=int(remained_time_sec))),
                loss=logs['loss']))
