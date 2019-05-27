from typing import Optional
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.callbacks import Callback


class ModelTrainingStatus:
    def __init__(self):
        self.nr_epochs_trained: int = 0
        self.trained_full_last_epoch: bool = False


class ModelCheckpointSaverCallback(Callback):
    def __init__(self, checkpoint_manager: tf.train.CheckpointManager,
                 training_status: ModelTrainingStatus,
                 nr_epochs_to_save: int = 1):
        self.checkpoint_manager: tf.train.CheckpointManager = checkpoint_manager
        self.training_status: ModelTrainingStatus = training_status
        self.nr_epochs_to_save: int = nr_epochs_to_save

        self.last_saved_epoch: int = 0
        super(ModelCheckpointSaverCallback, self).__init__()

    def on_train_begin(self, logs=None):
        self.last_saved_epoch = self.training_status.nr_epochs_trained

    def on_epoch_begin(self, epoch, logs=None):
        self.training_status.trained_full_last_epoch = False

    def on_epoch_end(self, epoch, logs=None):
        assert self.training_status.nr_epochs_trained == epoch
        self.training_status.nr_epochs_trained += 1
        self.training_status.trained_full_last_epoch = True
        nr_non_saved_epochs = self.training_status.nr_epochs_trained - self.last_saved_epoch
        if nr_non_saved_epochs >= self.nr_epochs_to_save:
            self.checkpoint_manager.save(checkpoint_number=self.training_status.nr_epochs_trained)
            self.last_saved_epoch = self.training_status.nr_epochs_trained
