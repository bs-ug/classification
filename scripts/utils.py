import os
import pickle
from glob import glob

from gensim.models.callbacks import CallbackAny2Vec
from keras.callbacks import Callback
from keras.preprocessing import text
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from numpy import argmax


class W2VCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

    def on_epoch_begin(self, model):
        print(f"{self.epoch} started.")

    def on_train_begin(self, model):
        print("Training started.")


class KerasCallback(Callback):
    def __init__(self):
        super().__init__()
        self.aucs = []
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.validation_data[0])
        self.aucs.append(roc_auc_score(self.validation_data[1], y_pred))
        return


def text_generator(path, file_extension):
    for file in glob(os.path.join(path, f"*.{file_extension}")):
        with open(file, "r", encoding="utf-8") as input_file:
            content = input_file.read()
        yield text.text_to_word_sequence(content)


def train_model(classifier, training_data, training_labels, validation_data, validation_labels):
    callbacks = KerasCallback()
    classifier.fit(
        training_data, training_labels,
        batch_size=128, epochs=1,
        validation_data=(validation_data, validation_labels),
        callbacks=[callbacks])
    predictions = classifier.predict(validation_data)
    predictions = [argmax(item) for item in predictions]
    validation_labels = [argmax(item) for item in validation_labels]
    return metrics.accuracy_score(validation_labels, predictions)


def test_model(classifier, test_data, test_labels):
    pass
