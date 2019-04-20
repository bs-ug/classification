import os
from glob import glob

from gensim.models.callbacks import CallbackAny2Vec
from keras.preprocessing import text
from sklearn import metrics


class Callback(CallbackAny2Vec):
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


def text_generator(path, file_extension):
    for file in glob(os.path.join(path, f"*.{file_extension}")):
        with open(file, "r", encoding="utf-8") as input_file:
            content = input_file.read()
        yield text.text_to_word_sequence(content)


def train_model(classifier, training_data, training_labels, validation_data, validation_labels):
    classifier.fit(training_data, training_labels)
    predictions = classifier.predict(validation_data)
    predictions = predictions.argmax(axis=-1)
    return classifier, metrics.accuracy_score(predictions, validation_labels)


def test_model(classifier, test_data, test_labels):
    pass
