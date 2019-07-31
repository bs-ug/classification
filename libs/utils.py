import json
import os
from glob import glob

import numpy as np
import pandas
from gensim.models.callbacks import CallbackAny2Vec
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from numpy import argmax
from sklearn import metrics, preprocessing
from sklearn.metrics import roc_auc_score

import settings


class W2VCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss() - self.loss
        self.loss += loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

    def on_epoch_begin(self, model):
        print(f"Epoch {self.epoch}.")

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


def text_generator(paths, clean=False):
    for file in glob(paths):
        with open(file, "r", encoding="utf-8") as input_file:
            try:
                content = input_file.read()
            except UnicodeDecodeError:
                continue
        yield text.text_to_word_sequence(content)


def train_model(classifier, training_data, training_labels, validation_data, validation_labels, batch_size, epochs,
                model_path, model_name, logs_path):
    # callbacks = KerasCallback()
    checkpoint = ModelCheckpoint(os.path.join(model_path, model_name), monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    if "CUDA_PATH" in os.environ.keys():
        tensorboard = TensorBoard(log_dir=logs_path, embeddings_freq=epochs, embeddings_data=validation_data)
    else:
        tensorboard = TensorBoard(log_dir=logs_path, embeddings_freq=epochs)
    classifier.fit(
        training_data, training_labels,
        batch_size=batch_size, epochs=epochs,
        validation_data=(validation_data, validation_labels),
        callbacks=[checkpoint, tensorboard])
    predictions = classifier.predict(validation_data)
    predictions = [argmax(item) for item in predictions]
    validation_labels = [argmax(item) for item in validation_labels]
    return classifier, metrics.accuracy_score(validation_labels, predictions), predictions


def test_model(model_path, test_data, test_labels, batch_size):
    classifier = load_model(model_path)
    loss, acc = classifier.evaluate(test_data, test_labels, batch_size=batch_size)
    predictions = classifier.predict(test_data, batch_size=batch_size)
    categories = np.argmax(predictions, axis=1)
    unique, counts = np.unique(categories, return_counts=True)
    return loss, acc, dict(zip(unique, counts)), categories


def get_word_embeddings_for_training(model, train_x, validation_x, padding_length):
    embeddings_index = {}
    with open(model, "r", encoding="utf-8") as file:
        for line in file:
            values = line.split()
            embeddings_index[values[0]] = np.asarray(values[-settings.EMBEDDINGS_VECTOR_LENGTH:], dtype='float32')

    token = text.Tokenizer()
    token.fit_on_texts(train_x)
    word_index = token.word_index

    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=padding_length)
    validation_seq_x = sequence.pad_sequences(token.texts_to_sequences(validation_x), maxlen=padding_length)

    embedding_matrix = np.zeros((len(word_index) + 1, settings.EMBEDDINGS_VECTOR_LENGTH))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, word_index, train_seq_x, validation_seq_x


def get_word_embeddings_for_test(test_x, padding_length):

    token = text.Tokenizer()
    token.fit_on_texts(test_x)
    test_seq_x = sequence.pad_sequences(token.texts_to_sequences(test_x), maxlen=padding_length)
    return test_seq_x


def prepare_data(labels_dict, files_path):
    labels, texts = [], []
    for filename, label in labels_dict.items():
        with open(os.path.join(files_path, f"{filename}.txt"), "r", encoding="utf-8") as file:
            content = file.read()
        labels.append(label)
        texts.append(content)
    return labels, texts


def prepare_dataset(labels_file, files_path):
    encoder = preprocessing.LabelEncoder()
    with open(labels_file, "r", encoding="utf-8") as file:
        labels = json.load(file)
    dataset = pandas.DataFrame()
    dataset["labels"], dataset["text"] = prepare_data(labels, files_path)
    return dataset["text"], encoder.fit_transform(dataset["labels"])


def prepare_training_datasets(data_dir, w2v_model, length):
    train_x, train_y = prepare_dataset(
        os.path.join(data_dir, settings.TRAINING_LABELS),
        os.path.join(data_dir, settings.TRAINING_FILES))
    validation_x, validation_y = prepare_dataset(
        os.path.join(data_dir, settings.VALIDATION_LABELS),
        os.path.join(data_dir, settings.VALIDATION_FILES))
    train_y = to_categorical(train_y)
    validation_y = to_categorical(validation_y)
    embedding_matrix, word_index, train_seq_x, validation_seq_x = get_word_embeddings_for_training(
        os.path.join(settings.MODELS_PATH, w2v_model), train_x, validation_x, length)

    return embedding_matrix, word_index, train_seq_x, train_y, validation_seq_x, validation_y


def prepare_test_dataset(data_dir, length):
    test_x, test_y = prepare_dataset(
        os.path.join(data_dir, settings.TEST_LABELS),
        os.path.join(data_dir, settings.TEST_FILES))
    test_y = to_categorical(test_y)
    test_seq_x = get_word_embeddings_for_test(test_x, length)
    return test_seq_x, test_y
