import json
import os

import numpy
import pandas as pandas
from keras.preprocessing import text, sequence
from sklearn import preprocessing

from scripts import settings


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


train_x, train_y = prepare_dataset(
    os.path.join(settings.CNN_DATA_DIR, settings.CNN_TRAINING_LABELS),
    settings.CNN_TRAINING_FILES_PATH)
validation_x, validation_y = prepare_dataset(
    os.path.join(settings.CNN_DATA_DIR, settings.CNN_VALIDATION_LABELS),
    settings.CNN_VALIDATION_FILES_PATH)
test_x, test_y = prepare_dataset(
    os.path.join(settings.CNN_DATA_DIR, settings.CNN_TEST_LABELS),
    settings.CNN_TEST_FILES_PATH)

embeddings_index = {}
with open(os.path.join(settings.CNN_MODEL_PATH, settings.CNN_MODEL_NAME), "r", encoding="utf-8") as file:
    for line in file:
        values = line.split()
        embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

# create a tokenizer
token = text.Tokenizer()
token.fit_on_texts(train_x)
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x))
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(validation_x))

# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
