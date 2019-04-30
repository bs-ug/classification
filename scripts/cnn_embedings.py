import os

import numpy
from keras.preprocessing import text, sequence

from scripts import settings


def get_word_embeddings(model, train_x, validation_x):
    embeddings_index = {}
    with open(model, "r", encoding="utf-8") as file:
        for line in file:
            values = line.split()
            embeddings_index[values[0]] = numpy.asarray(values[-settings.EMBEDDINGS_VECTOR_LENGTH:], dtype='float32')

    token = text.Tokenizer()
    token.fit_on_texts(train_x)
    word_index = token.word_index

    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x))
    validation_seq_x = sequence.pad_sequences(token.texts_to_sequences(validation_x), maxlen=len(train_seq_x[0]))

    embedding_matrix = numpy.zeros((len(word_index) + 1, settings.EMBEDDINGS_VECTOR_LENGTH))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, word_index, train_seq_x, validation_seq_x
