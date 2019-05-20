from keras import layers, models, optimizers

from scripts import settings


def cnn(word_index, embedding_matrix, number_of_classes, vector_length):
    input_layer = layers.Input((vector_length,))
    embedding_layer = layers.Embedding(
        len(word_index) + 1,
        settings.EMBEDDINGS_VECTOR_LENGTH,
        weights=[embedding_matrix],
        trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.1)(embedding_layer)
    convolution_layer = layers.Convolution1D(200, 3, activation="relu")(embedding_layer)
    pooling_layer = layers.GlobalMaxPooling1D()(convolution_layer)
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.1)(output_layer1)
    output_layer2 = layers.Dense(number_of_classes, activation="sigmoid")(output_layer1)
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def rnn(word_index, embedding_matrix, number_of_classes, vector_length):
    input_layer = layers.Input((vector_length,))
    embedding_layer = layers.Embedding(
        len(word_index) + 1,
        settings.EMBEDDINGS_VECTOR_LENGTH,
        weights=[embedding_matrix],
        trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.1)(embedding_layer)
    lstm_layer = layers.LSTM(100)(embedding_layer)
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.1)(output_layer1)
    output_layer2 = layers.Dense(number_of_classes, activation="softmax")(output_layer1)
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def simple(word_index, embedding_matrix, number_of_classes, vector_length):
    model = models.Sequential()
    model.add(layers.Embedding(
        len(word_index) + 1,
        settings.EMBEDDINGS_VECTOR_LENGTH,
        weights=[embedding_matrix],
        input_length=vector_length,
        trainable=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(number_of_classes, activation="softmax"))
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model
