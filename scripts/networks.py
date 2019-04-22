from keras import layers, models, optimizers


def cnn(word_index, embedding_matrix):
    input_layer = layers.Input((1881,))
    embedding_layer = layers.Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
    convolution_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)
    pooling_layer = layers.GlobalMaxPool1D()(convolution_layer)
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model
