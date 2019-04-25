import os

from keras.utils import to_categorical

from scripts import settings
from scripts.cnn_datasets import prepare_dataset
from scripts.cnn_embedings import get_word_embeddings
from scripts.networks import cnn, simple
from scripts.utils import train_model

train_x, train_y = prepare_dataset(
    os.path.join(settings.CNN_DATA_DIR, settings.CNN_TRAINING_LABELS),
    settings.CNN_TRAINING_FILES_PATH)
validation_x, validation_y = prepare_dataset(
    os.path.join(settings.CNN_DATA_DIR, settings.CNN_VALIDATION_LABELS),
    settings.CNN_VALIDATION_FILES_PATH)
test_x, test_y = prepare_dataset(
    os.path.join(settings.CNN_DATA_DIR, settings.CNN_TEST_LABELS),
    settings.CNN_TEST_FILES_PATH)
train_y = to_categorical(train_y)
validation_y = to_categorical(validation_y)
test_y = to_categorical(test_y)

embedding_matrix, word_index, train_seq_x, validation_seq_x = get_word_embeddings(
    os.path.join(settings.CNN_MODEL_PATH, settings.CNN_MODEL_NAME), train_x, validation_x)

# TODO: select network to train via script params
classifier = cnn(word_index, embedding_matrix)
print(classifier.summary())
train_model(classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=512, epochs=1,
            model_name="cnn_model.hdf5")

classifier = simple(word_index, embedding_matrix)
print(classifier.summary())
train_model(classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=512, epochs=10,
            model_name="simple_model.hdf5")

