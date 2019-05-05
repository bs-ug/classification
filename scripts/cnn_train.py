import os
from statistics import median, mean

from keras.utils import to_categorical

from scripts import settings
from scripts.networks import simple, cnn, rnn
from scripts.utils import get_word_embeddings, train_model, prepare_dataset

train_x, train_y = prepare_dataset(
    os.path.join(settings.CNN_DATA_DIR, settings.CNN_TRAINING_LABELS),
    settings.CNN_TRAINING_FILES_PATH)
validation_x, validation_y = prepare_dataset(
    os.path.join(settings.CNN_DATA_DIR, settings.CNN_VALIDATION_LABELS),
    settings.CNN_VALIDATION_FILES_PATH)
test_x, test_y = prepare_dataset(
    os.path.join(settings.CNN_DATA_DIR, settings.CNN_TEST_LABELS),
    settings.CNN_TEST_FILES_PATH)

max_length = int(max([len(item.split(' ')) for item in train_x]))
min_length = int(min([len(item.split(' ')) for item in train_x]))
median_length = int(median([len(item.split(' ')) for item in train_x]))
mean_length = int(mean([len(item.split(' ')) for item in train_x]))

print(f"max_length: {max_length}, min_length: {min_length}, median_length: {median_length}, mean_length: {mean_length}")

train_y = to_categorical(train_y)
validation_y = to_categorical(validation_y)
test_y = to_categorical(test_y)

median_article_length = int(median([len(item.split(' ')) for item in train_x]))

embedding_matrix, word_index, train_seq_x, validation_seq_x = get_word_embeddings(
    os.path.join(settings.CNN_MODEL_PATH, "cnn_w2v_300.model"), train_x, validation_x, median_article_length)

# TODO: select network to train via script params
# classifier = cnn(word_index, embedding_matrix, len(settings.CNN_TOPICS), median_article_length)
# print(classifier.summary())
# train_model(classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=128, epochs=50,
#             model_path=settings.CNN_MODEL_PATH, model_name="cnn_model.hdf5",
#             logs_path=os.path.join(settings.CNN_DATA_DIR, "logs"))

# classifier = simple(word_index, embedding_matrix, len(settings.CNN_TOPICS), median_article_length)
# print(classifier.summary())
# train_model(classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=128, epochs=50,
#             model_path=settings.CNN_MODEL_PATH, model_name="simple_model.hdf5",
#             logs_path=os.path.join(settings.CNN_DATA_DIR, "logs"))

classifier = rnn(word_index, embedding_matrix, len(settings.CNN_TOPICS), median_article_length)
print(classifier.summary())
train_model(classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=256, epochs=50,
            model_path=settings.CNN_MODEL_PATH, model_name="rnn_model.hdf5",
            logs_path=os.path.join(settings.CNN_DATA_DIR, "logs"))
