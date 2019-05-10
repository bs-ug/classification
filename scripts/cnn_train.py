import os
from datetime import datetime
from statistics import median, mean

from keras.utils import to_categorical

from scripts import settings
from scripts.networks import simple
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

embedding_model = "cnn_w2v_300.model"
embedding_matrix, word_index, train_seq_x, validation_seq_x = get_word_embeddings(
    os.path.join(settings.CNN_MODEL_PATH, embedding_model), train_x, validation_x, median_article_length)

# TODO: select network to train via script params
# classifier = cnn(word_index, embedding_matrix, len(settings.CNN_TOPICS), median_article_length)
# print(classifier.summary())
# model_name="rnn.hdf5"
# batch_size=128
# epochs=50
# log_dir = os.path.join(settings.DATA_DIR, "logs",
#                        f"cnn-{model_name.split('.')[0]}-{embedding_model.split('.')[0]}-{batch_size}-{epochs}"
#                        f"-{datetime.now().strftime('%Y%m%d%TH%M')}")
# os.makedirs(log_dir, exist_ok=True)
# train_model(classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=batch_size, epochs=epochs,
#             model_path=settings.CNN_MODEL_PATH, model_name=model_name, logs_path=log_dir)

classifier = simple(word_index, embedding_matrix, len(settings.CNN_TOPICS), median_article_length)
print(classifier.summary())
model_name = "simple.hdf5"
batch_size = 128
epochs = 50
log_dir = os.path.join(settings.DATA_DIR, "logs",
                       f"cnn-{model_name.split('.')[0]}-{embedding_model.split('.')[0]}-{batch_size}-{epochs}"
                       f"-{datetime.now().strftime('%m%dT%H%M')}")
os.makedirs(log_dir, exist_ok=True)
train_model(classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=batch_size, epochs=epochs,
            model_path=settings.CNN_MODEL_PATH, model_name=model_name, logs_path=log_dir)

# classifier = rnn(word_index, embedding_matrix, len(settings.CNN_TOPICS), median_article_length)
# print(classifier.summary())
# model_name="rnn.hdf5"
# batch_size=128
# epochs=50
# log_dir = os.path.join(settings.DATA_DIR, "logs",
#                        f"cnn-{model_name.split('.')[0]}-{embedding_model.split('.')[0]}-{batch_size}-{epochs}"
#                        f"-{datetime.now().strftime('%Y%m%d%TH%M')}")
# os.makedirs(log_dir, exist_ok=True)
# train_model(classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=batch_size, epochs=epochs,
#             model_path=settings.CNN_MODEL_PATH, model_name=model_name, logs_path=log_dir)
