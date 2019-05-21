import os
from datetime import datetime
from statistics import median, mean

from keras.utils import to_categorical

from scripts import settings
from scripts.networks import simple, cnn, rnn
from scripts.utils import get_word_embeddings, train_model, prepare_dataset, test_model

train_x, train_y = prepare_dataset(
    os.path.join(settings.BBC_DATA_DIR, settings.TRAINING_LABELS),
    os.path.join(settings.BBC_DATA_DIR, settings.TRAINING_FILES))
validation_x, validation_y = prepare_dataset(
    os.path.join(settings.BBC_DATA_DIR, settings.VALIDATION_LABELS),
    os.path.join(settings.BBC_DATA_DIR, settings.VALIDATION_FILES))
test_x, test_y = prepare_dataset(
    os.path.join(settings.BBC_DATA_DIR, settings.TEST_LABELS),
    os.path.join(settings.BBC_DATA_DIR, settings.TEST_FILES))

max_length = int(max([len(item.split(' ')) for item in train_x]))
min_length = int(min([len(item.split(' ')) for item in train_x]))
median_length = int(median([len(item.split(' ')) for item in train_x]))
mean_length = int(mean([len(item.split(' ')) for item in train_x]))

print(f"max_length: {max_length}, min_length: {min_length}, median_length: {median_length}, mean_length: {mean_length}")

train_y = to_categorical(train_y)
validation_y = to_categorical(validation_y)
test_y = to_categorical(test_y)

median_article_length = int(median([len(item.split(' ')) for item in train_x]))

embedding_matrix, word_index, train_seq_x, validation_seq_x, test_seq_x = get_word_embeddings(
    os.path.join(settings.MODELS_PATH, "bbc_100.w2v"), train_x, validation_x, test_x, settings.PADDING_LENGTH)

# TODO: select network to train via script params
# classifier = cnn(word_index, embedding_matrix, len(settings.BBC_TOPICS), settings.PADDING_LENGTH)
# print(classifier.summary())
# model_name = "bbc_cnn.h5"
# batch_size = 128
# epochs = 120
# log_dir = os.path.join(settings.DATA_DIR, "logs",
#                        f"{model_name.split('.')[0]}-{settings.BBC}-{settings.EMBEDDINGS_VECTOR_LENGTH}-{batch_size}-{epochs}"
#                        f"-{datetime.now().strftime('%m%dT%H%M')}")
# os.makedirs(log_dir, exist_ok=True)
# train_model(classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=batch_size, epochs=epochs,
#             model_path=settings.MODELS_PATH, model_name=model_name, logs_path=log_dir)

classifier = simple(word_index, embedding_matrix, len(settings.BBC_TOPICS), settings.PADDING_LENGTH)
print(classifier.summary())
model_name = "bbc_simple_400.h5"
batch_size = 128
epochs = 20
log_dir = os.path.join(settings.DATA_DIR, "logs",
                       f"{model_name.split('.')[0]}-{settings.BBC}-{settings.EMBEDDINGS_VECTOR_LENGTH}-{batch_size}-{epochs}"
                       f"-{datetime.now().strftime('%m%dT%H%M')}")
os.makedirs(log_dir, exist_ok=True)
score = train_model(
    classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=batch_size, epochs=epochs,
    model_path=settings.MODELS_PATH, model_name=model_name, logs_path=log_dir)

# classifier = rnn(word_index, embedding_matrix, len(settings.BBC_TOPICS), settings.PADDING_LENGTH)
# print(classifier.summary())
# model_name = "bbc_rnn.h5"
# batch_size = 128
# epochs = 120
# log_dir = os.path.join(settings.DATA_DIR, "logs",
#                        f"{model_name.split('.')[0]}-{settings.BBC}-{settings.EMBEDDINGS_VECTOR_LENGTH}-{batch_size}-{epochs}"
#                        f"-{datetime.now().strftime('%m%dT%H%M')}")
# os.makedirs(log_dir, exist_ok=True)
# train_model(classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=batch_size, epochs=epochs,
#             model_path=settings.MODELS_PATH, model_name=model_name, logs_path=log_dir)

loss, acc, predictions = test_model(os.path.join(settings.MODELS_PATH, model_name), test_seq_x, test_y, batch_size)
print(f"score: {score}\ntest loss: {loss}\ntest accuracy: {acc}\npredictions: {predictions}")
