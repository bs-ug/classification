import hashlib
import json
import os
from glob import glob
from random import choice

from gensim.models import Word2Vec

import settings
from .networks import simple, cnn, rnn
from .utils import text_generator, W2VCallback, train_model, prepare_training_datasets, get_log_dir


def hash_hex(s):
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


def prepare_cnn_topics():
    urls = {}
    with open(os.path.join(settings.CNN_DATA_DIR, settings.CNN_SOURCE_URLS_FILE), "r") as file:
        for line in file:
            urls[hash_hex(line.strip('\n'))] = line.strip('\n')
    filtered_urls = []
    for name, url in urls.items():
        for key, value in settings.CNN_TOPICS.items():
            stop = False
            for item in sorted(value, reverse=True):
                if item in url.lower() and (name, key) not in filtered_urls:
                    filtered_urls.append((name, key))
    train_files = {}
    validation_files = {}
    test_files = {}
    global_counter = {key: {"test": 0, "train": 0, "validation": 0} for key in settings.CNN_TOPICS.keys()}
    for path in ["train", "validation", "test"]:
        try:
            os.mkdir(os.path.join(settings.CNN_DATA_DIR, path))
        except FileExistsError:
            files = glob(os.path.join(settings.CNN_DATA_DIR, path, "*.txt"))
            for file in files:
                os.remove(file)
    while filtered_urls:
        item, value = choice(filtered_urls)
        filtered_urls.pop(filtered_urls.index((item, value)))
        try:
            with open(os.path.join(settings.CNN_SOURCE_FILES, f"{item}.story"), "r", encoding="utf-8") as file:
                news = file.read()
            news = news.split()
            news = news[:news.index("@highlight")]
            words = len(news)
        except FileNotFoundError:
            continue
        if words >= settings.CNN_MIN_ARTICLE_LENGTH:
            if global_counter[value]["train"] < settings.CNN_TRAIN_QUANTITY:
                train_files[item] = value
                with open(os.path.join(settings.CNN_DATA_DIR, settings.TRAINING_FILES, f"{item}.txt"), "w",
                          encoding="UTF-8") as file:
                    file.write(" ".join(news))
                    global_counter[value]["train"] += 1
            elif global_counter[value]["validation"] < settings.CNN_VALIDATION_QUANTITY:
                validation_files[item] = value
                with open(os.path.join(settings.CNN_DATA_DIR, settings.VALIDATION_FILES, f"{item}.txt"), "w",
                          encoding="UTF-8") as file:
                    file.write(" ".join(news))
                    global_counter[value]["validation"] += 1
            elif global_counter[value]["test"] < settings.CNN_TEST_QUANTITY:
                test_files[item] = value
                with open(os.path.join(settings.CNN_DATA_DIR, settings.TEST_FILES, f"{item}.txt"), "w",
                          encoding="UTF-8") as file:
                    file.write(" ".join(news))
                    global_counter[value]["test"] += 1
    for name, data in zip(["train", "validation", "test"], [train_files, validation_files, test_files]):
        with open(os.path.join(settings.CNN_DATA_DIR, f"{name}.json"), "w") as output_file:
            json.dump(data, output_file)
        print(f"{len(data)} {name} labels saved")


def train_cnn_w2v(filename):
    iterable = text_generator(os.path.join(settings.CNN_DATA_DIR, '*', "*.txt"))
    model = Word2Vec(
        [item for item in iterable],
        size=settings.EMBEDDINGS_VECTOR_LENGTH,
        window=5, min_count=1, workers=4,
        compute_loss=True,
        callbacks=[W2VCallback()]
    )
    model.wv.save_word2vec_format(os.path.join(settings.MODELS_PATH, filename), binary=False)


def train_cnn_simple(model_name, w2v_model, batch_size=128, epochs=100, length=400):
    embedding_matrix, word_index, train_seq_x, train_y, validation_seq_x, validation_y = prepare_training_datasets(
        settings.CNN_DATA_DIR, w2v_model, length
    )
    classifier = simple(word_index, embedding_matrix, len(settings.CNN_TOPICS), settings.PADDING_LENGTH)
    print(classifier.summary())
    log_dir = get_log_dir(model_name, batch_size, epochs)
    os.makedirs(log_dir, exist_ok=True)
    score = train_model(
        classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=batch_size,
        epochs=epochs, model_path=settings.MODELS_PATH, model_name=model_name, logs_path=log_dir
    )
    return score


def train_cnn_cnn(model_name, w2v_model, batch_size=128, epochs=100, length=400):
    embedding_matrix, word_index, train_seq_x, train_y, validation_seq_x, validation_y = prepare_training_datasets(
        settings.CNN_DATA_DIR, w2v_model, length
    )
    classifier = cnn(word_index, embedding_matrix, len(settings.CNN_TOPICS), length)
    print(classifier.summary())
    log_dir = get_log_dir(model_name, batch_size, epochs)
    os.makedirs(log_dir, exist_ok=True)
    score = train_model(
        classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=batch_size,
        epochs=epochs, model_path=settings.MODELS_PATH, model_name=model_name, logs_path=log_dir
    )
    return score


def train_cnn_rnn(model_name, w2v_model, batch_size=128, epochs=100, length=400):
    embedding_matrix, word_index, train_seq_x, train_y, validation_seq_x, validation_y = prepare_training_datasets(
        settings.CNN_DATA_DIR, w2v_model, length
    )
    classifier = rnn(word_index, embedding_matrix, len(settings.CNN_TOPICS), settings.PADDING_LENGTH)
    print(classifier.summary())
    log_dir = get_log_dir(model_name, batch_size, epochs)
    os.makedirs(log_dir, exist_ok=True)
    score = train_model(
        classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=batch_size,
        epochs=epochs, model_path=settings.MODELS_PATH, model_name=model_name, logs_path=log_dir
    )
    return score
