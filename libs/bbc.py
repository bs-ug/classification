import json
import os
from glob import glob

from gensim.models import Word2Vec

import settings
from .networks import simple, cnn, rnn
from .utils import text_generator, W2VCallback, train_model, prepare_training_datasets, get_log_dir


def prepare_bbc_topics():
    train_files = {}
    validation_files = {}
    test_files = {}
    counter = 0
    global_counter = {key: {"test": 0, "train": 0, "validation": 0} for key in settings.BBC_TOPICS.keys()}
    for path in ["train", "validation", "test"]:
        try:
            os.mkdir(os.path.join(settings.BBC_DATA_DIR, path))
        except FileExistsError:
            files = glob(os.path.join(settings.BBC_DATA_DIR, path, "*.txt"))
            for file in files:
                os.remove(file)
    for label, path in settings.BBC_TOPICS.items():
        for file in glob(os.path.join(settings.BBC_DATA_DIR, path, "*.txt")):
            counter += 1
            filename = f"{counter:04d}"
            with open(file, "r", encoding="utf-8") as input_file:
                try:
                    input_text = input_file.read()
                except UnicodeDecodeError:
                    pass
            content = ' '.join(input_text.split())
            words = len(content.split())
            if settings.BBC_MIN_ARTICLE_LENGTH <= words:
                if global_counter[label]["train"] < settings.BBC_TRAIN_QUANTITY:
                    train_files[filename] = label
                    with open(os.path.join(settings.BBC_DATA_DIR, settings.TRAINING_FILES, f"{filename}.txt"), "w",
                              encoding="UTF-8") as file:
                        file.write(content)
                        global_counter[label]["train"] += 1
                elif global_counter[label]["validation"] < settings.BBC_VALIDATION_QUANTITY:
                    validation_files[filename] = label
                    with open(os.path.join(settings.BBC_DATA_DIR, settings.VALIDATION_FILES, f"{filename}.txt"), "w",
                              encoding="UTF-8") as file:
                        file.write(content)
                        global_counter[label]["validation"] += 1
                elif global_counter[label]["test"] < settings.BBC_TEST_QUANTITY:
                    test_files[filename] = label
                    with open(os.path.join(settings.BBC_DATA_DIR, settings.TEST_FILES, f"{filename}.txt"), "w",
                              encoding="UTF-8") as file:
                        file.write(content)
                        global_counter[label]["test"] += 1
    for name, data in zip(["train", "validation", "test"], [train_files, validation_files, test_files]):
        with open(os.path.join(settings.BBC_DATA_DIR, f"{name}.json"), "w") as output_file:
            json.dump(data, output_file)
        print(f"{len(data)} {name} labels saved")
    print(global_counter)


def train_bbc_w2v(filename):
    iterable = text_generator(os.path.join(settings.BBC_DATA_DIR, '*', "*.txt"))
    model = Word2Vec(
        [item for item in iterable],
        size=settings.EMBEDDINGS_VECTOR_LENGTH,
        window=5, min_count=1, workers=4,
        compute_loss=True,
        callbacks=[W2VCallback()]
    )
    model.wv.save_word2vec_format(os.path.join(settings.MODELS_PATH, filename), binary=False)


def train_bbc_simple(model_name, w2v_model, batch_size=128, epochs=100, length=400):
    embedding_matrix, word_index, train_seq_x, train_y, validation_seq_x, validation_y = prepare_training_datasets(
        settings.BBC_DATA_DIR, w2v_model, length
    )
    classifier = simple(word_index, embedding_matrix, len(settings.BBC_TOPICS), settings.PADDING_LENGTH)
    print(classifier.summary())
    log_dir = get_log_dir(model_name, batch_size, epochs)
    os.makedirs(log_dir, exist_ok=True)
    score = train_model(
        classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=batch_size,
        epochs=epochs, model_path=settings.MODELS_PATH, model_name=model_name, logs_path=log_dir
    )
    return score


def train_bbc_cnn(model_name, w2v_model, batch_size=128, epochs=100, length=400):
    embedding_matrix, word_index, train_seq_x, train_y, validation_seq_x, validation_y = prepare_training_datasets(
        settings.BBC_DATA_DIR, w2v_model, length
    )
    classifier = cnn(word_index, embedding_matrix, len(settings.BBC_TOPICS), length)
    print(classifier.summary())
    log_dir = get_log_dir(model_name, batch_size, epochs)
    os.makedirs(log_dir, exist_ok=True)
    score = train_model(
        classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=batch_size,
        epochs=epochs, model_path=settings.MODELS_PATH, model_name=model_name, logs_path=log_dir
    )
    return score


def train_bbc_rnn(model_name, w2v_model, batch_size=128, epochs=100, length=400):
    embedding_matrix, word_index, train_seq_x, train_y, validation_seq_x, validation_y = prepare_training_datasets(
        settings.BBC_DATA_DIR, w2v_model, length
    )
    classifier = rnn(word_index, embedding_matrix, len(settings.BBC_TOPICS), settings.PADDING_LENGTH)
    print(classifier.summary())
    log_dir = get_log_dir(model_name, batch_size, epochs)
    os.makedirs(log_dir, exist_ok=True)
    score = train_model(
        classifier, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=batch_size,
        epochs=epochs, model_path=settings.MODELS_PATH, model_name=model_name, logs_path=log_dir
    )
    return score
