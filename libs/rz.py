import json
import os
import re
from glob import glob
from random import choice

from gensim.models import Word2Vec

import settings
from .utils import text_generator, W2VCallback, prepare_training_datasets, train_model, get_log_dir


def prepare_rz_files():
    pattern_category = '<META NAME="DZIAL" CONTENT="([, A-Ża-ż0-9\/\"\-]*)">'
    pattern_p = "<[pP]?>"
    pattern_meta = "<META NAME"
    pattern_tag = "<[ _\-\.=a-zA-Z0-9/\"]+>"
    pattern_n = "\n *"
    pattern_blank = "  +"
    categories = {}
    os.makedirs(os.path.join(settings.DATA_DIR, "rz", "source"), exist_ok=True)
    for file in glob(os.path.join(settings.DATA_DIR, "rz", "Rzeczpospolita", "*.html")):
        with open(file, "r", encoding="iso-8859-2") as source:
            source_text = source.read()
        file_name = file.split('\\')[-1].split('.')[0]
        result = re.search(pattern_category, source_text)
        try:
            category = source_text[result.regs[1][0]:result.regs[1][1]]
            if category in settings.RZ_TOPICS.keys():
                categories[file_name] = settings.RZ_TOPICS[category]
        except AttributeError:
            print(f"category: {file_name}")
        if os.path.exists(os.path.join(settings.DATA_DIR, "rz", "source", f"{file_name}.txt")):
            continue
        result = re.search(pattern_p, source_text)
        try:
            source_text = source_text[result.regs[0][1]:]
        except AttributeError:
            print(f"p: {file_name}")
        result = re.search(pattern_meta, source_text)
        try:
            source_text = source_text[:result.regs[0][0]]
        except AttributeError:
            print(f"meta: {file_name}")
        result = re.search(pattern_tag, source_text)
        while result:
            source_text = source_text[:result.regs[0][0]] + " " + source_text[result.regs[0][1]:]
            result = re.search(pattern_tag, source_text)
        result = re.search(pattern_n, source_text)
        while result:
            source_text = source_text[:result.regs[0][0]] + source_text[result.regs[0][1]:]
            result = re.search(pattern_n, source_text)
        result = re.search(pattern_blank, source_text)
        while result:
            source_text = source_text[:result.regs[0][0]] + source_text[result.regs[0][1] - 1:]
            result = re.search(pattern_blank, source_text)
        with open(os.path.join(settings.DATA_DIR, "rz", "source", f"{file_name}.txt"), "w", encoding="utf-8") as output:
            output.write(source_text)
    with open(os.path.join(settings.RZ_DATA_DIR, settings.RZ_LABELS), "w", encoding="utf-8") as labels_file:
        json.dump(categories, labels_file)


def prepare_rz_topics():
    with open(os.path.join(settings.RZ_DATA_DIR, settings.RZ_LABELS), "r") as file:
        labels = json.load(file)
    train_files = {}
    validation_files = {}
    test_files = {}
    global_counter = {value: {"test": 0, "train": 0, "validation": 0} for value in settings.RZ_TOPICS.values()}
    for path in ["train", "validation", "test"]:
        try:
            os.mkdir(os.path.join(settings.RZ_DATA_DIR, path))
        except FileExistsError:
            files = glob(os.path.join(settings.RZ_DATA_DIR, path, "*.txt"))
            for file in files:
                os.remove(file)
    labels_list = [(key, value) for key, value in labels.items()]
    while labels_list:
        item, value = choice(labels_list)
        labels_list.pop(labels_list.index((item, value)))
        with open(os.path.join(settings.RZ_SOURCE_FILES, f"{item}.txt"), "r", encoding="utf-8") as file:
            content = file.read()
        words = len(content.split())
        if words >= settings.RZ_MIN_ARTICLE_LENGTH:
            if global_counter[value]["train"] < settings.RZ_TRAIN_QUANTITY:
                train_files[item] = value
                with open(os.path.join(settings.RZ_DATA_DIR, settings.TRAINING_FILES, f"{item}.txt"), "w",
                          encoding="UTF-8") as file:
                    file.write(content)
                    global_counter[value]["train"] += 1
            elif global_counter[value]["validation"] < settings.RZ_VALIDATION_QUANTITY:
                validation_files[item] = value
                with open(os.path.join(settings.RZ_DATA_DIR, settings.VALIDATION_FILES, f"{item}.txt"), "w",
                          encoding="UTF-8") as file:
                    file.write(content)
                    global_counter[value]["validation"] += 1
            elif global_counter[value]["test"] < settings.RZ_TEST_QUANTITY:
                test_files[item] = value
                with open(os.path.join(settings.RZ_DATA_DIR, settings.TEST_FILES, f"{item}.txt"), "w",
                          encoding="UTF-8") as file:
                    file.write(content)
                    global_counter[value]["test"] += 1
    for name, data in zip(["train", "validation", "test"], [train_files, validation_files, test_files]):
        with open(os.path.join(settings.RZ_DATA_DIR, f"{name}.json"), "w") as output_file:
            json.dump(data, output_file)
        print(f"{len(data)} {name} labels saved")
    print(global_counter)


def train_rz_w2v(filename, vector_length):
    iterable = text_generator(os.path.join(settings.RZ_SOURCE_FILES, "*.txt"))
    model = Word2Vec(
        [item for item in iterable],
        size=vector_length,
        sg=1, window=5, min_count=1, workers=4,
        compute_loss=True,
        callbacks=[W2VCallback()]
    )
    model.wv.save_word2vec_format(os.path.join(settings.MODELS_PATH, filename), binary=False)


def train_rz(network_type, monitor, model_name, w2v_model, vector_length, batch_size=128, epochs=100, text_length=400):
    embedding_matrix, word_index, train_seq_x, train_y, validation_seq_x, validation_y = prepare_training_datasets(
        settings.RZ_DATA_DIR, w2v_model, vector_length, text_length
    )
    classifier = network_type(word_index, embedding_matrix, len(settings.RZ_TOPICS), text_length, vector_length)
    print(classifier.summary())
    log_dir = get_log_dir(model_name, w2v_model, batch_size, epochs)
    os.makedirs(log_dir, exist_ok=True)
    score = train_model(
        classifier, monitor, train_seq_x, train_y, validation_seq_x, validation_y, batch_size=batch_size,
        epochs=epochs, model_path=settings.MODELS_PATH, model_name=model_name, logs_path=log_dir
    )
    return score
