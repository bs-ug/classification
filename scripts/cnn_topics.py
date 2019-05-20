import hashlib
import json
import os
from glob import glob
from random import choice

from scripts import settings


def hash_hex(s):
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


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
            with open(os.path.join(settings.CNN_DATA_DIR, settings.TRAINING_FILES, f"{item}.txt"), "w", encoding="UTF-8") as file:
                file.write(" ".join(news))
                global_counter[value]["train"] += 1
        elif global_counter[value]["validation"] < settings.CNN_VALIDATION_QUANTITY:
            validation_files[item] = value
            with open(os.path.join(settings.CNN_DATA_DIR, settings.VALIDATION_FILES, f"{item}.txt"), "w", encoding="UTF-8") as file:
                file.write(" ".join(news))
                global_counter[value]["validation"] += 1
        elif global_counter[value]["test"] < settings.CNN_TEST_QUANTITY:
            test_files[item] = value
            with open(os.path.join(settings.CNN_DATA_DIR, settings.TEST_FILES, f"{item}.txt"), "w", encoding="UTF-8") as file:
                file.write(" ".join(news))
                global_counter[value]["test"] += 1
for name, data in zip(["train", "validation", "test"], [train_files, validation_files, test_files]):
    with open(os.path.join(settings.CNN_DATA_DIR, f"{name}.json"), "w") as output_file:
        json.dump(data, output_file)
    print(f"{len(data)} {name} labels saved")
