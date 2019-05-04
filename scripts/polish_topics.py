import json
import os
from random import choice
from shutil import copy

from scripts import settings

filtered_urls = []
with open(os.path.join(settings.POLISH_DATA_DIR, settings.POLISH_SOURCE_URLS_FILE), "r") as file:
    urls = json.load(file)
for name, url in urls.items():
    for key, value in settings.POLISH_TOPICS.items():
        stop = False
        for item in sorted(value, key=lambda x: len(x))[::-1]:
            if item in url.lower():
                filtered_urls.append((name, key))
train_files = {}
validation_files = {}
test_files = {}
global_counter = {key: {"test": 0, "train": 0, "validation": 0} for key in settings.POLISH_TOPICS.keys()}
for path in ["train", "validation", "test"]:
    try:
        os.mkdir(os.path.join(settings.POLISH_DATA_DIR, path))
    except FileExistsError:
        pass
while filtered_urls:
    item, value = choice(filtered_urls)
    filtered_urls.pop(filtered_urls.index((item, value)))
    with open(os.path.join(settings.POLISH_SOURCE_FILES, f"{item}.txt"), "r", encoding="utf-8") as file:
        words = len(file.read().split())
    if words >= settings.POLISH_MIN_ARTICLE_LENGTH:
        if global_counter[value]["train"] < settings.POLISH_TRAIN_QUANTITY:
            global_counter[value]["train"] += 1
            train_files[item] = value
            copy(
                os.path.join(settings.POLISH_SOURCE_FILES, f"{item}.txt"),
                os.path.join(settings.POLISH_TRAINING_FILES_PATH, f"{item}.txt")
            )
        elif global_counter[value]["validation"] < settings.POLISH_VALIDATION_QUANTITY:
            global_counter[value]["validation"] += 1
            validation_files[item] = value
            copy(
                os.path.join(settings.POLISH_SOURCE_FILES, f"{item}.txt"),
                os.path.join(settings.POLISH_VALIDATION_FILES_PATH, f"{item}.txt")
            )
        elif global_counter[value]["test"] < settings.POLISH_TEST_QUANTITY:
            global_counter[value]["test"] += 1
            test_files[item] = value
            copy(
                os.path.join(settings.POLISH_SOURCE_FILES, f"{item}.txt"),
                os.path.join(settings.POLISH_TEST_FILES_PATH, f"{item}.txt")
            )
for name, data in zip(["train", "validation", "test"], [train_files, validation_files, test_files]):
    with open(os.path.join(settings.POLISH_DATA_DIR, f"{name}.json"), "w") as output_file:
        json.dump(data, output_file)
    print(f"{len(data)} {name} labels saved")
