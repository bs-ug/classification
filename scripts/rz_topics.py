import json
import os
from random import choice

from scripts import settings


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
        pass
labels_list = [(key, value) for key, value in labels.items()]
while labels_list:
    item, value = choice(labels_list)
    labels_list.pop(labels_list.index((item, value)))
    with open(os.path.join(settings.RZ_SOURCE_FILES, f"{item}.txt"), "r", encoding="utf-8") as file:
        content = file.read()
    words = len(content.split())
    if settings.RZ_MIN_ARTICLE_LENGTH <= words:
        if global_counter[value]["train"] < settings.RZ_TRAIN_QUANTITY:
            train_files[item] = value
            with open(os.path.join(settings.RZ_TRAINING_FILES_PATH, f"{item}.txt"), "w", encoding="UTF-8") as file:
                file.write(content)
                global_counter[value]["train"] += 1
        elif global_counter[value]["validation"] < settings.RZ_VALIDATION_QUANTITY:
            validation_files[item] = value
            with open(os.path.join(settings.RZ_VALIDATION_FILES_PATH, f"{item}.txt"), "w", encoding="UTF-8") as file:
                file.write(content)
                global_counter[value]["validation"] += 1
        elif global_counter[value]["test"] < settings.RZ_TEST_QUANTITY:
            test_files[item] = value
            with open(os.path.join(settings.RZ_TEST_FILES_PATH, f"{item}.txt"), "w", encoding="UTF-8") as file:
                file.write(content)
                global_counter[value]["test"] += 1
for name, data in zip(["train", "validation", "test"], [train_files, validation_files, test_files]):
    with open(os.path.join(settings.RZ_DATA_DIR, f"{name}.json"), "w") as output_file:
        json.dump(data, output_file)
    print(f"{len(data)} {name} labels saved")
