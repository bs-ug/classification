import json
import os
from glob import glob

from scripts import settings


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
                with open(os.path.join(settings.BBC_DATA_DIR, settings.TRAINING_FILES, f"{filename}.txt"), "w", encoding="UTF-8") as file:
                    file.write(content)
                    global_counter[label]["train"] += 1
            elif global_counter[label]["validation"] < settings.BBC_VALIDATION_QUANTITY:
                validation_files[filename] = label
                with open(os.path.join(settings.BBC_DATA_DIR, settings.VALIDATION_FILES, f"{filename}.txt"), "w", encoding="UTF-8") as file:
                    file.write(content)
                    global_counter[label]["validation"] += 1
            elif global_counter[label]["test"] < settings.BBC_TEST_QUANTITY:
                test_files[filename] = label
                with open(os.path.join(settings.BBC_DATA_DIR, settings.TEST_FILES, f"{filename}.txt"), "w", encoding="UTF-8") as file:
                    file.write(content)
                    global_counter[label]["test"] += 1
for name, data in zip(["train", "validation", "test"], [train_files, validation_files, test_files]):
    with open(os.path.join(settings.BBC_DATA_DIR, f"{name}.json"), "w") as output_file:
        json.dump(data, output_file)
    print(f"{len(data)} {name} labels saved")

print(global_counter)
