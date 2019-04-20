import hashlib
import json
import os

from scripts import settings


def hash_hex(s):
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


urls = []
train_files = {}
validation_files = {}
test_files = {}
global_counter = {key: {"test": 0, "train": 0, "validation": 0} for key in settings.CNN_TOPICS.keys()}
with open(os.path.join(settings.CNN_DATA_DIR, settings.CNN_URLS_FILE), "r") as file:
    for line in file:
        urls.append(line.strip('\n'))
for url in urls:
    for key, value in settings.CNN_TOPICS.items():
        stop = False
        for item in sorted(value, key=lambda x: len(x))[::-1]:
            if item in url.lower():
                if os.path.isfile(os.path.join(settings.CNN_SOURCE_FILES, f"{hash_hex(url)}.story")):
                    if global_counter[key]["train"] < settings.TRAIN_QUANTITY:
                        global_counter[key]["train"] += 1
                        train_files[hash_hex(url)] = key
                    elif global_counter[key]["validation"] < settings.VALIDATION_QUANTITY:
                        global_counter[key]["validation"] += 1
                        validation_files[hash_hex(url)] = key
                    elif global_counter[key]["test"] < settings.TEST_QUANTITY:
                        global_counter[key]["test"] += 1
                        test_files[hash_hex(url)] = key
                    stop = True
                    break
        if stop:
            break
for name, data in zip(["train", "validation", "test"], [train_files, validation_files, test_files]):
    with open(os.path.join(settings.CNN_DATA_DIR, f"{name}.json"), "w") as output_file:
        json.dump(data, output_file)
    print(f"{len(data)} {name} labels saved")
