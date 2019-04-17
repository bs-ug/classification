import hashlib
import json
import os

CNN_TOPICS = {
    0: ["/crime/"],
    1: ["/health/"],
    2: ["/politics/"],
    3: ["/showbiz/"],
    4: ["/sport/"],
    5: ["/tech/"],
    6: ["/travel/"],
    7: ["/us/"],
    8: ["/africa/", "/world/africa/"],
    9: ["/world/americas/"],
    10: ["/world/asiapcf/", "/asia/", "/world/asia/"],
    11: ["/europe/", "/world/europe/"],
    12: ["/middleeast/", "/world/meast/"],
    13: ["/living/"],
    14: ["/opinion/", "/opinions/"]
}
CNN_DATA_DIR = "../data/cnn"
CNN_URLS = "wayback_training_urls.txt"
SOURCE_FILES = "../data/cnn/stories"
TRAIN_QUANTITY = 2500
VALIDATION_QUANTITY = 100
TEST_QUANTITY = 100


def hash_hex(s):
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


urls = []
train_files = {}
validation_files = {}
test_files = {}
global_counter = {key: {"test": 0, "train": 0, "validation": 0} for key in CNN_TOPICS.keys()}
with open(os.path.join(CNN_DATA_DIR, CNN_URLS), "r") as file:
    for line in file:
        urls.append(line.strip('\n'))
for url in urls:
    for key, value in CNN_TOPICS.items():
        stop = False
        for item in sorted(value, key=lambda x: len(x))[::-1]:
            if item in url.lower():
                if os.path.isfile(os.path.join(SOURCE_FILES, f"{hash_hex(url)}.story")):
                    if global_counter[key]["train"] < TRAIN_QUANTITY:
                        global_counter[key]["train"] += 1
                        train_files[hash_hex(url)] = key
                    elif global_counter[key]["validation"] < VALIDATION_QUANTITY:
                        global_counter[key]["validation"] += 1
                        validation_files[hash_hex(url)] = key
                    elif global_counter[key]["test"] < TEST_QUANTITY:
                        global_counter[key]["test"] += 1
                        test_files[hash_hex(url)] = key
                    stop = True
                    break
        if stop:
            break
for name, data in zip(["train", "validation", "test"], [train_files, validation_files, test_files]):
    with open(os.path.join(CNN_DATA_DIR, f"{name}.json"), "w") as output_file:
        json.dump(data, output_file)
    print(f"{len(data)} {name} labels saved")
