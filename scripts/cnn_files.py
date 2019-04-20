import json
import os
from shutil import copy

from scripts import settings

for path, labels_file_name in zip(
        [settings.CNN_TRAINING_FILES_PATH, settings.CNN_VALIDATION_FILES_PATH, settings.CNN_TEST_FILES_PATH],
        [settings.CNN_TRAINING_LABELS, settings.CNN_VALIDATION_LABELS, settings.CNN_TEST_LABELS]):
    with open(os.path.join(settings.CNN_DATA_DIR, labels_file_name), "r") as file:
        list_of_files = json.load(file)
    try:
        os.mkdir(settings.CNN_TRAINING_FILES_PATH)
    except FileExistsError:
        pass
    for file_name in list_of_files.keys():
        copy(
            os.path.join(settings.CNN_SOURCE_FILES, f"{file_name}.story"),
            os.path.join(path, f"{file_name}.txt")
        )
        with open(os.path.join(path, f"{file_name}.txt"), "r", encoding="UTF-8") as file:
            news = file.read()
        news = news.split()
        news = news[:news.index("@highlight")]
        with open(os.path.join(path, f"{file_name}.txt"), "w", encoding="UTF-8") as file:
            file.write(" ".join(news))
