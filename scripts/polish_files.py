import json
import os
from shutil import copy

from scripts import settings

for path, labels_file_name in zip(
        [settings.POLISH_TRAINING_FILES_PATH, settings.POLISH_VALIDATION_FILES_PATH, settings.POLISH_TEST_FILES_PATH],
        [settings.POLISH_TRAINING_LABELS, settings.POLISH_VALIDATION_LABELS, settings.POLISH_TEST_LABELS]):
    with open(os.path.join(settings.POLISH_DATA_DIR, labels_file_name), "r") as file:
        list_of_files = json.load(file)
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    for file_name in list_of_files.keys():
        copy(
            os.path.join(settings.POLISH_SOURCE_FILES, f"{file_name}.txt"),
            os.path.join(path, f"{file_name}.txt")
        )
