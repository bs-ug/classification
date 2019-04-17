import json
import os
from shutil import copy

CNN_DATA_DIR = "../data/cnn/"
SOURCE_FILES = "../data/cnn/stories"

for name in ["train", "validation", "test"]:
    with open(os.path.join(CNN_DATA_DIR, f"{name}.json"), "r") as file:
        list_of_files = json.load(file)
    try:
        os.mkdir(os.path.join(CNN_DATA_DIR, name))
    except FileExistsError:
        pass
    for file_name in list_of_files.keys():
        copy(
            os.path.join(SOURCE_FILES, f"{file_name}.story"),
            os.path.join(CNN_DATA_DIR, name, f"{file_name}.txt")
        )
