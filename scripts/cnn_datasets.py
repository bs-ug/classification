import json
import os

import pandas as pandas
from sklearn import preprocessing


def prepare_data(labels_dict, files_path):
    labels, texts = [], []
    for filename, label in labels_dict.items():
        with open(os.path.join(files_path, f"{filename}.txt"), "r", encoding="utf-8") as file:
            content = file.read()
        labels.append(label)
        texts.append(content)
    return labels, texts


def prepare_dataset(labels_file, files_path):
    encoder = preprocessing.LabelEncoder()
    with open(labels_file, "r", encoding="utf-8") as file:
        labels = json.load(file)
    dataset = pandas.DataFrame()
    dataset["labels"], dataset["text"] = prepare_data(labels, files_path)
    return dataset["text"], encoder.fit_transform(dataset["labels"])
