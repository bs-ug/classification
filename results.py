import argparse
import os

import numpy as np

import settings
from libs.utils import test_model, prepare_test_dataset


def main():
    parser = argparse.ArgumentParser("data preparation")
    parser.add_argument("--dataset", type=str, choices=["bbc", "cnn", "rz"], required=True, help="dataset name")
    parser.add_argument("--model", type=str, required=True, help="Neural Net model file name")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--length", type=int, default=400, help="Articles length")
    args = parser.parse_args()

    test_seq_x, test_y = prepare_test_dataset(os.path.join(settings.DATA_DIR, args.dataset), args.length)
    loss, acc, predictions, categories = test_model(
        os.path.join(settings.MODELS_PATH, args.model), test_seq_x, test_y, args.batch_size)
    print(f"test loss: {loss}\ntest accuracy: {acc}\npredictions: {predictions}")
    test_categories = np.argmax(test_y, axis=1)
    result = [(i, j) for i, j in zip(test_categories.tolist(), categories.tolist()) if i != j]
    results = {}
    for item in result:
        if results.get(item):
            results[item] += 1
        else:
            results[item] = 1
    print(results)


if __name__ == '__main__':
    main()
