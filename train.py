import argparse
import os
import settings
from libs.utils import factory
from libs.bbc import train_bbc
from libs.cnn import train_cnn
from libs.rz import train_rz


def main():
    parser = argparse.ArgumentParser("model training")
    parser.add_argument("--dataset", type=str, choices=["bbc", "cnn", "rz"], required=True, help="Dataset name")
    parser.add_argument("--nn_type", type=str, choices=["simple", "conv", "lstm"], required=True, help="Neural Network type")
    parser.add_argument("--monitor", type=str, choices=["loss", "acc", "val_loss", "val_acc"], required=True, help="Quantity to monitor")
    parser.add_argument("--model", type=str, required=True, help="Neural Net model file name")
    parser.add_argument("--w2v", type=str, required=True, help="Word2Vec model filename")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--length", type=int, default=400, help="Articles length (characters)")
    args = parser.parse_args()
    with open(os.path.join(settings.MODELS_PATH, args.w2v), "r") as model_file:
        first_line = model_file.readline().split(" ")
        try:
            vector_length = int(first_line[1])
            assert type(vector_length) == int
        except (AssertionError, ValueError) :
            vector_length = len(first_line) - 1
    print(f"Word embeddings vector length: {vector_length}")
    if args.dataset == "bbc":
        train_bbc(factory[args.nn_type], args.monitor, args.model, args.w2v, vector_length, args.batch_size, args.epochs, args.length)
    elif args.dataset == "cnn":
        train_cnn(factory[args.nn_type], args.monitor, args.model, args.w2v, vector_length, args.batch_size, args.epochs, args.length)
    else:
        train_rz(factory[args.nn_type], args.monitor, args.model, args.w2v, vector_length, args.batch_size, args.epochs, args.length)


if __name__ == '__main__':
    main()
