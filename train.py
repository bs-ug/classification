import argparse
import os
import settings
from libs.bbc import train_bbc_simple, train_bbc_conv, train_bbc_lstm
from libs.cnn import train_cnn_simple, train_cnn_conv, train_cnn_lstm
from libs.rz import train_rz_simple, train_rz_conv, train_rz_lstm


def main():
    parser = argparse.ArgumentParser("model training")
    parser.add_argument("--dataset", type=str, choices=["bbc", "cnn", "rz"], required=True, help="dataset name")
    parser.add_argument("--nn_type", type=str, choices=["simple", "conv", "lstm"], required=True, help="dataset name")
    parser.add_argument("--model", type=str, required=True, help="Neural Net model file name")
    parser.add_argument("--w2v", type=str, required=True, help="Word2Vec model filename")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--length", type=int, default=400, help="Articles length")
    args = parser.parse_args()
    with open(os.path.join(settings.MODELS_PATH, args.w2v), "r") as model_file:
        first_line = model_file.readline().split(" ")
        try:
            vector_length = int(first_line[1])
            assert type(vector_length) == int
        except (AssertionError, ValueError) :
            vector_length = len(first_line) - 1
    print(vector_length)
    if args.dataset == "bbc":
        if args.nn_type == "simple":
            train_bbc_simple(args.model, args.w2v, vector_length, args.batch_size, args.epochs, args.length)
        elif args.nn_type == "conv":
            train_bbc_conv(args.model, args.w2v, vector_length, args.batch_size, args.epochs, args.length)
        else:
            train_bbc_lstm(args.model, args.w2v, vector_length, args.batch_size, args.epochs, args.length)
    elif args.dataset == "cnn":
        if args.nn_type == "simple":
            train_cnn_simple(args.model, args.w2v, vector_length, args.batch_size, args.epochs, args.length)
        elif args.nn_type == "conv":
            train_cnn_conv(args.model, args.w2v, vector_length, args.batch_size, args.epochs, args.length)
        else:
            train_cnn_lstm(args.model, args.w2v, vector_length, args.batch_size, args.epochs, args.length)
    else:
        if args.nn_type == "simple":
            train_rz_simple(args.model, args.w2v, vector_length, args.batch_size, args.epochs, args.length)
        elif args.nn_type == "conv":
            train_rz_conv(args.model, args.w2v, vector_length, args.batch_size, args.epochs, args.length)
        else:
            train_rz_lstm(args.model, args.w2v, vector_length, args.batch_size, args.epochs, args.length)


if __name__ == '__main__':
    main()
