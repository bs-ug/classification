import argparse

from libs.bbc import train_bbc_simple, train_bbc_cnn, train_bbc_rnn
from libs.cnn import train_cnn_simple, train_cnn_cnn, train_cnn_rnn
from libs.rz import train_rz_simple, train_rz_cnn, train_rz_rnn


def main():
    parser = argparse.ArgumentParser("data preparation")
    parser.add_argument("--dataset", type=str, choices=["bbc", "cnn", "rz"], required=True, help="dataset name")
    parser.add_argument("--nn_type", type=str, choices=["simple", "cnn", "rnn"], required=True, help="dataset name")
    parser.add_argument("--model", type=str, required=True, help="Neural Net model file name")
    parser.add_argument("--w2v", type=str, required=True, help="Word2Vec model filename")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--length", type=int, default=400, help="Articles length")
    args = parser.parse_args()
    if args.dataset == "bbc":
        if args.nn_type == "simple":
            train_bbc_simple(args.mode, args.w2v, args.batch_size, args.epochs, args.length)
        elif args.nn_type == "cnn":
            train_bbc_cnn(args.model, args.w2v, args.batch_size, args.epochs, args.length)
        else:
            train_bbc_rnn(args.model, args.w2v, args.batch_size, args.epochs, args.length)
    elif args.dataset == "cnn":
        if args.nn_type == "simple":
            train_cnn_simple(args.model, args.w2v, args.batch_size, args.epochs, args.length)
        elif args.nn_type == "cnn":
            train_cnn_cnn(args.model, args.w2v, args.batch_size, args.epochs, args.length)
        else:
            train_cnn_rnn(args.model, args.w2v, args.batch_size, args.epochs, args.length)
    else:
        if args.nn_type == "simple":
            train_rz_simple(args.model, args.w2v, args.batch_size, args.epochs, args.length)
        elif args.nn_type == "cnn":
            train_rz_cnn(args.model, args.w2v, args.batch_size, args.epochs, args.length)
        else:
            train_rz_rnn(args.model, args.w2v, args.batch_size, args.epochs, args.length)


if __name__ == '__main__':
    main()
