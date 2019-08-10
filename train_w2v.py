import argparse

from libs.bbc import train_bbc_w2v
from libs.cnn import train_cnn_w2v
from libs.rz import train_rz_w2v


def main():
    parser = argparse.ArgumentParser("W2V model training")
    parser.add_argument("--dataset", type=str, choices=["bbc", "cnn", "rz"], required=True, help="dataset name")
    parser.add_argument("--w2v", type=str, required=True, help="Word2Vec model file name")
    parser.add_argument("--length", type=int, default=100, help="length of Word2Vec vectors, default is 100")
    args = parser.parse_args()
    if args.dataset == "bbc":
        train_rz_w2v(args.w2v, args.length)
    elif args.dataset == "cnn":
        train_cnn_w2v(args.w2v, args.length)
    else:
        train_bbc_w2v(args.w2v, args.length)


if __name__ == '__main__':
    main()
