import argparse

from libs.bbc import prepare_bbc_topics, train_bbc_w2v
from libs.cnn import prepare_cnn_topics, train_cnn_w2v
from libs.rz import prepare_rz_files, prepare_rz_topics, train_rz_w2v


def main():
    parser = argparse.ArgumentParser("data preparation")
    parser.add_argument("--dataset", type=str, choices=["bbc", "cnn", "rz"], required=True, help="dataset name")
    parser.add_argument("--w2v", type=str, help="when added will train Word2Vec model with given file name")
    args = parser.parse_args()
    if args.dataset == "bbc":
        prepare_rz_files()
        prepare_rz_topics()
        if args.w2v:
            train_rz_w2v(args.w2v)
    elif args.dataset == "cnn":
        prepare_cnn_topics()
        if args.w2v:
            train_cnn_w2v(args.w2v)
    else:
        prepare_bbc_topics()
        if args.w2v:
            train_bbc_w2v(args.w2v)


if __name__ == '__main__':
    main()
