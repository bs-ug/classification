import argparse

from libs.bbc import prepare_bbc_topics
from libs.cnn import prepare_cnn_topics
from libs.rz import prepare_rz_files, prepare_rz_topics


def main():
    parser = argparse.ArgumentParser("data preparation")
    parser.add_argument("--dataset", type=str, choices=["bbc", "cnn", "rz"], required=True, help="dataset name")
    args = parser.parse_args()
    if args.dataset == "bbc":
        prepare_rz_files()
        prepare_rz_topics()
    elif args.dataset == "cnn":
        prepare_cnn_topics()
    else:
        prepare_bbc_topics()


if __name__ == '__main__':
    main()
