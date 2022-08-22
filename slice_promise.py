# -*- coding: utf-8 -*-
import os
import random
import argparse
from preprocess.slice_promise import slice_promise


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, default="./data/promise")
    parser.add_argument('--dest_dir', type=str, default="./data/promise_WSS")

    parser.add_argument('--img_dir', type=str, default="IMG")
    parser.add_argument('--gt_dir', type=str, default="GT")
    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retain', type=int, default=10, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_augment', type=int, default=0, help="Number of augmentation to create per image, only for the training set")

    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args

if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    slice_promise(args)






