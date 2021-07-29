#!/usr/bin/env python

from src import *

from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("folders", type=Path, nargs="+", 
        help="The folders to parse.")
    parser.add_argument("--recursive", "-r", action="store_true",
        help="Parse the folders recursively.")
    parser.add_argument("--save_dir", "-s", type=Path, default="yolo_trainval/",
        help="Where to store the created database.")
    parser.add_argument("--labels", "-l", nargs="+", default=None,
        help="The labels to consider for parsing. Default: all found labels.")

    parser.add_argument("--train_ratio", "-t", type=float, default=80/100,
        help="The percent of train samples.")
    parser.add_argument("--norm", "-m", nargs="*", default=None,
        help="A list of labels to normalize to a square bounding box. If \
            this option is specified without a list of labels all labels \
            are normalized.")
    parser.add_argument("--norm_ratio", "-n", type=float, default=7.5/100,
        help="The side length of the square bounding boxes expressed as \
            the percent of the image shortest side length.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    annotations = parse_xml_folders(
        folders=args.folders, 
        recursive=args.recursive,
        labels=args.labels)
    annotations.print_stats()

    labels = sorted(annotations.labels())
    labels_to_numbers = {l: str(n) for n, l in enumerate(labels)}

    if args.norm is not None and len(args.norm) == 0:
        annotations.square_boxes(ratio=args.norm_ratio, labels=None)
    elif args.norm is not None:
        annotations.square_boxes(ratio=args.norm_ratio, labels=args.norm)

    annotations.map_labels(labels_to_numbers)

    create_yolo_trainval(
        annotations=annotations, 
        save_dir=args.save_dir, 
        train_ratio=args.train_ratio, 
        exist_ok=True)

    (args.save_dir / "obj.names").write_text("\n".join(labels))