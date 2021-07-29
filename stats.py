#!/usr/bin/env python

from src import *

from pathlib import Path
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Print stats about the XML annotations contained in specified directories.")

    parser.add_argument("folders", type=Path, nargs="+", 
        help="The folders to parse.")
    parser.add_argument("--recursive", "-r", action="store_true",
        help="Weither to parse directories recursively or not.")
    parser.add_argument("--labels", "-l", nargs="*", type=str, default=None,
        help="The labels to parse.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    annotations = parse_xml_folders(
        args.folders, 
        recursive=args.recursive, 
        labels=args.labels)
    annotations.print_stats()