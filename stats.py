#!/usr/bin/env python

from darknet_utils import *

from pathlib import Path
from argparse import ArgumentParser

from rich.table import Table
from rich import print as rprint


def parse_args():
    parser = ArgumentParser(description="Print stats about the XML annotations contained in specified directories.")

    parser.add_argument("folders", type=Path, nargs="+", 
        help="The folders to parse.")
    parser.add_argument("--recursive", "-r", action="store_true",
        help="Weither to parse directories recursively or not.")
    parser.add_argument("--labels", "-l", nargs="+", type=str, default=None,
        help="The labels to parse.")
    parser.add_argument("--show_empty", "-e", action="store_true",
        help="Include empty annotations.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    annotations = parse_xml_folders(
        args.folders, 
        recursive=args.recursive, 
        labels=args.labels)

    if args.labels is not None and not args.show_empty:
        annotations.remove_empty()

    annotations.print_stats()

    sizes = defaultdict(list)
    for annotation in annotations:
        img_g, img_h = annotation.image_size

        for box in annotation.boxes:
            area = box.area / img_g / img_h
            sizes[box.label].append(area)

    total = [s for v in sizes.values() for s in v]
    total = sum(total) / len(total)
    sizes = {l: sum(v) / len(v) for l, v in sizes.items()}

    table = Table(title="Mean box area", show_footer=True)
    table.add_column("Label", footer="Total")
    table.add_column("Mean Area", footer=f"{total:.2%}", justify="right")

    for label, area in sizes.items():
        table.add_row(label, f"{area:.2%}")

    rprint(table)