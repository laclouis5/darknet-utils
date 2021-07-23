from .bounding_box import BoundingBox
from .utils import *

from typing import Callable, KeysView, Sequence, Union, ItemsView, ValuesView
from copy import copy
from itertools import chain

from rich.table import Table
from rich import print as rprint


class BoundingBoxes:
    """
    Represents a set of images with associated object
    annotations. Works similar to a dictionary.
    """

    def __init__(self, boxes: "dict[str, list[BoundingBox]]" = None):
        self.boxes = boxes or {}

    @property
    def image_names(self) -> KeysView[str]:
        """Image names annotated"""
        return self.boxes.keys()

    def keys(self) -> KeysView[str]:
        """Keys of the underlying dictionary, i.e. view into the image names."""
        return self.boxes.keys()

    def values(self) -> ValuesView["list[BoundingBox]"]:
        """Values of the underlying dictionary, i.e. view into lists of boxes."""
        return self.boxes.values()

    def items(self) -> ItemsView[str, "list[BoundingBox]"]:
        """Iterator of (key, value) pair of the underlying dictionary iterator."""
        return self.boxes.items()

    def __ior__(self, other: Union["BoundingBoxes", "dict[str, list[BoundingBox]]"]) -> "BoundingBoxes":
        """
        Add to this BoundingBox object annotations from another one or an equivalent dictionary.
        """
        if isinstance(other, dict):
            self.boxes.update(other)
            return self
        elif isinstance(other, BoundingBoxes):
            self.boxes.update(other.boxes)
            return self
        else:
            raise NotImplementedError

    def __or__(self, other: Union["BoundingBoxes", "dict[str, list[BoundingBox]]"]) -> "BoundingBoxes":
        """
        Create a new BoundingBox object that holds the annotations from the two operands.
        """
        c = copy(self)
        c |= other
        return c

    def __len__(self) -> int:
        """The number of annotated images."""
        return len(self.boxes)

    def __getitem__(self, key) -> "list[BoundingBox]":
        """Retreives a list of BoundingBox associated with an image"""
        return self.boxes[key]

    def all_boxes(self) -> Sequence[BoundingBox]:
        """Iterator yielding all the bounding boxes"""
        return chain(*self.values())

    def filtered(self, is_included: "Callable[[BoundingBox], bool]") -> "BoundingBoxes":
        """
        Filter the BoundingBoxes given the box predicate and return the result.

        Parameters:
        - is_included: the box predicate
        """
        return BoundingBoxes({l: [b for b in boxes if is_included(b)] for l, boxes in self.items()})

    def filter(self, is_included: "Callable[[BoundingBox], bool]") -> "BoundingBoxes":
        """
        Filter the BoundingBoxes given the box predicate.

        Parameters:
        - is_included: the box predicate
        """
        self.boxes = {l: [b for b in boxes if is_included(b)] for l, boxes in self.items()}
        return self

    def map_labels(self, mapping: "dict[str, str]") -> "BoundingBoxes":
        """
        Translates the box label of all the boxes according to a mapping.

        Parameters:
        - mapping: a dictionary of label names translations
        """
        for b in self.all_boxes():
            b.label = mapping[b.label]
        return self

    def print_stats(self):
        """Prints the dataset statistics represented by this object."""
        tot_imgs = len(self)
        all_boxes = group(self.all_boxes(), by_key=lambda v: v.label)
        tot_boxes = sum(len(v) for v in all_boxes.values())

        table = Table(title=f"{tot_imgs} Images", show_footer=True)

        table.add_column("Label", "Total")
        table.add_column("Boxes", f"{tot_boxes}", justify="right")

        for label in sorted(all_boxes.keys()):
            boxes = all_boxes[label]
            table.add_row(label, f"{len(boxes)}")

        rprint(table)
