from .bounding_box import BoundingBox
from .utils import *

from typing import Callable, Iterator, Mapping, Sequence
from os import PathLike
from pathlib import Path

from rich.table import Table
from rich import print as rprint


class Annotation:
    """
    Bounding box annotations for one image.
    """

    def __init__(self, 
        image_path: PathLike, 
        image_size: "tuple[int, int]", 
        boxes: "list[BoundingBox]" = None
    ):
        self.image_path = Path(image_path)
        self.image_size = image_size
        self.boxes = boxes or []

    @property
    def image_width(self) -> int:
        return self.image_size[0]

    @property
    def image_height(self) -> int:
        return self.image_size[1]

    @property
    def image_name(self) -> str:
        return self.image_path.name
    
    def filter(self, is_included: Callable[[BoundingBox], bool]) -> "Annotation":
        self.boxes = [b for b in self.boxes if is_included(b)]
        return self

    def filtered(self, is_included: Callable[[BoundingBox], bool]) -> "Annotation":
        return Annotation(
            self.image_path,
            self.image_size,
            [b for b in self.boxes if is_included(b)])

    def map_labels(self, mapping: Mapping[str, str]) -> "Annotation":
        for box in self.boxes:
            box.label = mapping[box.label]
        return self

    def square_boxes(self, ratio: float, labels: Sequence[str] = None) -> "Annotation":
        """
        Transform bounding boxes to be of square shape if their label
        is included in a given list.

        Parameters:
        - ratio: the percent of the minimum image side size to use
        as the bounding box side length.
        - labels: the labels of boxes to be be transformed. By default
        all bounding boxes are transformed.

        Returns:
        - The transformed annotation.
        """
        assert 0 <= ratio <=1, "ratio should be in 0...1"

        ratio *= min(self.image_size) / 2
        width = self.image_width * ratio
        height = self.image_height * ratio
        
        for box in self.boxes:
            if labels is not None and box.label in labels:
                xmid, ymid = box.xmid, box.ymid
                box._xmin = xmid - width
                box._ymin = ymid - height
                box._xmax = xmid + width
                box._ymax = ymid + height
        
        return self

    def yolo_repr(self, include_confidence=True) -> str:
        """
        The YOLO representation of the annotation:

        ```
        <label1> <confidence1> <xmid1> <ymid1> <width1> <height1>
        <label2> <confidence2> <xmid2> <ymid2> <width2> <height2>
        ...
        ```

        Parameters:
        - include_confidence: if True, bounding box confidence scores
        are included if present.

        Returns:
        - The string representation.
        """
        return "\n".join(b.yolo_repr(self.image_size, include_confidence) for b in self.boxes)

    
class Annotations:

    def __init__(self, annotations: "list[Annotation]" = None):
        self.annotations = annotations or []

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index) -> Annotation:
        return self.annotations[index]

    def __iadd__(self, other: "Annotations") -> "Annotations":
        self.annotations += other.annotations
        return self
    
    def __add__(self, other: "Annotations") -> "Annotations":
        return Annotations(self.annotations + other.annotations)

    def append(self, annotation: Annotation):
        """Append an annotation to the annotations"""
        self.annotations.append(annotation)

    def all_bounding_boxes(self) -> Iterator[BoundingBox]:
        """Iterator of all bounding boxes."""
        for annotation in self.annotations:
            yield from annotation.boxes

    def map_labels(self, map: Mapping[str, str]) -> "Annotations":
        """
        Translates the box label of all the boxes according to a mapping.

        Parameters:
        - mapping: a dictionary of label names translations
        """
        for annotation in self.annotations:
            annotation.map_labels(map)
        return self

    def filter(self, is_included: Callable[[BoundingBox], bool]) -> "Annotations":
        """
        Filter all bounding boxes given the box predicate.

        Parameters:
        - is_included: the box predicate
        """
        for annotation in self.annotations:
            annotation.filter(is_included)
        return self

    def filtered(self, is_included: Callable[[BoundingBox], bool]) -> "Annotations":
        """
        Filter all bounding boxes given the box predicate and return a copy.

        Parameters:
        - is_included: the box predicate
        """
        return Annotations([a.filtered(is_included) for a in self.annotations])

    def print_stats(self) -> "Annotations":
        """Prints the annotations statistics."""
        tot_imgs = len(self)
        all_boxes = dict_grouping(self.all_bounding_boxes(), by_key=lambda v: v.label)
        tot_boxes = sum(len(v) for v in all_boxes.values())

        table = Table(title=f"{tot_imgs} Image{'s' if tot_imgs > 1 else ''}", show_footer=True)

        table.add_column("Label", "Total")
        table.add_column("Boxes", f"{tot_boxes}", justify="right")

        for label in sorted(all_boxes.keys()):
            nb_boxes = len(all_boxes[label])
            table.add_row(label, f"{nb_boxes}")

        rprint(table)

        return self

    def square_boxes(self, ratio: float, labels: Sequence[str] = None) -> "Annotations":
        """
        Transform bounding boxes to be of square shape if their label
        is included in a given list.

        Parameters:
        - ratio: the percent of the minimum image side size to use
        as the bounding box side length.
        - labels: the labels of boxes to be be transformed. By default
        all bounding boxes are transformed.

        Returns:
        - The transformed annotations.
        """
        for annotation in self.annotations:
            annotation.square_boxes(ratio, labels)
        return self