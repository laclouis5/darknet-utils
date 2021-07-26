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
        """Image width in pixels."""
        return self.image_size[0]

    @property
    def image_height(self) -> int:
        """Image height in pixels."""
        return self.image_size[1]

    @property
    def image_name(self) -> str:
        """The image name."""
        return self.image_path.name
    
    def filter(self, is_included: Callable[[BoundingBox], bool]) -> "Annotation":
        """
        Filter bounding boxes given the box predicate.
        
        Parameters:
        - is_included: the box predicate.
        """
        self.boxes = [b for b in self.boxes if is_included(b)]
        return self

    def map_labels(self, mapping: Mapping[str, str]) -> "Annotation":
        """
        Translates the box label of all boxes according to a mapping.

        Parameters:
        - mapping: a dictionary of label translations.
        """
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
        assert 0.0 <= ratio <=1.0, "ratio should be in 0...1"

        side = min(self.image_size) * ratio / 2
        for box in self.boxes:
            if labels is not None and box.label in labels:
                xmid, ymid = box.xmid, box.ymid
                box._xmin = xmid - side
                box._ymin = ymid - side
                box._xmax = xmid + side
                box._ymax = ymid + side
        
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

    def __setitem__(self, index, value):
        self.annotations[index] = value

    def __iadd__(self, other: "Annotations") -> "Annotations":
        self.annotations += other.annotations
        return self
    
    def __add__(self, other: "Annotations") -> "Annotations":
        return Annotations(self.annotations + other.annotations)

    def append(self, annotation: Annotation):
        """Append an annotation to the annotations"""
        self.annotations.append(annotation)

    def image_paths(self) -> "list[Path]":
        """Returns the image paths of all the annotations."""
        return [a.image_path for a in self.annotations]

    def labels(self) -> "set[str]":
        """Returns the unique labels of all the annotations."""
        return {b.label for b in self.all_bounding_boxes()}  

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

    def filter(self, 
        is_included: Callable[[BoundingBox], bool],
    ) -> "Annotations":
        """
        Filter all bounding boxes given the box predicate.
        
        Parameters:
        - is_included: the box predicate.
        """
        for annotation in self.annotations:
            annotation.filter(is_included)
        return self

    def remove_empty(self) -> "Annotations":
        """Removes empty annotations."""
        self.annotations = [a for a in self.annotations if len(a.boxes) > 0]
        return self

    def print_stats(self) -> "Annotations":
        """Prints the annotations statistics."""
        table = Table(show_footer=True)

        box_count = defaultdict(int)
        image_count = defaultdict(set)

        for a in self.annotations:
            if len(a.boxes) == 0:
                image_count["<empty>"].add(a.image_path)
            for b in a.boxes:
                label = b.label
                box_count[label] += 1
                image_count[label].add(a.image_path)

        tot_boxes = sum(box_count.values())
        image_count = {l: len(p) for l, p in image_count.items()}
        tot_imgs = len(self)

        table.add_column("Label", "Total")
        table.add_column("Images", f"{tot_imgs}", justify="right")
        table.add_column("Boxes", f"{tot_boxes}", justify="right")

        for label in sorted(image_count.keys()):
            nb_boxes = box_count[label]
            nb_images = image_count[label]
            table.add_row(label, f"{nb_images}", f"{nb_boxes}")

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