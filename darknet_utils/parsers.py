import logging
from .bounding_box import BoundingBox
from .annotation import Annotation, Annotations

from os import PathLike
from pathlib import Path
from typing import Sequence
import logging

import lxml.etree as ET


logging.basicConfig(filename="parser.log")

def parse_xml_file(file: PathLike, labels: Sequence[str] = None) -> Annotation:
    """
    Parse an .xml file annotated with labelImg of other
    software that used the same anntotation format.

    Take a look at the source code to check if the parser
    is suitable for your annotation format. Bounding boxes
    annotations are in absolute coordinates and use the 
    top-left (xmin, ymin) and bottom-right (xmax, ymax) scheme.

    This parser will keep empty annotations but will remove
    any annotation that results in being empty because it
    does not contain at least one bounding box with a label 
    in `labels`. If may remove such empty annotation manually 
    or use the `.remove_empty()` method of `Annotations`.
    
    It will also return `None` if the fil is not readable 
    and log it in `parser.log` file.

    Parameters:
    - file: the xml file to process.
    - labels: a set of box labels to parse.
    
    Returns:
    - An object representing the image annotations or None if 
    the .xml file was not readable.
    """

    try:
        tree = ET.parse(str(file)).getroot()
        
        path = tree.find("path").text
        name = tree.find("filename").text

        img_size_node = tree.find("size")
        img_w = int(img_size_node.find("width").text)
        img_h = int(img_size_node.find("height").text)

        object_nodes = tree.findall("object")
        boxes = (_read_xml_object(o, labels) for o in object_nodes)
        boxes = [box for box in boxes if box]

        # Remove empty annotations resulting from the box label filtering
        if len(object_nodes) != 0 and len(boxes) == 0:
            return None
    except ET.ParseError:
        logging.warning(f"Error while reading '{file}'.")
        return None

    image_path = Path(path).with_name(name).expanduser().resolve()

    return Annotation(image_path, (img_w, img_h), boxes)


def parse_xml_folder(
    folder: PathLike, 
    recursive: bool = False, 
    labels: Sequence[str] = None
) -> Annotations:
    """
    Parse .xml annotations present in a folder. See `parse_xml`
    for more details.

    Parameters:
    - folder: a path to a folder containing .xml annotations.
    - labels: a set of box labels to parse.

    Returns:
    - A list of annotations.
    """
    folder = Path(folder).expanduser().resolve()
    files = folder.glob("**/*.xml" if recursive else "*.xml")
    return Annotations([ann for f in files if (ann := parse_xml_file(f, labels))])


def parse_xml_folders(
    folders: "list[PathLike]", 
    recursive=False,
    labels: Sequence[str] = None) -> Annotations:
    """
    Parse .xml annotations present in several folders. See `parse_xml`
    for more details.

    Parameters:
    - folders: list of paths to folders containing .xml annotations.
    - labels: a set of box labels to parse.

    Returns:
    - An list of annotations.
    """
    return Annotations([a for f in folders 
            for a in parse_xml_folder(f, recursive, labels)])


def _read_xml_object(obj, labels: Sequence[str] = None) -> BoundingBox:
    label = obj.find("name").text

    if labels and label not in labels:
        return None

    box = obj.find("bndbox")
    xmin = float(box.find("xmin").text)
    ymin = float(box.find("ymin").text)
    xmax = float(box.find("xmax").text)
    ymax = float(box.find("ymax").text)

    return BoundingBox(label, xmin, ymin, xmax, ymax)