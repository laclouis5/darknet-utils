import logging
from .bounding_box import BoundingBox
from .annotation import Annotation, Annotations

from os import PathLike
from pathlib import Path
import logging

import lxml.etree as ET


logging.basicConfig(filename="parser.log")

def parse_xml_file(file: PathLike) -> Annotation:
    """
    Parse an .xml file annotated with labelImg of other
    software that used the same anntotation format.

    Take a look at the source code to check if the parser
    is suitable for your annotation format. Bounding boxes
    annotations are in absolute coordinates and use the 
    top-left (xmin, ymin) and bottom-right (xmax, ymax) scheme.

    Parameters:
    - file: the xml file to process.
    
    Returns:
    - An object representing the image annotations or None if 
    the .xml file was not readable.
    """

    try:
        tree = ET.parse(str(file)).getroot()
        
        path = tree.find("path").text
        name = tree.find("filename").text

        img_size_et = tree.find("size")
        img_w = int(img_size_et.find("width").text)
        img_h = int(img_size_et.find("height").text)

        boxes = [_read_xml_object(o) for o in tree.findall("object")]
    except:
        logging.warning(f"Error while reading '{file}'.")
        return None

    image_path = Path(path).with_name(name).expanduser().resolve()

    return Annotation(image_path, (img_w, img_h), boxes)


def parse_xml_folder(folder: PathLike, recursive=False) -> Annotations:
    """
    Parse .xml annotations present in a folder. See `parse_xml`
    for more details.

    Parameters:
    - folder: a path to a folder containing .xml annotations.

    Returns:
    - A list of annotations.
    """
    folder = Path(folder).expanduser().resolve()
    files = folder.glob("**/*.xml" if recursive else "*.xml")
    annotations = (parse_xml_file(f) for f in files)
    return Annotations([a for a in annotations if a])


def parse_xml_folders(folders: "list[PathLike]", recursive=False) -> Annotations:
    """
    Parse .xml annotations present in several folders. See `parse_xml`
    for more details.

    Parameters:
    - folders: list of paths to folders containing .xml annotations.

    Returns:
    - An list of annotations.
    """
    return Annotations([a for f in folders for a in parse_xml_folder(f, recursive)])


def _read_xml_object(obj) -> BoundingBox:
    label = obj.find('name').text
    box = obj.find('bndbox')
    xmin = float(box.find('xmin').text)
    ymin = float(box.find('ymin').text)
    xmax = float(box.find('xmax').text)
    ymax = float(box.find('ymax').text)

    return BoundingBox(label, xmin, ymin, xmax, ymax)