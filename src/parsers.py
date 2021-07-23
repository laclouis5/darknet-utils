from .bounding_box import BoundingBox
from .bounding_boxes import BoundingBoxes
from .utils import *

from os import PathLike
from pathlib import Path
import lxml.etree as ET
from functools import reduce


def _read_xml_object(obj) -> BoundingBox:
    label = obj.find('name').text
    box = obj.find('bndbox')
    xmin = float(box.find('xmin').text)
    ymin = float(box.find('ymin').text)
    xmax = float(box.find('xmax').text)
    ymax = float(box.find('ymax').text)

    return BoundingBox(label, xmin, ymin, xmax, ymax)


def parse_xml(file: PathLike) -> BoundingBoxes:
    """
    Parse an .xml file annotated with labelImg of other
    software that used the same anntotation format.

    Take a look at the source code to check if the parser
    is suitable for your annotation format. Bounding boxes
    annotations are in absolute coordinates and use the 
    top-left (xmin, ymin) and bottom-right (xmax, ymax) scheme.

    Parameters:
    - file: the xml file to process
    
    Returns:
    - A BoundingBoxes object representing the image annotations
    """
    tree = ET.parse(str(file)).getroot()
    path = tree.find("path").text
    name = tree.find("filename").text
    image = Path(path).with_name(name).expanduser().resolve()

    assert image.is_file(), f"Image '{image}' does not exists, the field 'path' of xml file '{file}' may be invalid"

    return BoundingBoxes({str(image): [_read_xml_object(o) for o in tree.findall('object')]})


def parse_xml_folder(folder: PathLike) -> BoundingBoxes:
    """
    Parse .xml annotations present in a folder. See `parse_xml`
    for more details.

    Parameters:
    - folder: a path to a folder containing .xml annotations

    Returns:
    - A BoundingBoxes object representing the annotations
    """
    folder = Path(folder).expanduser().resolve()
    files = folder.glob("*.xml")
    return reduce(BoundingBoxes.__ior__, (parse_xml(f) for f in files), BoundingBoxes())


def parse_xml_folders(folders: "list[PathLike]") -> BoundingBoxes:
    """
    Parse .xml annotations present in several folders. See `parse_xml`
    for more details.

    Parameters:
    - folders: list of paths to folders containing .xml annotations

    Returns:
    - A BoundingBoxes object representing the annotations
    """
    return reduce(BoundingBoxes.__ior__, (parse_xml_folder(f) for f in folders), BoundingBoxes())
