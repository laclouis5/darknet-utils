from .bounding_box import BoundingBox
from .bounding_boxes import BoundingBoxes
from .utils import image_size

from os import PathLike
from pathlib import Path
from random import Random
import shutil

import lxml.etree as ET
from tqdm.contrib import tenumerate


def _get_yolo_repr(
    box: BoundingBox,
    img_size: "tuple[int, int]",
    norm_ratio: float = None, 
    norm_names: "list[str]" = ["tige", "stem"],
) -> str:
    if norm_ratio and any(n in box.label for n in norm_names):
        return box.yolo_repr(img_size, norm_ratio)
    
    return box.yolo_repr(img_size)


def create_yolo_trainval(
    boxes: BoundingBoxes, 
    save_dir: PathLike = "yolo_trainval/", 
    prefix: PathLike = "data/", 
    train_ratio: float = 80/100,
    norm_ratio: float = None,
    norm_names: "list[str]" = ["tige", "stem"],
):
    """
    Create a YOLO database suitable for training with Darknet
    from a set of BoundingBoxes representing image annotations.

    You should first parse BoundingBoxes from directories where
    images and annotations are stored.

    Parameters:
    - boxes: the BoundingBoxes representing the annotations and
    images for the database creation
    - save_dir: the path where to store the YOLO database
    - prefix: optional path prefix to insert before the image
    paths stored in train.txt and val.txt
    - train_ratio: the percent of images to use in the training set
    - norm_ratio: if provided, BoundingBoxes with a label containing
    a name present in 'norm_names' will have a box normalized to a 
    square one with side length equal to norm_ratio * min(img_w, img_h)
    - norm_names: a list of string fragments that may be found in a
    BoundingBox label to be normalized
    """
    save_dir = Path(save_dir).expanduser().resolve()
    train_dir = save_dir / "train/"
    valid_dir = save_dir / "val/"

    save_dir.mkdir()
    train_dir.mkdir()
    valid_dir.mkdir()

    train_txt = save_dir / "train.txt"
    valid_txt = save_dir / "val.txt"

    train_list = []
    valid_list = []

    random_gen = Random(149_843_046_101)
    names = random_gen.sample(boxes.image_names, len(boxes))
    len_train = int(train_ratio * len(names))

    for i, name in tenumerate(names, unit="imgs"):
        dir = train_dir if i < len_train else valid_dir
        file_list = train_list if i < len_train else valid_list

        img_boxes = boxes[name]
        img_size = image_size(name)

        img_filename = dir / (f"im_{i:06}" + Path(name).suffix)
        ann_filename = img_filename.with_suffix(".txt")

        shutil.copy(name, str(img_filename))

        yolo_repr = "\n".join(_get_yolo_repr(b, img_size, norm_ratio, norm_names) for b in img_boxes)
        ann_filename.write_text(yolo_repr)

        dir_str = Path("train/" if i < len_train else "val/")
        file_list.append(Path(prefix) / dir_str / img_filename.name)

    train_txt.write_text("\n".join(str(p) for p in train_list))
    valid_txt.write_text("\n".join(str(p) for p in valid_list))


def create_noobj_folder(
    folder: PathLike, 
    img_ext: str = ".jpg",
):
    """
    Add empty .xml files for each image in a folder
    which does not contain annotation.

    Parameters:
    - folder: the path where images are stored
    - img_ext: the image extension to consider
    """
    folder = Path(folder).expanduser().resolve()
    images = folder.glob(f"*{img_ext}")
    
    for image in images:
        filename = image.name
        _folder = image.parent.name
        path = folder / (image.stem + ".xml")
        img_w, img_h = image_size(image)

        tree = ET.Element("annotation")

        et_folder = ET.SubElement(tree, "folder")
        et_folder.text = _folder

        et_filename = ET.SubElement(tree, "filename")
        et_filename.text = filename

        et_path = ET.SubElement(tree, "path")
        et_path.text = str(path)

        et_img_size = ET.SubElement(tree, "size")
        ET.SubElement(et_img_size, "width").text = str(img_w)
        ET.SubElement(et_img_size, "height").text = str(img_h)
        ET.SubElement(et_img_size, "depth").text = "3"

        path.write_text(ET.tostring(tree, encoding="unicode", pretty_print=True))


def resolve_xml_file_paths(folders: "list[PathLike]"):
    """
    Change the 'path' field of xml file to be the current path
    of the xml file. this function should be used if the original 
    database has been moved and the 'path' field no longer matches
    the correct path.

    Parameters:
    - folders: paths to folders with .xml files to process
    """
    for folder in folders:
        folder = Path(folder).expanduser().resolve()
        for file in folder.glob("*.xml"):
            tree = ET.parse(str(file))
            tree.find("path").text = str(file)
            file.write_text(ET.tostring(tree, encoding="unicode", pretty_print=True))