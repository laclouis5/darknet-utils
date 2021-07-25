from .annotation import Annotation
from .utils import *

from os import PathLike
from pathlib import Path
from random import Random
import shutil
from copy import copy

import lxml.etree as ET
from tqdm.contrib import tenumerate


def create_yolo_trainval(
    annotations: "list[Annotation]", 
    save_dir: PathLike = "yolo_trainval/", 
    prefix: PathLike = "data/", 
    train_ratio: float = 80/100,
    shuffle: bool = True,
    exist_ok: bool = False
):
    """
    Create a YOLO database suitable for training with Darknet
    from a set of BoundingBoxes representing image annotations.

    You should first parse BoundingBoxes from directories where
    images and annotations are stored.

    Parameters:
    - annotations: the annotations for the database creation.
    - save_dir: the path where to store the YOLO database
    - prefix: optional path prefix to insert before the image
    paths stored in train.txt and val.txt.
    - train_ratio: the percent of images to use in the training set.
    """
    assert 0.0 <= train_ratio <= 1.0, "train_ratio must be in 0...1"

    save_dir = Path(save_dir).expanduser().resolve()
    prefix = Path(prefix)
    train_dir = save_dir / "train/"
    valid_dir = save_dir / "val/"

    save_dir.mkdir(exist_ok=exist_ok)
    train_dir.mkdir(exist_ok=exist_ok)
    valid_dir.mkdir(exist_ok=exist_ok)

    if shuffle:
        random_gen = Random(149_843_046_101)
        annotations = copy(annotations)
        random_gen.shuffle(annotations)

    len_train = int(train_ratio * len(annotations))

    for i, annotation in tenumerate(annotations, unit="imgs"):
        dir = train_dir if i < len_train else valid_dir
        img_filename = dir / f"im_{i:06}{annotation.image_path.suffix}"
        ann_filename = img_filename.with_suffix(".txt")

        shutil.copy(annotation.image_path, img_filename)
        ann_filename.write_text(annotation.yolo_repr())

    train_file = save_dir / "train.txt"
    valid_file = save_dir / "val.txt"

    train_file.write_text(
        "\n".join(str(train_dir / prefix / f"train/im_{i:06}{annotation.image_path.suffix}")
            for i in range(len_train)))
    valid_file.write_text(
        "\n".join(str(valid_dir / prefix / f"val/im_{i:06}{annotation.image_path.suffix}" )
            for i in range(len_train, len(annotations))))


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
        img_w, img_h = get_image_size(image)

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