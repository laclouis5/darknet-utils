from .annotation import Annotation, Annotations
from .utils import *

from concurrent.futures import ThreadPoolExecutor
from tqdm.contrib.concurrent import thread_map
from sys import exit
from os import PathLike
from pathlib import Path
from random import Random
import shutil
import lxml.etree as ET


def create_yolo_trainval(
    annotations: Annotations,
    labels: "list[str]" = None,
    save_dir: PathLike = "yolo_trainval/", 
    prefix: PathLike = "data/", 
    train_ratio: float = 80/100,
    shuffle: bool = True,
    random_seed: int = 149_843_046_101,
    exist_ok: bool = False,
):
    """
    Create a YOLO database suitable for training with Darknet
    from a set of BoundingBoxes representing image annotations.

    You should first parse BoundingBoxes from directories where
    images and annotations are stored.

    Randomness is reproductible if the annotation image paths
    comparison order and number stay the same across runs.

    Parameters:
    - annotations: the annotations for the database creation.
    - labels: list of labels specifying the label order in `obj.names`.
    - save_dir: the path where to store the YOLO database.
    - prefix: optional path prefix to insert before the image.
    paths stored in train.txt and val.txt.
    - train_ratio: the percent of images to use in the training set.
    - shuffle: set to True to shuffle the dataset.
    """
    assert 0.0 <= train_ratio <= 1.0, "train_ratio must be in 0...1"

    save_dir = Path(save_dir).expanduser().resolve()
    prefix = Path(prefix)
    train_dir = save_dir / "train/"
    valid_dir = save_dir / "val/"

    save_dir.mkdir(exist_ok=exist_ok)
    train_dir.mkdir(exist_ok=exist_ok)
    valid_dir.mkdir(exist_ok=exist_ok)

    labels = labels or sorted(annotations.labels())
    labels_to_numbers = {l: str(n) for n, l in enumerate(labels)}

    annotations.map_labels(labels_to_numbers)

    if shuffle:
        annotations.annotations = sorted(annotations, key=lambda a: a.image_path)
        random_gen = Random(random_seed)
        random_gen.shuffle(annotations)

    len_train = int(train_ratio * len(annotations))

    def create_annotation(indexed_annotation: "tuple[int, Annotation]") -> str:
        i, annotation = indexed_annotation

        dir = train_dir if i < len_train else valid_dir
        img_filename = dir / f"im_{i:06}{annotation.image_path.suffix}"
        ann_filename = img_filename.with_suffix(".txt")
        ann_content = annotation.yolo_repr()

        try:
            shutil.copyfile(annotation.image_path, img_filename)
            ann_filename.write_text(ann_content)
        except KeyboardInterrupt:
            shutil.copyfile(annotation.image_path, img_filename)
            ann_filename.write_text(ann_content)
            exit()

        return img_filename.name

    image_names = thread_map(create_annotation, enumerate(annotations), 
        total=len(annotations), unit="imgs")

    train_file = save_dir / "train.txt"
    valid_file = save_dir / "val.txt"
    names_file = save_dir / "obj.names"

    train_file.write_text(
        "\n".join(str(prefix / f"train/{n}") for n in image_names[:len_train]))
    valid_file.write_text(
        "\n".join(str(prefix / f"val/{n}") for n in image_names[len_train:]))

    names_file.write_text("\n".join(labels))


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

        content = ET.tostring(tree, encoding="unicode", pretty_print=True)
        try: 
            path.write_text(content)
        except KeyboardInterrupt:
            path.write_text(content)
            exit()


def resolve_xml_file_paths(folders: "list[PathLike]", recursive: bool = False):
    """
    Change the 'path' field of xml file to be the current path
    of the xml file. this function should be used if the original 
    database has been moved and the `path` field no longer matches
    the file path.

    Parameters:
    - folders: paths to folders with .xml files to process
    """
    files = (f for folder in folders for f in folder.glob("**/*.xml" if recursive else "*.xml"))
    executor = ThreadPoolExecutor()
    executor.map(_resolve, files)


def _resolve(file: Path):
    filename = str(file)
    try:
        tree = ET.parse(filename)
        tree.find("path").text = filename
    except ET.ParseError: 
        return
    else:
        content = ET.tostring(tree, encoding="unicode", pretty_print=True)
        try:  # Avoid data corruption if KeyboardInterrupt while writing
            file.write_text(content)
        except KeyboardInterrupt:
            file.write_text(content)
            exit()