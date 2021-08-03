from collections import defaultdict
from typing import Hashable, TypeVar, Sequence, Callable, Iterator
from pathlib import Path

from PIL import Image


def get_image_size(image: str) -> "tuple[int, int]":
    return Image.open(image).size


T = TypeVar("T")
S = TypeVar("S", bound=Hashable)
def dict_grouping(iterable: Sequence[T], by_key: Callable[[T], S]) -> "defaultdict[S, list[T]]":
    ret = defaultdict(list)
    for item in iterable:
        ret[by_key(item)].append(item)
    ret.default_factory = None
    return ret


def glob(folder: Path, extension: str, recursive: bool = False) -> Iterator[Path]:
    """Glob files with the specified extension in a folder."""

    assert extension.startswith("."), "Parameter 'extension' should start with a '.'."

    extension = extension.lower()
    files = folder.glob(f"**/*") if recursive else folder.glob(f"*")

    return (f for f in files if f.suffix.lower() == extension and not f.name.startswith("."))
    