from collections import defaultdict
from typing import Hashable, TypeVar, Sequence, Callable
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
