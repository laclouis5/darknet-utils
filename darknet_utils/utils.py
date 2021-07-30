from collections import defaultdict
from typing import Hashable, TypeVar, Sequence, Callable
from PIL import Image


def get_image_size(image: str) -> "tuple[int, int]":
    return Image.open(image).size


T = TypeVar("T")
S = TypeVar("S", bound=Hashable)
def dict_grouping(it: Sequence[T], by_key: Callable[[T], S]) -> "defaultdict[S, list[T]]":
    d = defaultdict(list)
    for e in it: 
        d[by_key(e)].append(e)
    return d