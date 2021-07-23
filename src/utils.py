from collections import defaultdict
from PIL import Image


def image_size(image: str):
    return Image.open(image).size


def group(it, by_key):
    d = defaultdict(list)
    for e in it: d[by_key(e)].append(e)
    return d