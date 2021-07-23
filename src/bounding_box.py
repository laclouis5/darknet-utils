class BoundingBox:
    """
    A bounding box with a label and an optional confidence.
    The coordinates are absolute (in pixels) and specified as the top-left
    point (xmin, ymin) and the bottom-right one (xmax, ymax).
    the implementation takes care or edge-cases such as xmin being greater
    than xmax.
    """

    def __init__(self, 
        label: str, 
        xmin: float, 
        ymin: float, 
        xmax: float, 
        ymax: float,
    ):
        self.label = label
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax

    @property
    def xmid(self) -> float:
        return (self._xmax + self._xmin) / 2

    @property
    def ymid(self) -> float:
        return (self._ymax + self._ymin) / 2

    @property
    def width(self) -> float:
        return abs(self._xmax - self._xmin)
    
    @property
    def height(self) -> float:
        return abs(self._ymax - self._ymin)

    @property
    def xmin(self) -> float:
        return min(self._xmin, self._xmax)

    @property
    def ymin(self) -> float:
        return min(self._ymin, self._ymax)

    @property
    def xmax(self) -> float:
        return max(self._xmin, self._xmax)

    @property
    def ymax(self) -> float:
        return max(self._ymin, self._ymax)

    def yolo_coords(self, 
        img_size, 
        norm_ratio: float = None
    ) -> "tuple[float, float, float, float]":
        """
        Convert the box coordinates to YOLO coordinate format
        which is relative (xmid, ymid, width, height). If norm_ratio is not
        None, the box width and height is normalized to the ratio.

        Parameters:
         - img_size: the image width and height in pixels
         - norm_ratio: the lenght of square boxes for stem annotations

        Returns:
         - A 4-element tuple of relative coordinates (xmid, ymid, width, height)
        """
        img_w, img_h = img_size

        if norm_ratio is not None:
            assert 0 <= norm_ratio <=1, "norm_ratio should be in 0...1"
            side = min(img_size) * norm_ratio
            return self.xmid / img_w, self.ymid / img_h, side / img_w, side / img_h

        return self.xmid / img_w, self.ymid / img_h, self.width / img_w, self.height / img_h

    def yolo_repr(self, img_size, norm_ratio: float = None) -> str:
        """
        YOLO string representation of a box annotation, which is relative coordinates
        separated by a whitespace:

         `<label> <xmid> <ymid> <width> <height>`

        Parameters:
         - img_size: the image width and height in pixels
         - norm_ratio: the lenght of square boxes for stem annotations

        Returns:
         - The string representation
        """
        coords = self.yolo_coords(img_size, norm_ratio)
        return " ".join((self.label, *(f"{c}" for c in coords)))