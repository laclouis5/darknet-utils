class BoundingBox:
    """
    A bounding box with a label and an optional confidence.
    The coordinates are absolute (in pixels) and specified as the top-left
    point (xmin, ymin) and the bottom-right one (xmax, ymax).
    """

    __slots__ = ("label", "_xmin", "_ymin", "_xmax", "_ymax", "confidence")

    def __init__(self, 
        label: str, 
        xmin: float, 
        ymin: float, 
        xmax: float, 
        ymax: float,
        confidence: float = None
    ):
        if confidence: 
            assert 0.0 <= confidence <= 1.0, f"Confidence ({confidence}) should be in 0...1"

        self.label = label
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax
        self.confidence = confidence

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

    @xmid.setter
    def xmid(self, value):
        delta = self.xmid - value
        self._xmin += delta
        self._xmax += delta
        
    @ymid.setter
    def ymid(self, value):
        delta = self.ymid - value
        self._ymin += delta
        self._ymax += delta

    @xmin.setter
    def xmin(self, value):
        self._xmin = value

    @ymin.setter
    def ymin(self, value):
        self._ymin = value

    @xmax.setter
    def xmax(self, value):
        self._xmax = value

    @ymax.setter
    def ymax(self, value):
        self._ymax = value

    @staticmethod
    def from_xywh(
        label: str, 
        xmid: float, 
        ymid: float, 
        width: float, 
        height: float, 
        confidence: float = None
    ) -> "BoundingBox":
        """
        Intantiates a BoundingBox from middle point (xmid, ymid)
        and box size (width, height).
        """
        mid_w = width / 2
        mid_h = height / 2

        xmin = xmid - mid_w
        ymin = ymid - mid_h
        xmax = xmid + mid_w
        ymax = ymid + mid_h

        return BoundingBox(label, xmin, ymin, xmax, ymax, confidence)

    def yolo_coords(self, 
        img_size: "tuple[int, int]"
    ) -> "tuple[float, float, float, float]":
        """
        Convert the box coordinates to YOLO coordinate format
        which is relative (xmid, ymid, width, height).

        Parameters:
         - img_size: the image width and height in pixels

        Returns:
         - A 4-element tuple of relative coordinates (xmid, ymid, width, height)
        """
        img_w, img_h = img_size
        return self.xmid / img_w, self.ymid / img_h, self.width / img_w, self.height / img_h

    def yolo_repr(self, 
        img_size: "tuple[int, int]", 
        include_confidence=True
    ) -> str:
        """
        YOLO string representation of a box annotation, which is relative coordinates
        separated by a whitespace:

        `<label> <optional: confidence> <xmid> <ymid> <width> <height>`

        Parameters:
         - img_size: the image width and height in pixels
         - include_confidence: if True, bounding box confidence score is included if present

        Returns:
         - The string representation
        """
        coords = self.yolo_coords(img_size)
        if include_confidence and self.confidence is not None:
            return " ".join((self.label, self.confidence, *(f"{c}" for c in coords)))      
        return " ".join((self.label, *(f"{c}" for c in coords)))