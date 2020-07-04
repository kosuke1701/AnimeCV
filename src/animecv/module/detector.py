class BoundingBoxDetector(object):
    def detect(self, images, **kwargs):
        """
        Args:
            images (list): A list of PIL images.
        Returns:
            list: List of lists of bounding box 
                information dictionary.
                Dictionary contains at least `coordinates` key
                which corresponds to a tuple, (x_min, y_min, x_max, y_max).
                X-axis is horizontal, and (0,0) is the top-left corner.
                N-th list contains bounding boxes from
                N-th image.
        """
        raise NotImplementedError()