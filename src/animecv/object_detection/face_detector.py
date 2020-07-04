from .efficientdet_detector import YetAnotherEfficientDetDetector
from ..util import download

class FaceDetector_EfficientDet(YetAnotherEfficientDetDetector):
    def __init__(self, coef, use_cuda=True, base_dir=None):
        anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

        if coef == 0:
            model_fn = download(
                "https://github.com/kosuke1701/pretrained-efficientdet-character-head/releases/download/0.0/character-face-efficientdet-c0.pth",
                "character-face-efficientdet-c0.pth",
                base_dir=base_dir
            )
        elif coef == 2:
            model_fn = download(
                "https://github.com/kosuke1701/pretrained-efficientdet-character-head/releases/download/0.0/character-face-efficientdet-c2.pth",
                "character-face-efficientdet-c2.pth",
                base_dir=base_dir
            )
        
        self.class_lst = ["face"]

        super().__init__(
            anchor_ratios, anchor_scales, coef,
            len(self.class_lst), model_fn, use_cuda
        )
    
    def detect(self, pil_images, threshold=0.5, iou_threshold=0.5):
        return self._detect(
            pil_images, self.class_lst, threshold, iou_threshold
        )
