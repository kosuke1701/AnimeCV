import numpy as np
import torch

from .. import effdet_wrapper as det
from ..module import BoundingBoxDetector
from ..util import from_PIL_to_cv

class YetAnotherEfficientDetDetector(BoundingBoxDetector):
    def __init__(self, anchor_ratios, anchor_scales,
        compound_coef, num_classes, model_fn,
        use_cuda=True):
        if not torch.cuda.is_available():
            use_cuda = False
        self.use_cuda = use_cuda

        self.model = det.EfficientDetBackbone(
            compound_coef=compound_coef,
            num_classes=num_classes,
            ratios=anchor_ratios,
            scales=anchor_scales
        )
        state_dict = torch.load(model_fn, map_location="cpu")
        self.model.load_state_dict(state_dict)
        if use_cuda:
            self.model.to("cuda")
        self.model.eval()
        self.model.requires_grad_(False)

        self.input_size = \
            input_size = [512, 640, 768, 896, 1024, 1280, 1280, 1536][compound_coef]
    
    def _detect(self, pil_images, class_lst, threshold=0.5, iou_threshold=0.5):
        images = [from_PIL_to_cv(img) for img in pil_images]
        _, framed_imgs, framed_metas = \
            det.preprocess(images, max_size=self.input_size)

        x = torch.stack([torch.from_numpy(f_img) 
            for f_img in framed_imgs], dim=0)
        x = x.to(torch.float32).permute(0, 3, 1, 2)
        if self.use_cuda:
            x = x.to("cuda")
        
        with torch.no_grad():
            feat, reg, clsf, anchor = self.model(x)

            reg_box = det.BBoxTransform()
            clip_box = det.ClipBoxes()

            out = det.postprocess(x, anchor,reg, clsf, reg_box, 
                clip_box, threshold, iou_threshold)
        
        out = det.invert_affine(framed_metas, out)

        out = [
            [
                {
                    "coordinates": tuple(out_image["rois"][j].astype(np.int)),
                    "class": class_lst[out_image["class_ids"][j]],
                    "score": out_image["scores"][j]
                }
                for j in range(len(out_image["rois"]))
            ]
            for out_image in out 
        ]

        return out
