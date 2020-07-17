import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.transforms.functional import resize, pad

from .identifier import ImageCharacterIdentifierBBox
from ..module import ImageBBEncoder, Similarity
from ..util import download

from .res18_single import Identity, Normalize, Res18Normalized

class PadResize(object):
    def __init__(self, size, fill=0, padding_mode="constant", interpolation=2):
        self.fill = fill
        self.padding_mode = padding_mode

        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img):
        width, height = img.size
        max_size = max(width, height)

        left, right, top, bottom = 0, 0, 0, 0
        if width > height:
            top = bottom = (width - height) // 2
            if (width - height) % 2 == 1:
                top += 1
        elif height > width:
            left = right = (height - width) // 2
            if (height - width) % 2 == 1:
                left += 1
        
        img = pad(img, (left, top, right, bottom), fill=self.fill,
            padding_mode=self.padding_mode)
        img = resize(img, size=self.size, interpolation=self.interpolation)

        return img

class Res18_CharacterIdentifier_BBox(ImageCharacterIdentifierBBox):
    def __init__(self, base_dir=None):
        model_fn = download(
            "https://github.com/kosuke1701/character-reid/releases/download/0.0/0716_bbox.mdl",
            "0716_bbox.mdl"
        )
        saved_models = torch.load(model_fn, map_location="cpu")
        state_dict = {key.replace("trunk", "model.0"): val
            for key, val in saved_models["trunk"].items()}
        state_dict.update(saved_models["trunk"])
        state_dict["embedder.1.weight"] = saved_models["embedder"]["module.0.weight"]
        state_dict["embedder.1.bias"] = saved_models["embedder"]["module.0.bias"]
        state_dict["model.1.1.weight"] = saved_models["embedder"]["module.0.weight"]
        state_dict["model.1.1.bias"] = saved_models["embedder"]["module.0.bias"]
        
        model = Res18Normalized(0.0, 500)
        model.load_state_dict(state_dict)

        transform = [
            PadResize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        transform = transforms.Compose(transform)

        encoder = ImageBBEncoder(model, post_trans=transform,
            scale=1.5)
    
        similarity = Similarity(sim_func="L2")

        super().__init__(encoder, similarity)

