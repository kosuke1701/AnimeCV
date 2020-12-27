import json
import os

import torch
import torch.nn as nn
from torchvision import transforms, models

from ..module import Similarity, ImageEncoder

MODEL_CLASSES = {
    "ResNet-18": models.resnet18,
    "ResNet-34": models.resnet34,
    "ResNet-50": models.resnet50,
    "ResNet-101": models.resnet101,
    "ResNet-152": models.resnet152
}

def load_OML_ImageFolder_models(model_dir):
    embedder_statedict = torch.load(
        os.path.join(model_dir, "embedder.pth"), map_location="cpu"
    )
    trunk_statedict = torch.load(
        os.path.join(model_dir, "trunk.pth"), map_location="cpu"
    )
    with open(os.path.join(model_dir, "conf.json")) as h:
        CONF, PARAMS = json.load(h)
    
    assert CONF["trunk_model"] in MODEL_CLASSES
    trunk = MODEL_CLASSES[CONF["trunk_model"]](pretrained=True)
    trunk_output_size = trunk.fc.in_features
    trunk.fc = nn.Identity()

    embedder = nn.Sequential(
        nn.Linear(trunk_output_size, CONF["dim"]),
        nn.Dropout(PARAMS["p_dropout"])
    )

    trunk.load_state_dict(trunk_statedict)
    embedder.load_state_dict(embedder_statedict)
    
    return trunk, embedder

class OML_ImageFolder_Pretrained(nn.Module):
    def __init__(self, model_dir):
        super().__init__()

        trunk, embedder = load_OML_ImageFolder_models(model_dir)        

        self.trunk = trunk
        self.embedder = embedder
    
    def forward(self, x):
        x = self.trunk(x)
        x = self.embedder(x)
        x = x / torch.norm(x, dim=1, keepdim=True)
        return x

def create_OML_ImageFolder_Encoder(model_dir):
    return ImageEncoder(
        OML_ImageFolder_Pretrained(model_dir),
        Similarity(sim_func="L2")
    )