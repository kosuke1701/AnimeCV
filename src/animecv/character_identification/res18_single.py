import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

from .identifier import ImageCharacterIdentifier
from ..module import ImageEncoder, Similarity
from ..util import download

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        return x / torch.norm(x, dim=1, keepdim=True)

class Res18Normalized(nn.Module):
    def __init__(self, dropout, emb_dim):
        super().__init__()

        self.trunk = models.resnet18(pretrained=True)
        trunk_output_size = self.trunk.fc.in_features
        self.trunk.fc = Identity()

        self.embedder = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0.0 else Identity(),
            nn.Linear(trunk_output_size, emb_dim),
            Normalize()
        )

        self.model = nn.Sequential(self.trunk, self.embedder)
    
    def forward(self, x):
        return self.model(x)

class Res18_CharacterIdentifier(ImageCharacterIdentifier):
    def __init__(self, model_fn=None, base_dir=None):
        if model_fn is None:
            model_fn = download(
                "https://github.com/kosuke1701/character-reid/releases/download/0.0/res18_0222.mdl",
                "res18_0222.mdl"
            )

        saved_models = torch.load(model_fn, map_location="cpu")

        model = Res18Normalized(0.0, 500)
        model.trunk.load_state_dict(saved_models["trunk"])
        model.embedder.load_state_dict(saved_models["embedder"])

        transform = [
            transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        transform = transforms.Compose(transform)

        encoder = ImageEncoder(model, transform)

        similarity = Similarity(sim_func="L2")

        super().__init__(encoder, similarity)
        
