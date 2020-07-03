import torch

class ImageEncoder(object):
    def __init__(self, torch_model, transform=None, batch_size=100):
        self.model = torch_model
        self.transform = transform
        self.bs = batch_size

    def _get_model(self):
        return self.model, next(self.model.parameters()).device
    
    def _tensorize_images(self, images):
        if self.transform is not None:
            imgs = [self.transform(img) for img in images]
        else:
            imgs = [torch.FloatTensor(img) for img in images]
        imgs = torch.cat([img.unsqueeze(0) for img in imgs])
        return imgs
    
    def _get_embedding(self, image_tensors):
        model, device = self._get_model()
        model.eval()

        image_tensors = image_tensors.to(device)

        with torch.no_grad():
            embs = []
            n_img = image_tensors.size(0)
            for i_start in range(0, n_img, self.bs):
                i_end = i_start + self.bs
                emb = model(
                    image_tensors[i_start:i_end]
                )
                embs.append(emb)
            if len(embs) > 1:
                embs = torch.cat(embs, dim=0)
            else:
                embs = embs[0]

        return embs

    def encode(self, images):
        x = self._tensorize_images(images)

        return self._get_embedding(x)
    
    def to(self, device):
        self.model.to(device)
        
        