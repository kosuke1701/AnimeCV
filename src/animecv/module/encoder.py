import torch
from torchvision.transforms.functional import crop
from tqdm import tqdm

class ImageEncoder(object):
    def __init__(self, torch_model, transform=None, batch_size=100,
        verbose=False):
        self.model = torch_model
        self.transform = transform
        self.bs = batch_size

        self.verbose = verbose

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
        n_img = image_tensors.size(0)

        it = list(range(0, n_img, self.bs))
        if self.verbose:
            it = tqdm(it)
        with torch.no_grad():
            embs = []
            n_img = image_tensors.size(0)
            for i_start in it:
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

class ImageBBEncoder(ImageEncoder):
    def __init__(self, torch_model, post_trans=None, scale=1.0, 
        batch_size=100):
        super().__init__(torch_model, None, batch_size)

        self.post_trans = post_trans
        self.scale = scale

    def _rescale_bbox(self, xmin, ymin, xmax, ymax):
        xcenter = (xmin + xmax) / 2
        xwid = xmax - xmin
        ycenter = (ymin + ymax) / 2
        ywid = ymax - ymin

        new_bbox = [
            xcenter - xwid/2*self.scale,
            ycenter - ywid/2*self.scale,
            xcenter + xwid/2*self.scale,
            ycenter + ywid/2*self.scale
        ]
        return list(map(int, new_bbox))
    
    def _crop_bounding_box(self, images, bboxs):
        crop_imgs = []
        lst_i_img = []
        lst_i_bbox = []
        for i_img, img_bboxs in enumerate(bboxs):
            image = images[i_img]
            for i_bbox, bbox in enumerate(img_bboxs):
                coord = bbox["coordinates"]
                xmin, ymin, xmax, ymax = self._rescale_bbox(*coord)

                crop_img = crop(image, ymin, xmin, ymax-ymin, xmax-xmin)
                if self.post_trans is not None:
                    crop_img = self.post_trans(crop_img)

                crop_imgs.append(crop_img)
                lst_i_img.append(i_img)
                lst_i_bbox.append(i_bbox)
        
        crop_imgs = [img.unsqueeze(0) for img in crop_imgs]
        crop_imgs = torch.cat(crop_imgs, dim=0)

        return crop_imgs, lst_i_img, lst_i_bbox
    
    def encode(self, images, bboxs):
        img_tensors, lst_i_img, lst_i_bbox = \
            self._crop_bounding_box(images, bboxs)
        
        embs = self._get_embedding(img_tensors)

        return embs, lst_i_img, lst_i_bbox