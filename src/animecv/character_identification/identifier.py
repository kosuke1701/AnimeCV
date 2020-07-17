import numpy as np
import torch

def id_enroll(scores, mode):
    """
    Args:
        scores: A list of PyTorch FloatTensor. N-th tensor contains matching scores
            between images of an N-th character and target images
            (size: num_character_image x num_target_image).
        mode: Either "Max" or "Avg".
    Returns:
        FloatTensor: An numpy.ndarray of size (M x N) which contains identification scores between
            M target images and N known characters.
    """
    if mode == "Max":
        scores = [torch.max(_, dim=0, keepdim=True)[0] for _ in scores]
    elif mode == "Avg":
        scores = [torch.mean(_, dim=0, keepdim=True) for _ in scores]
    else:
        raise Exception(f"Illegal model: {mode}")
    
    scores = torch.t(torch.cat(scores, dim=0))

    return scores

class ImageCharacterIdentifier(object):
    def __init__(self, encoder, similarity):
        self.encoder = encoder
        self.similarity = similarity
    
    def encode_image(self, images):
        return self.encoder.encode(images)
    
    def identify(self, target_embeddings, character_embeddings, mode):
        """
        Args:
            target_embeddings (FloatTensor): 2D tensor of embeddings of target images.
            character_embeddings (list): List of 2D tensors. Each tensor is embeddings of character images.
        Returns:
            FloatTensor: 2D tensor of size M x N, 
                where M is number of target images and
                N is number of characters. 
        """
        scores = []

        for i_char, char_embs in enumerate(character_embeddings):
            scores.append(self.similarity.compute_similarity(char_embs, target_embeddings, "pair"))
        
        scores = id_enroll(scores, mode)

        return scores
    
    def to(self, device):
        self.encoder.to(device)

class ImageCharacterIdentifierBBox(ImageCharacterIdentifier):
    def __init__(self, encoder, similarity):
        super().__init__(encoder, similarity)

    def encode_image(self, images, bboxs):
        embs, lst_i_img, lst_i_bbox = self.encoder.encode(images, bboxs)
        return embs, lst_i_img, lst_i_bbox
    
    def identify_bbox(self, target_embeddings, target_bboxs,
        lst_i_img, lst_i_bbox, character_embeddings, mode):
        scores = self.identify(target_embeddings, character_embeddings, mode)

        for i_target, i_img, i_bbox in \
            zip(range(target_embeddings.size(0)), lst_i_img, lst_i_bbox):
            target_bboxs[i_img][i_bbox]["identification_score"] = scores[i_target]
