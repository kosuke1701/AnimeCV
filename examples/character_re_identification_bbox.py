from argparse import ArgumentParser
import glob
import os

import numpy as np

from animecv.util import load_image, from_PIL_to_cv, \
    add_bounding_box, write_image
from animecv.character_identification import Res18_CharacterIdentifier_BBox
from animecv.object_detection import FaceDetector_EfficientDet

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument("--target", help="Filename of target picture.")
    parser.add_argument("--reference", nargs="+", 
        help="List of directories which contains reference pictures. One directory per character.")
    parser.add_argument("--name", nargs="+",
        help="List of character names. Character order must matches to that of --reference.")
    parser.add_argument("--cuda", action="store_true",
        help="Use GPU or not.")
    parser.add_argument("--save-name",
        help="Filename to save target picture with bounding boxes.")

    args = parser.parse_args()

    # Load target picture.
    target_img = load_image(args.target)

    # Load all reference pictures.
    reference_img = [
        [load_image(fn) for fn in glob.glob(os.path.join(directory, "*"))]
        for directory in args.reference
    ]

    # Prepare pre-trained model.
    # Parameter file will be saved under ~/.animecv by default.
    identifier = Res18_CharacterIdentifier_BBox()
    detector = FaceDetector_EfficientDet(coef=0, use_cuda=args.cuda)

    if args.cuda:
        identifier.to("cuda")
    
    # Encode all bounding boxes in pictures.
    target_bbox = detector.detect([target_img])
    target_emb, target_i_img, target_i_bbox = \
        identifier.encode_image([target_img], target_bbox)
    
    reference_embs = []
    for lst_images in reference_img:
        reference_bbox = detector.detect(lst_images)
        reference_emb, _, _ = \
            identifier.encode_image(lst_images, reference_bbox)
        reference_embs.append(reference_emb)
    
    # Compute similarity scores between target bounding box and characters.
    # Scores will be stored in dictionaries in `target_bbox`
    identifier.identify_bbox(
        target_emb,
        target_bbox,
        target_i_img,
        target_i_bbox,
        reference_embs,
        mode="Avg"
    )

    # Visualize bounding boxes.
    target_img = from_PIL_to_cv(target_img)
    target_bbox = target_bbox[0]
    for bbox_dict in target_bbox:
        scores = bbox_dict["identification_score"].detach().cpu().numpy()
        i_character = np.argmax(scores)
        bbox_dict["label"] = f"{args.name[i_character]} - {scores[i_character]:.2f}"
        add_bounding_box(target_img, bbox_dict)

    write_image(target_img, args.save_name)