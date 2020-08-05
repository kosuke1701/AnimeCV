from argparse import ArgumentParser
import glob
import os

import numpy as np

from animecv.util import load_image, from_PIL_to_cv, \
    add_bounding_box, write_image, load_video, write_mp4_video, \
    get_all_image_filenames_from_directory
    
from animecv.character_identification import Res18_CharacterIdentifier_BBox
from animecv.object_detection import FaceDetector_EfficientDet

COLOR_LIST = [
    "navy", "blue", "aqua", "teal", "olive", "green",
    "lime", "yellow", "orange", "red", "maroon", "fuchsia",
    "purple", "black", "gray", "silver"
]

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument("--target", help="Filename of target video.")
    parser.add_argument("--reference-dir", 
        help="Directory whoes subdirectories contain reference images for each character.")
    parser.add_argument("--save-name", help="Filename of output video.")
    parser.add_argument("--cuda", action="store_true",
        help="Use GPU or not.")
    
    args = parser.parse_args()

    # Prepare pre-trained model.
    # Parameter file will be saved under ~/.animecv by default.
    identifier = Res18_CharacterIdentifier_BBox()
    identifier.encoder.verbose = True
    detector = FaceDetector_EfficientDet(coef=0, use_cuda=args.cuda)
    detector.verbose = True
    if args.cuda:
        identifier.to("cuda")

    # Prepare character information
    target_directories = glob.glob(os.path.join(args.reference_dir, "*"))
    characters = [os.path.basename(dirname) for dirname in target_directories]
    colors = [COLOR_LIST[np.random.randint(len(COLOR_LIST))] for _ in characters]
    
    # Encode reference images of characters
    print("Encoding reference images.")
    reference_embs = []
    for dirname in target_directories:
        print(dirname)
        image_fns = get_all_image_filenames_from_directory(dirname)
        images = [load_image(fn) for fn in image_fns]

        bboxes = detector.detect(images)
        emb, _, _ = \
            identifier.encode_image(images, bboxes)
        reference_embs.append(emb)

    # Detect character for each frame of input Video.
    print("Detecting reference images.")
    target_frames, (width, height, fps) = load_video("Target.mp4")
    
    target_bboxes = detector.detect(target_frames)
    target_emb, target_i_img, target_i_bbox = \
        identifier.encode_image(target_frames, target_bboxes)
    
    identifier.identify_bbox(
        target_emb,
        target_bboxes,
        target_i_img,
        target_i_bbox,
        reference_embs,
        mode="Max"
    )

    # Write bounding boxes to each frames.
    target_frames = [from_PIL_to_cv(frame) for frame in target_frames]
    for i_frame, bbox_frame in enumerate(target_bboxes):
        for bbox_dict in bbox_frame:
            scores = bbox_dict["identification_score"].detach().cpu().numpy()
            i_character = np.argmax(scores)
            bbox_dict["label"] = f"{characters[i_character]} : {scores[i_character]:.2f}"
            bbox_dict["color"] = colors[i_character]
            add_bounding_box(target_frames[i_frame], bbox_dict)
    
    write_mp4_video(target_frames, args.save_name, fps, (width, height))

    


    # target_imgs, (width, height, fps) = load_video("Target.mp4")
    # print(len(target_imgs))
    # for i_frame, frame in enumerate(target_imgs[::100]):
    #     frame = from_PIL_to_cv(frame)
    #     write_image(frame, f"target_{i_frame}.png")
    
    # target_imgs = [from_PIL_to_cv(frame) for frame in target_imgs]
    # write_mp4_video(target_imgs[:50], "test.mp4", fps, (width, height))
