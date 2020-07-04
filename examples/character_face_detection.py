from argparse import ArgumentParser
import sys

from animecv.util import load_image, from_PIL_to_cv, \
    add_bounding_box, write_image
from animecv.object_detection import FaceDetector_EfficientDet

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument("--target", help="Filename of target picture.")
    parser.add_argument("--save-name",
        help="Filename to save target picture with bounding boxes.")
    parser.add_argument("--cuda", action="store_true",
        help="Use GPU or not.")

    args = parser.parse_args()

    # Load target picture.
    target_img = load_image(args.target)

    # Load face detector.
    # Parameter file will be saved under ~/.animecv by default.
    detector = FaceDetector_EfficientDet(coef=0, use_cuda=args.cuda)

    # Get bounding boxes of faces.
    # Note that argument is a list of PIL images.
    bbox = detector.detect([target_img])

    # Output is a list of bounding boxes for each target image.
    target_bbox = bbox[0]
    if len(target_bbox) == 0:
        print("No face was detected in given picture.")
        sys.exit(0)
    
    # Convert PIL image to OpenCV format for display purpose.
    target_img = from_PIL_to_cv(target_img)
    
    for bbox_dict in target_bbox:
        print(bbox_dict["coordinates"], bbox_dict["score"])

        # You can draw bounding box to OpenCV image.
        # Label will be printed with bounding box.
        bbox_dict["label"] = f"{bbox_dict['score']:.2f}"
        add_bounding_box(target_img, bbox_dict)
    
    # Write OpenCV image to file.
    write_image(target_img, args.save_name)


    