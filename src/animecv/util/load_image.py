import glob
import os

import cv2
from PIL import Image

from .display import from_cv_to_PIL

IMAGE_EXTENSIONS = ["jpg", "png"]
def get_all_image_filenames_from_directory(directory):
    all_dir_files = glob.glob(os.path.join(directory, "**", "*")) + \
        glob.glob(os.path.join(directory, "*"))
    image_files = [fn for fn in all_dir_files 
        if os.path.isfile(fn) and fn.split(".")[-1].lower() in IMAGE_EXTENSIONS]

    return image_files

def load_image(filename):
    try:
        with open(filename, "rb") as f:
            image = Image.open(f)
            return image.convert("RGB")
    except UserWarning as e:
        print(filename)
        input("Something wrong happens while loading image: {} {}".format(filename, str(e)))

def load_video(filename):
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        raise Exception(f"Cannot open video: {filename}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(from_cv_to_PIL(frame))
        else:
            break
    
    cap.release()
    return frames, (width, height, frame_rate)
