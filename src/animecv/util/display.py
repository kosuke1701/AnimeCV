from bounding_box import bounding_box as bb
import cv2
import numpy as np
from PIL import Image

def from_PIL_to_cv(image):
    image = np.array(image, dtype=np.uint8)
    if image.ndim == 2:
        image = np.concatenate([image[:,:,np.newaxis]]*3, axis=2)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.shape[3] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image

def from_cv_to_PIL(image):
    image = image.copy()
    if image.shape[2] == 3:
        image = image[:,:,::-1]
    elif image.shape[2] == 4:
        image = image[:,:,[2,1,0,3]]
    return Image.fromarray(image)


def add_bounding_box(image, bb_info):
    xmin, ymin, xmax, ymax = bb_info["coordinates"]
    label = bb_info["label"] if "label" in bb_info else ""
    color = bb_info["color"] if "color" in bb_info else "green"

    bb.add(image, xmin, ymin, xmax, ymax, label, color)

def write_image(image, path):
    if isinstance(image, np.ndarray):
        cv2.imwrite(path, image)

def write_mp4_video(images, filename, frame_rate=24.0, size=(640, 480)):
    fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(filename, fmt, frame_rate, size)
    for frame in images:
        writer.write(frame)
    writer.release()