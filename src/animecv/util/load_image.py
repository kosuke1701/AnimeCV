import glob
import os

from PIL import Image

IMAGE_EXTENSIONS = ["jpg", "png"]
def get_all_image_filenames_from_directory(self, directory):
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