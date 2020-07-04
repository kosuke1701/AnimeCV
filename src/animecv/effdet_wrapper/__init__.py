import sys
import os

EFFDET_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "..",
    "effdet"
)

sys.path = [EFFDET_PATH, *sys.path]

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, \
    STANDARD_COLORS, standard_to_bgr, get_index_label, \
    plot_one_box

sys.path.remove(EFFDET_PATH)