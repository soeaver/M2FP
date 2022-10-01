import numpy as np
from PIL import Image

from detectron2.utils.file_io import PathManager

def load_image_into_numpy_array(filename, copy=False, dtype=np.int):
    with PathManager.open(filename, "rb") as f:
        array = np.array(Image.open(f), copy=copy, dtype=dtype)
        if len(array.shape) == 3:
            assert array.shape[2] == 3
            array = array.transpose(2, 0, 1)[0, :, :]
    return array
