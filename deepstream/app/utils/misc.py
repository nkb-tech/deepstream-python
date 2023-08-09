import ctypes
import sys
import io
import cv2
import base64
import numpy as np
from typing import List

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')


def long_to_int(long):
    value = ctypes.c_int(long & 0xffffffff).value
    return value

def long_to_uint64(long):
    value = ctypes.c_uint64(long & 0xffffffffffffffff).value
    return value

def get_label_names_from_file(filepath: str) -> List[str]:
    """ Read a label file and convert it to string list """
    f = io.open(filepath, "r")
    labels = f.readlines()
    labels = [elm[:-1] for elm in labels]
    f.close()
    return labels

def img2base64(image: np.ndarray) -> str:
    ret_val, jpg_img = cv2.imencode(".jpg", image)
    b64_string = base64.b64encode(jpg_img).decode("utf-8")
    return b64_string
