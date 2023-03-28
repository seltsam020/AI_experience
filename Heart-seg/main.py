# coding=utf-8
from seg_detect import *
from classify_detect import *

try:
    from PIL import Image
except ImportError:
    import Image

input_path=input("请输入图片位置：")
val(input_path)
classify(input_path)