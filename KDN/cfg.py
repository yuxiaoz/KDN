import os
import numpy as np
from easydict import EasyDict as edict

configs = edict()
cfg = configs

configs.domain_name = ["GTA5", "SYNTHIA", "Cityscapes", "BDDS"]
configs.GTA5_IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32) #BGR
configs.SYNTHIA_IMG_MEAN = np.array((63.31370899, 70.81393094, 80.35045896), dtype=np.float32) #BGR
configs.Cityscapes_IMG_MEAN = np.array((72.392398761941593, 82.908917542625858, 73.158359210711552), dtype=np.float32) #BGR
configs.image_width = 640
configs.image_height = 360
configs.output_stride = 8

configs.restore_lambda = 0.001
