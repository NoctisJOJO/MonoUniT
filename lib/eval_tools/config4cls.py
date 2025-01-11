#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
File: config.py
4cls
"""
import os
import time


current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
current_time = current_time.replace(" ", "-")

# dataset dir info
data_root = "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side"
image_root = "%s/image" % data_root
label_dir_9cls = "%s/label/camera" % data_root
label_dir = "./coarse4_label_path/label_2_filter"

intrinsic_dir = "%s/calib/camera_intrinsic" % data_root
denorm_dir = "%s/denorm" % data_root


transform_detection_dir = ""

current_dir = os.path.dirname(os.path.abspath(__file__))
debug_flag = True
debug_dir = os.path.join(current_dir, "debug")
test_dir = "./test"
if not os.path.exists(debug_dir):
    os.mkdir(debug_dir)
    print(debug_dir)
