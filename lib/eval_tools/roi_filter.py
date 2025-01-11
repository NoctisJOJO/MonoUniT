#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" 
File: roi_filter.py
author: shumao, liyingying
"""
import json
import argparse
import pdb
import os
import sys
import cv2
from PIL import Image
import glob

from tqdm import tqdm

import config4cls as config

# parser = argparse.ArgumentParser(description="convert parameters")
# parser.add_argument("--txt_dir", required=True)
# parser.add_argument("--output_dir", required=True)
# args = parser.parse_args()

# txt_path = args.txt_dir
# output_dir = args.output_dir
# calib_dir = config.intrinsic_dir
BASE_DIR = os.path.join("/root/MonoUniT/dataset/V2X-Seq/infrastructure-side")
txt_path = os.path.join(BASE_DIR, "label/camera")
output_dir = os.path.join(BASE_DIR, "label_2")
calib_dir = os.path.join(BASE_DIR, "calib/camera_intrinsic")


def filter(txt_path, output_dir, calib_dir):
    """只留下路面上的物体"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("outputdir:", output_dir)
    print("doing roi filter ...")
    files = os.listdir(txt_path)
    for file in tqdm(files):
        calib_path = os.path.join(calib_dir, file)
        # calib = open(calib_path).readlines()[0].split(' ')[1]

        with open(calib_path) as f:
            data = json.load(f)
        calib = data["cam_K"][0]

        img_mask_path = os.path.join(
            "/root/MonoUniT/lib/eval_tools/mask", f"{calib}_mask.jpg"
        )

        img_mask = Image.open(img_mask_path)
        output_path = os.path.join(output_dir, file)

        if not os.path.exists(output_path):
            os.system(f"touch {output_path} && chmod 777 {output_path}")
        with open(os.path.join(txt_path, file)) as f:
            fh1 = json.load(f)
        # fh1 = open(os.path.join(txt_path, f), "r")

        filter_data = []
        for line in fh1:
            # line = line.replace("\n", "")
            # if line.replace(" ", "") == "":
            #     continue
            # splitLine = line.split(" ")

            # xmin = float(splitLine[4])
            # ymin = float(splitLine[5])
            # xmax = float(splitLine[6])
            # ymax = float(splitLine[7])
            # center_x = int((xmin + xmax) / 2)
            # center_y = int((ymin + ymax) / 2)
            xmin = line["2d_box"]["xmin"]
            ymin = line["2d_box"]["ymin"]
            xmax = line["2d_box"]["xmax"]
            ymax = line["2d_box"]["ymax"]
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)

            if (
                center_x >= img_mask.width
                or center_y >= img_mask.height
                or center_y < 200
            ):
                continue
            if img_mask.getpixel((center_x, center_y)) == (255, 255, 255):
                filter_data.append(line)
                # fp.write("{}\n".format(line))

        with open(output_path, "a") as fp:
            json.dump(filter_data, fp)

    print("Done")


filter(txt_path, output_dir, calib_dir)
