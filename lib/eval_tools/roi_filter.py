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

# import config4cls as config

# parser = argparse.ArgumentParser(description = "convert parameters")
# parser.add_argument("--txt_dir", required=True)
# parser.add_argument("--output_dir", required=True)
# args = parser.parse_args()

# txt_path = "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/label_2_4cls_for_train"
# output_dir = "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/label_2_4cls_filter_with_roi_for_eval"
# calib_dir = config.intrinsic_dir


def filter(txt_path, output_dir, calib_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    print("outputdir:", output_dir)
    print("doing roi filter ...")
    files = os.listdir(txt_path)
    for f in files:
        calib_path = os.path.join(calib_dir, f)
        # calib = open(calib_path).readlines()[0].split(' ')[1]
        calib = json.load(open(calib_path))["cam_K"][0]

        # img_mask_path = glob.glob("./lib/eval_tools/mask/" + str(calib) + "*.jpg")[0]
        img_mask_path = os.path.join(
            "/root/MonoUniT/lib/eval_tools/mask/", f"{calib}_mask.jpg"
        )

        img_mask = Image.open(img_mask_path)
        part = f.split("_")
        output_path = os.path.join(output_dir, f)
        fh1 = json.load(open(os.path.join(txt_path, f), "r"))

        after_roi = []
        for line in fh1:
            # line = line.replace("\n", "")
            # if line.replace(' ', '') == '':
            #     continue
            # splitLine = line.split(" ")

            # xmin = float(splitLine[4])
            # ymin = float(splitLine[5])
            # xmax = float(splitLine[6])
            # ymax = float(splitLine[7])
            # center_x = int((xmin+xmax) / 2)
            # center_y = int((ymin+ymax) / 2)
            center_x = int((line["2d_box"]["xmin"] + line["2d_box"]["xmax"]) / 2)
            center_y = int((line["2d_box"]["ymin"] + line["2d_box"]["ymax"]) / 2)

            if (
                center_x >= img_mask.width
                or center_y >= img_mask.height
                or center_y < 200
            ):
                continue
            if img_mask.getpixel((center_x, center_y)) == (255, 255, 255):
                # with open(output_path, 'a') as fp:
                #     fp.write("{}\n".format(line))
                after_roi.append(line)
        with open(output_path, "w") as f:
            json.dump(after_roi, f, indent=2)

    print("Done")


# if __name__ == "__main__":
#     filter(txt_path, output_dir, calib_dir)
