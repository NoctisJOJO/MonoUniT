import json
import os
from tkinter import W
import yaml

DATA_DIR = os.path.join("/root/MonoUniT/dataset/V2X-Seq/infrastructure-side")
with open(os.path.join(DATA_DIR, "data_info.json")) as f:
    data_info = json.load(f)

train_file = open(os.path.join(DATA_DIR, "imageset/train.txt"), "w")
val_file = open(os.path.join(DATA_DIR, "imageset/val.txt"), "w")
all_file = open(os.path.join(DATA_DIR, "imageset/all.txt"), mode="w")

# 获得当前帧、起始帧、结束帧
for entry in data_info:
    # print(entry, "\n")
    frame_id = entry["frame_id"]
    frame_ids = int(frame_id)
    start_frame_id = int(entry["valid_frames_splits"][0]["start_frame_id"])
    end_frame_id = int(entry["valid_frames_splits"][0]["end_frame_id"])
    num_frames = entry["num_frames"]

    # 计算条目中当前帧的范围
    if frame_ids <= num_frames * 0.7 + start_frame_id:
        train_file.write(str(frame_id) + "\n")
    else:
        val_file.write(str(frame_id) + "\n")
    all_file.write(str(frame_id) + "\n")

train_file.close()
val_file.close()
