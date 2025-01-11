import cv2
import os
import sys
import numpy as np
import yaml
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /root/MonoUNI/lib
ROOT_DIR = os.path.dirname(BASE_DIR)  # /root/MonoUNI
sys.path.append(ROOT_DIR)

from lib.datasets.rope3d_utils import Calibration, Object3d
from lib.datasets.rope3d_utils import get_objects_from_label

cfg = yaml.load(
    open(ROOT_DIR + "/lib/config.yaml", "r"), Loader=yaml.Loader
)  # 读取配置文件内容

num_class = 4
max_objs = 50
class_name = ["car", "big_vehicle", "pedestrain", "cyclist"]
class2id = {"car": 0, "big_vehicle": 1, "pedestrain": 2, "cyclist": 3}
split_dir = os.path.join(ROOT_DIR, "ImageSets/Rope3D/ImageSets_for_eval/val.txt")
roi_idx_list = [x.strip() for x in open(split_dir).readlines()]
source_roi_label_dir = os.path.join(
    cfg["dataset"]["root_dir"], "label_2_4cls_filter_with_roi_for_eval"
)
output_roi_label_dir = os.path.join(cfg["tester"]["out_dir"], "data_filter")
image_dir = os.path.join(cfg["dataset"]["root_dir"], "image_2")
calib_dir = os.path.join(cfg["dataset"]["root_dir"], "calib")

source_labels = []
for i in tqdm(range(len(roi_idx_list)), desc="load_source_roi_labels Progress"):
    idx = roi_idx_list[i]
    label_file = os.path.join(source_roi_label_dir, idx + ".txt")
    assert os.path.exists(label_file)
    source_labels.append(get_objects_from_label(label_file))
    # with open(label_file, "r") as f:
    #     lines = f.readlines()
    # objects = [Object3d(line) for line in lines]
    # source_labels.append(objects)

output_labels = []
for i in tqdm(range(len(roi_idx_list)), desc="load_output_roi_labels Progress"):
    idx = roi_idx_list[i]
    label_file = os.path.join(output_roi_label_dir, idx + ".txt")
    assert os.path.exists(label_file)
    output_labels.append(get_objects_from_label(label_file))

# img_list = []
# for i in tqdm(range(len(roi_idx_list)), desc="load_image Progress"):
#     idx = roi_idx_list[i]
#     img_file = os.path.join(image_dir, idx + ".jpg")
#     assert os.path.exists(img_file)
#     img_list.append(cv2.imread(img_file))
#     # if img is None:
#     #     print(f"Failed to load image {img_file}")
#     # else:
#     #     print(f"Successfully loaded image {img_file}")

calib = []
for i in tqdm(range(len(roi_idx_list)), desc="load_calib Progress"):
    idx = roi_idx_list[i]
    calib_file = os.path.join(calib_dir, idx + ".txt")
    if os.path.exists(calib_file) != True:
        print("calib_file: ", calib_file)
    assert os.path.exists(calib_file)
    calib.append(Calibration(calib_file))


def draw_bev_box(bev_image, box2d, color=(0, 255, 0), thickness=1):
    """
    在俯视图上绘制2D边界框
    :param bev_image: 俯视图图片
    :param box2d: 2D边界框 (4, 2) 或 (4,)
    :param color: 边框颜色 BGR
    :param thickness: 边框厚度
    """
    if len(box2d.shape) == 1:
        xmin, ymin, xmax, ymax = box2d
        top_left = (round(xmin), round(ymin))  # 四舍五入
        bottom_right = (round(xmax), round(ymax))  # 四舍五入
        cv2.rectangle(bev_image, top_left, bottom_right, color, thickness)
    else:
        # 如果 box2d 是 (4, 2) 形式，表示四个角点的坐标
        for i in range(4):
            start = tuple(map(lambda x: round(x), box2d[i].astype(float)))
            end = tuple(map(lambda x: round(x), box2d[(i + 1) % 4].astype(float)))
            cv2.line(bev_image, start, end, color, thickness)


def draw_3d_box(image, calibration, corners3d, color=(0, 255, 0), thickness=1):
    """
    在图像上绘制3D长方体框
    :param image: 图片
    :param corners3d: 3D角点 (8, 3) 在相机坐标系下
    :param calibration: Calibration 类的实例
    :param color: 边框颜色 BGR
    :param thickness: 边框厚度
    """
    # 将8个角点的坐标由相机坐标系转到图像坐标系
    corners2d, _ = calibration.rect_to_img(corners3d)
    # 定义3D框的线的连接
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # 底面四条边
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # 顶面四条边
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # 底面与顶面连接边
    ]
    for edge in edges:
        start, end = edge
        # print(f"({start},{end})")
        # print(corners3d)
        # start_point = tuple(corners2d[start].astype(int))  # 投影为2D点
        # end_point = tuple(corners2d[end].astype(int))
        start_point = tuple(map(lambda x: round(x), corners2d[start].astype(float)))
        end_point = tuple(map(lambda x: round(x), corners2d[end].astype(float)))
        # print(f"({start_point},{end_point})")
        cv2.line(image, start_point, end_point, color, thickness)


for idx in tqdm(range(len(roi_idx_list)), desc="Drawing 3D boxes"):
    source_label = source_labels[idx]
    output_label = output_labels[idx]
    calibration = calib[idx]
    image_file = os.path.join(image_dir, roi_idx_list[idx] + ".jpg")
    img = cv2.imread(image_file)  # 一张一张加载图片
    bev_image = np.zeros((1080, 500, 3), dtype=np.uint8)  # 创建空白俯视图

    for source_obj in source_label:
        source_corners3d = source_obj.generate_corners3d()
        # print(source_corners3d)
        draw_3d_box(img, calibration, source_corners3d, color=(0, 0, 255), thickness=2)
        bev_box2d = source_obj.to_bev_box2d()
        draw_bev_box(bev_image, bev_box2d, color=(0, 0, 255), thickness=1)
    for output_obj in output_label:
        output_corners3d = output_obj.generate_corners3d()
        draw_3d_box(img, calibration, output_corners3d, color=(0, 255, 0), thickness=2)
        bev_box2d = output_obj.to_bev_box2d()
        draw_bev_box(bev_image, bev_box2d, color=(0, 255, 0), thickness=1)
    combined_image = np.hstack((img, bev_image))

    final_img_dir = os.path.join("/root/autodl-tmp/Rope3D_data/final")
    os.makedirs(final_img_dir, exist_ok=True)
    final_img_path = os.path.join(final_img_dir, f"{roi_idx_list[idx]}.jpg")
    cv2.imwrite(final_img_path, combined_image)

    del img  # 释放内存
