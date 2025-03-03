####  将标注中的物体3D Box 从虚拟LiDAR坐标系转至相机坐标系( height, width, length, x_loc, y_loc, Z_loc, rotation_y)
import os
import json
from matplotlib.font_manager import json_load
import numpy as np
import math

from tqdm import tqdm

LABEL_DIR = "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/label/camera"
CALIB_DIR = "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/calib/camera_intrinsic"
EXTRINSIC_DIR = (
    "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/calib/virtuallidar_to_camera"
)
SAVE_DIR = "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/label_2"


def load_extrinsic(extrifile):
    """加载虚拟LiDAR到相机的外参矩阵"""
    with open(extrifile) as f:
        extri = json.load(f)
    R = extri["rotation"]
    T = extri["translation"]
    return R, T


# 转换参数
def transform_Lidar2Camera(Lidar_pos, R, T, ry):
    """Args:
        Lidar_pos: 虚拟LiDAR坐标系下的物体位置
        R: 虚拟LiDAR坐标系到相机坐标系的旋转矩阵
        T: 虚拟LiDAR坐标系到相机坐标系的平移
        ry3d(在V2X-Seq中输入ry在虚拟LiDAR坐标系下): 虚拟LiDAR坐标系下，物体绕Z轴旋转到x轴的角度；相机坐标系下物体绕y轴旋转到x轴的角度

    Returns:
        camera_pos: 相机坐标系下的物体位置
        rot_camera_res: 相机坐标系下的物体角度（绕y轴旋转至x轴）
    """
    # 位置转换
    Lidar_pos = np.array(Lidar_pos).reshape(3, 1)
    cam_pos = np.array(R) @ Lidar_pos + np.array(T).reshape(3, 1)
    # NumPy的自动广播规则：
    # 当一维数组 (3,) 与矩阵相乘时，NumPy会将其视为列向量 (3×1)

    # Lidar_pos = np.array(Lidar_pos).T
    # R = np.array(R)
    # T = np.array(T).flatten()
    # ###   camera_pos = Lidar_pos.dot(R) + T  注意点积顺序，这里是 (3,1) * (3，3)，计算错误
    # # camera_pos = R.dot(Lidar_pos) + T       同下，为正确计算
    # cam_pos = R @ Lidar_pos + T

    # 方向转换
    # theta_Lidar = np.array([math.cos(ry), 0, math.sin(ry)]).reshape(3, 1)
    theta_Lidar = np.matrix(data=[math.cos(ry), math.sin(ry), 0]).reshape(3, 1)
    theta_Camera = np.array(R) @ theta_Lidar
    cam_yaw = math.atan2(theta_Camera[2], theta_Camera[0])

    return cam_pos.flatten(), cam_yaw


def process_single_file(label_path, extrinsic_path, save_path):
    """处理单个文件

    Args:
        label_path (_type_): 标签
        extrnsic_path (_type_): 外参
        save_path (_type_): 处理后的标签
    """
    with open(label_path) as f:
        annos = json.load(f)

    R, T = load_extrinsic(extrinsic_path)

    for obj in annos:
        lidar_pos = [
            obj["3d_location"]["x"],
            obj["3d_location"]["y"],
            obj["3d_location"]["z"],
        ]
        lidar_yaw = obj["rotation"]

        cam_pos, cam_yaw = transform_Lidar2Camera(lidar_pos, R, T, lidar_yaw)

        obj["3d_location"]["x"] = cam_pos[0]
        obj["3d_location"]["y"] = cam_pos[1]
        obj["3d_location"]["z"] = cam_pos[2]
        obj["rotation"] = cam_yaw

    # 保存转换结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(annos, f, indent=2)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 遍历所有标注文件
    for filename in tqdm(os.listdir(LABEL_DIR), desc="transform Data"):
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(LABEL_DIR, filename)
        extrinsic_path = os.path.join(EXTRINSIC_DIR, filename)
        save_path = os.path.join(SAVE_DIR, filename)

        # 处理单个文件
        try:
            process_single_file(label_path, extrinsic_path, save_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main()
