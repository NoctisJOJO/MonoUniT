import math
import os
import numpy as np
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

BASE_DIR = os.path.join("/root/MonoUniT/dataset/V2X-Seq/infrastructure-side")


def transform_to_camera_coordinates(rotation, translation, object_position):
    """
    将激光雷达坐标系下的物体位置转换到相机坐标系
    :param rotation: 雷达→相机 旋转矩阵
    :param translation: 雷达→相机 平移矩阵
    :param object_position: 雷达坐标系下的物体位置 (x, y, z)
    :return: 物体在相机坐标系中的位置 (x, y, z)
    """

    # 将物体位置转换到相机坐标系
    object_position = np.array(
        [object_position["x"], object_position["y"], object_position["z"]]
    )
    rotation = np.array(rotation)
    translation = np.array(translation).flatten()
    object_camera_position = np.dot(rotation, object_position) + translation
    return object_camera_position


def get_ground_contact_points(objects, rotation, translation):
    """
    计算物体接触地面的坐标（在相机坐标系中）
    :param objects: 包含物体信息的列表，每个元素为一个字典 {'type': type, 'dimensions': (h, w, l), 'position': (x, y, z)}
    :param rotation: 雷达→相机 旋转矩阵
    :param translation: 雷达→相机 平移矩阵
    :return: 接触地面的点列表 [(x, y, z), ...]
    """
    ground_contact_points = []

    for obj in objects:
        # 获取物体在激光雷达坐标系下的位置和尺寸
        obj_position = np.array(
            [obj["3d_location"][0], obj["3d_location"][1], obj["3d_location"][2]]
        )
        obj_dimensions = np.array(
            [obj["3d_dimensions"][0], obj["3d_dimensions"][1], obj["3d_dimensions"][2]]
        )

        # 将物体的激光雷达坐标系位置转换为相机坐标系
        camera_position = transform_to_camera_coordinates(
            rotation, translation, obj_position
        )

        # 计算接触地面的坐标（减去物体高度的一半）
        # ground_z = (camera_position[2] - obj_dimensions[0]) / 2.0
        # ground_z = 100
        # ground_contact_points.append([camera_position[0], camera_position[1], ground_z])
        ground_contact_points.append(camera_position)

    return np.array(ground_contact_points)


def fit_plane_to_points(points):
    R, Z = [], []
    for j in range(len(points)):
        R.append([float(points[j][0]), float(points[j][1]), 1])  # (x,y,1)
        Z.append([float(points[j][2])])  #  z

    R = np.mat(R)

    # 这是正规方法，最小化误差的平方和 power(|| R * A - Z || )
    # ==>  A = (R.T*R)的逆 * R.T * Z
    A = np.dot(np.dot(np.linalg.inv(np.dot(R.T, R)), R.T), Z)

    # 使用伪逆计算回归系数 可以计算 Moore-Penrose 伪逆，适用于 R.T * R 奇异或接近奇异的情况。
    # A = np.linalg.pinv(R) @ Z  # 或使用 np.dot(np.linalg.pinv(R), Z)

    # 使用 lstsq 直接求解最小二乘问题，内部使用更稳定和高效的算法，如 QR 分解或奇异值分解（SVD）
    # A, residuals, rank, s = np.linalg.lstsq(R, Z, rcond=None)
    A = np.array(A, dtype="float32").flatten()

    a, b, d = A
    C = -1.0 / math.sqrt(a * a + b * b + 1)
    A, B, D = -a * C, -b * C, -d * C
    return np.array([A, B, C, D])


def write_denorm(file, data):
    os.system(f"touch {file} && chmod 777 {file}")
    with open(file, "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # 获得全部序列
    denorm_dir = os.path.join(BASE_DIR, "denorm")
    if os.path.exists(denorm_dir):
        os.system(f"rm -r {denorm_dir}")
    os.mkdir(denorm_dir)
    with open(os.path.join(BASE_DIR, "imageset", "all.txt")) as f:
        ind = f.readlines()
    for i in tqdm(range(len(ind)), desc="Compute Denorm"):
        with open(
            os.path.join(BASE_DIR, "label", "camera", f"{int(ind[i]):06d}.json")
        ) as f:
            label_info = json.load(f)
        if (
            len(label_info) < 3
        ):  # 当标签中的物体数量少于3个，无法拟合地平面，直接用上一张图像的平面
            denorm_path = os.path.join(BASE_DIR, "denorm", f"{int(ind[i]):06d}.json")
            write_denorm(denorm_path, datas)
            continue
        with open(
            os.path.join(
                BASE_DIR, "calib", "virtuallidar_to_camera", f"{int(ind[i]):06d}.json"
            )
        ) as f:
            trans_info = json.load(f)
        R = trans_info["rotation"]
        t = trans_info["translation"]

        # 计算所有车辆底部3D坐标
        car_bottom_locations = []
        for annotation in label_info:
            car_3d_location = annotation["3d_location"]
            car_height = annotation["3d_dimensions"]["h"]
            car_bottom_location = transform_to_camera_coordinates(R, t, car_3d_location)
            car_bottom_locations.append(car_bottom_location)

        # print(f"ind: {ind},  len : {len(car_bottom_locations)}")
        # 使用最小二乘法拟合地平面
        a, b, c, d = fit_plane_to_points(car_bottom_locations)
        # a *= -1
        # b *= -1
        # c *= -1
        # d *= -1
        # print(f"相机坐标系下拟合得到的地平面方程{i}: {a}x + {b}y + {c}z + {d} = 0")
        datas = []
        data = {}
        data["Type"] = "camera"
        data["para"] = f"{a} {b} {c} {d}"
        datas.append(data.copy())

        denorm_path = os.path.join(BASE_DIR, "denorm", f"{int(ind[i]):06d}.json")
        write_denorm(denorm_path, datas)


if __name__ == "__main__":
    main()
