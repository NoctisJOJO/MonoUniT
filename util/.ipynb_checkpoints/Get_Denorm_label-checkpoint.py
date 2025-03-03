import numpy as np
from sklearn.linear_model import LinearRegression
import json
import os

BASE_DIR = os.path.join("/root/MonoUniT/dataset/V2X-Seq/infrastructure-side")


# 获取车辆底部位置的 3D 坐标
def get_car_bottom_location(car_3d_location, car_height):
    # 假设车辆底部在车辆高度的一半处
    z_bottom = car_3d_location["z"] - car_height / 2
    return np.array([car_3d_location["x"], car_3d_location["y"], z_bottom])


def fit_plane_to_points_2(points):
    """
    使用最小二乘法拟合平面方程 ax + by + cz + d = 0
    :param points: 地面接触点，形状为 (N, 3)
    :return: 拟合的平面方程系数 [a, b, c, d]
    """
    # 构建一个增广矩阵 [x, y, 1, z]
    points = np.array(points)
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    B = points[:, 2]

    # 使用最小二乘法求解
    plane_coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    # 平面方程为 ax + by + c = z ==> ax + by - z + c = 0
    a, b, c = plane_coeffs
    d = -1.0
    return a, b, -1.0, c


def fit_plane_svd(points):
    # 使用SVD拟合地平面方程
    points = np.array(points)

    # 构造数据矩阵 X
    # 每个点减去其均值（中心化数据）
    points_centered = points - np.mean(points, axis=0)

    # 使用 SVD 对点进行分解
    _, _, vh = np.linalg.svd(points_centered)

    # 通过 SVD 得到的最后一个特征向量即为法向量 [a, b, c]
    normal_vector = vh[2, :]

    # 平面方程：ax + by + cz + d = 0，其中 (a, b, c) 为法向量，d 为平面截距
    a, b, c = normal_vector
    d = -np.dot(normal_vector, np.mean(points, axis=0))  # 计算平面截距

    return a, b, c, d


def transform_virtualLidar_to_camera(a, b, c, d, point, extrinsic_path):
    # 将地平面方程从虚拟激光雷达坐标系转到相机坐标系
    # 问题就是，我需要的是激光雷达坐标系还是相机坐标系呢？
    with open(extrinsic_path) as f:
        extrinsic = json.load(f)
    R = extrinsic["rotation"]
    T = extrinsic["translation"]
    R = np.array(R)  # 激光雷达->相机 旋转矩阵
    T = np.array(T)  # 激光雷达->相机 平移矩阵
    n_Lidar = np.array([a, b, c])  # 激光雷达坐标系下 平面法向量
    n_Camera = R @ n_Lidar  # 相机坐标系下 平面法向量
    point_Lidar = np.array(
        point
    )  # 现在的点是激光雷达坐标系下的（标签文件中3维坐标是激光雷达坐标系）
    point_Camera = R @ point_Lidar + T.flatten()  # 转成相机坐标系下的点
    d = -n_Camera.dot(point_Camera)  # 计算出新的常数项 d
    a, b, c = n_Camera
    return a, b, c, d


def write_denorm(file, data):
    if os.path.exists(file):
        os.system(f"rm {file}")
    os.system(f"touch {file} && sudo chmod 777 {file}")
    with open(file, "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # 获得全部序列
    with open(os.path.join(BASE_DIR, "imageset", "all.txt")) as f:
        ind = f.readlines()
    for i in range(len(ind)):
        with open(
            os.path.join(BASE_DIR, "label", "camera", f"{int(ind[i]):06d}.json")
        ) as f:
            label_info = json.load(f)

        # 计算所有车辆底部3D坐标
        car_bottom_locations = []
        for annotation in label_info:
            car_3d_location = annotation["3d_location"]
            car_height = annotation["3d_dimensions"]["h"]
            car_bottom_location = get_car_bottom_location(car_3d_location, car_height)
            car_bottom_locations.append(car_bottom_location)

        # 使用最小二乘法拟合地平面
        a, b, c, d = fit_plane_to_points_2(car_bottom_locations)
        a *= -1
        b *= -1
        c *= -1
        d *= -1
        print(f"激光雷达坐标系下拟合得到的地平面方程{i}: {a}x + {b}y + {c}z + {d} = 0")
        datas = []
        data = {}
        data["Type"] = "Lidar"
        data["para"] = f"{a} {b} {c} {d}"
        datas.append(data.copy())
        denorm_path = os.path.join(BASE_DIR, "denorm", f"{int(ind[i]):06d}.json")

        a, b, c, d = transform_virtualLidar_to_camera(
            a,
            b,
            c,
            d,
            car_bottom_locations[0],
            os.path.join(
                BASE_DIR, "calib/virtuallidar_to_camera/", f"{int(ind[i]):06d}.json"
            ),
        )
        print(f"相机坐标系下拟合得到的地平面方程{i}: {a}x + {b}y + {c}z + {d} = 0")
        data["Type"] = "Camera"
        data["para"] = f"{a} {b} {c} {d}"
        datas.append(data.copy())
        write_denorm(denorm_path, datas)


if __name__ == "__main__":
    main()
