import numpy as np
from tqdm import tqdm
import json
import os
import cv2

BASE_PATH = os.path.join("/root/MonoUniT/dataset/V2X-Seq/infrastructure-side")


def Get_para(ind, i):

    # 地平面方程的参数
    Denorm_path = os.path.join(BASE_PATH, "denorm", f"{int(ind[i]):06d}.json")
    with open(Denorm_path) as f:
        Denorm = json.load(f)
    G = list(map(float, Denorm[1]["para"].split()))
    # a, b, c, d = (
    #     0.004414309963271002,
    #     -0.7998427806614293,
    #     -0.16978293609711798,
    #     5.792817564945654,
    # )
    # G = np.array(G)

    # 相机内参矩阵
    Calib_path = os.path.join(
        BASE_PATH, "calib/camera_intrinsic", f"{int(ind[i]):06d}.json"
    )
    with open(Calib_path) as f:
        Calib = json.load(f)
    K = Calib["cam_K"]
    K = np.array(K).reshape(3, 3)
    # K = np.array([[2162.827939, 0, 968.512893], [0, 2307.430671, 558.797546], [0, 0, 1]])

    return G, K


def compute_depth(x, y, K, G):
    """给定 图像坐标(x,y) 和 地平面方程G 计算深度Z

    Args:
        x (int): 像素
        y (int): 像素
        K (_type_): 内参矩阵
        G (_type_): 地平面方程
    """

    # 将图像坐标(x,y) 转换为相机坐标系下的归一化坐标（X/Z, Y/Z)
    K_inv = np.linalg.inv(K)  # 内参的逆矩阵
    normalized_coordinates = np.dot(K_inv, np.array([x, y, 1]))  # (X/Z, Y/Z, 1)
    X_div_Z, Y_div_Z, _ = normalized_coordinates

    # 使用地平面方程计算深度 Z
    a, b, c, d = G
    # 地平面方程 aX+bY+cZ+d=Z*(a(X/Z)+b(Y/Z)+c)+d=0
    # 解得Z:
    Z = -d / (a * X_div_Z + b * Y_div_Z + c)

    return Z


def Generate_DepthMap(K, G, output_path):
    # 生成深度图
    depth_map = np.zeros((H, W))

    # 计算每个像素的深度值
    for y in range(H):
        for x in range(W):
            # 计算像素坐标
            pixel = np.array([x, y, 1])
            z = compute_depth(x, y, K, G)
            depth_map[y, x] = z

    # 将深度图归一化到[0, 255]的范围
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

    # 保存或显示深度图
    if os.path.exists(output_path):
        os.system(f"rm {output_path}")
    cv2.imwrite(output_path, depth_map)
    # cv2.imshow("Depth Map", depth_map_normalized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    # 图像尺寸
    W = 1920
    H = 1080

    # 获得全部序列
    with open(os.path.join(BASE_PATH, "imageset", "all.txt")) as f:
        ind = f.readlines()
    for i in tqdm(range(len(ind)), desc="Draw Depth Map"):
        G, K = Get_para(ind, i)
        output_path = os.path.join(BASE_PATH, "depth", f"{int(ind[i]):06d}.jpg")
        Generate_DepthMap(K, G, output_path)
