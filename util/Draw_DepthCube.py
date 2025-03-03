### 绘制三维深度立方体
import json
import os
import cv2
import numpy as np
import math
import sys
import glob
import shutil
from pyquaternion import Quaternion
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 停止生成.pyc字节码
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# 配置路径
IMG_DIR = "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/image"
LABEL_DIR = "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/label/camera"
CALIB_DIR = "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/calib/camera_intrinsic"
EXTRINSIC_DIR = (
    "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/calib/virtuallidar_to_camera"
)
DENORM_DIR = "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/denorm"
SAVE_DIR = "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/box3d_depth_dense"

# 颜色列表
COLOR_LIST = {
    "car": (0, 0, 255),
    "truck": (0, 255, 255),
    "van": (255, 0, 255),
    "bus": (255, 255, 0),
    "cyclist": (0, 128, 128),
    "motorcyclist": (128, 0, 128),
    "tricyclist": (128, 128, 0),
    "pedestrian": (0, 128, 255),
    "barrow": (255, 0, 128),
}


# Data类定义
class Data:
    def __init__(
        self,
        obj_type="unset",
        truncation=-1,
        occlusion=-1,
        obs_angle=-10,
        x1=-1,
        y1=-1,
        x2=-1,
        y2=-1,
        w=-1,
        h=-1,
        l=-1,
        X=-1000,
        Y=-1000,
        Z=-1000,
        yaw=-10,
        score=-1000,
        detect_id=-1,
        vx=0,
        vy=0,
        vz=0,
    ):
        """init object data"""
        self.obj_type = obj_type
        self.truncation = truncation
        self.occlusion = occlusion
        self.obs_angle = obs_angle
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = w
        self.h = h
        self.l = l
        self.X = X
        self.Y = Y
        self.Z = Z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.yaw = yaw
        self.score = score
        self.ignored = False
        self.valid = False
        self.detect_id = detect_id

    def __str__(self):
        """str"""
        attrs = vars(self)
        return "\n".join("%s: %s" % item for item in attrs.items())


# 数据加载函数
def load_detect_data(filename):
    data = []
    with open(filename) as f:
        label = json.load(f)
        index = 0
        for obj in label:
            # (objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
            t_data = Data()
            # get fields from table
            t_data.obj_type = obj[
                "type"
            ].lower()  # object type [car, pedestrian, cyclist, ...]
            t_data.truncation = float(obj["truncated_state"])  # truncation [0..1]
            t_data.occlusion = int(float(obj["occluded_state"]))  # occlusion  [0,1,2]
            t_data.obs_angle = float(obj["alpha"])  # observation angle [rad]
            t_data.x1 = int(float(obj["2d_box"]["xmin"]))  # left   [px]
            t_data.y1 = int(float(obj["2d_box"]["ymin"]))  # top    [px]
            t_data.x2 = int(float(obj["2d_box"]["xmax"]))  # right  [px]
            t_data.y2 = int(float(obj["2d_box"]["ymax"]))  # bottom [px]
            t_data.h = float(obj["3d_dimensions"]["h"])  # height [m]
            t_data.w = float(obj["3d_dimensions"]["w"])  # width  [m]
            t_data.l = float(obj["3d_dimensions"]["l"])  # length [m]
            t_data.X = float(obj["3d_location"]["x"])  # X [m]
            t_data.Y = float(obj["3d_location"]["y"])  # Y [m]
            t_data.Z = float(obj["3d_location"]["z"])  # Z [m]
            t_data.yaw = float(
                obj["rotation"]
            )  # yaw angle [rad] ; V2X-Seq中是虚拟LiDAR坐标系下（前左上），物体的全局方向角（物体前进方向与虚拟LiDAR坐标系x轴的夹角），范围：-pi~pi
            if len(obj) >= 16:
                t_data.score = float(obj[15])  # detection score
            else:
                t_data.score = 1
            t_data.detect_id = index
            data.append(t_data)
            index = index + 1
    return data


# 地平面方程加载
def load_denorm_data(denormfile):
    with open(denormfile) as f:
        text_file = json.load(f)
    line = text_file[0]["para"]
    parsed = line.split(" ")
    if parsed is not None and len(parsed) > 3:
        de_norm = []
        de_norm.append(float(parsed[0]))
        de_norm.append(float(parsed[1]))
        de_norm.append(float(parsed[2]))
    return np.array(de_norm)


# V2X-Seq 标签文件中的三维目标框信息(pos,dim,ry)在虚拟LiDAR坐标系下，所以需要转至相机坐标系
def transfer_Lidar2Camera(Lidar_pos, R, T, ry):
    """Args:
        Lidar_pos: 虚拟LiDAR坐标系下的物体位置
        R: 虚拟LiDAR坐标系到相机坐标系的旋转矩阵
        T: 虚拟LiDAR坐标系到相机坐标系的平移
        ry3d(在V2X-Seq中输入ry在虚拟LiDAR坐标系下): 虚拟LiDAR坐标系下，物体绕Z轴旋转到x轴的角度；相机坐标系下物体绕y轴旋转到x轴的角度

    Returns:
        camera_pos: 相机坐标系下的物体位置
        rot_camera_res: 相机坐标系下的物体角度（绕y轴旋转至x轴）
    """
    Lidar_pos = np.array(Lidar_pos).T
    R = np.array(R)
    T = np.array(T).flatten()
    ###   camera_pos = Lidar_pos.dot(R) + T  注意点积顺序，这里是 (3,1) * (3，3)，计算错误
    # camera_pos = R.dot(Lidar_pos) + T       同下，为正确计算
    camera_pos = R @ Lidar_pos + T

    theta_Lidar = np.matrix(data=[math.cos(ry), math.sin(ry), 0]).reshape(3, 1)
    theta_Camera = R @ theta_Lidar
    # 因为在相机坐标系下是绕y旋转，所以 x-z 平面, 函数输入的y是z轴的向量
    rot_camera_res = math.atan2(theta_Camera[2], theta_Camera[0])

    return camera_pos, rot_camera_res


# 相机坐标系到地面坐标系的转换 获得地面坐标系是为了获得8个角点的坐标
def compute_c2g_trans(de_norm):
    ground_z_axis = de_norm
    cam_xaxis = np.array([1.0, 0.0, 0.0])
    ground_x_axis = cam_xaxis - cam_xaxis.dot(ground_z_axis) * ground_z_axis
    ground_x_axis = ground_x_axis / np.linalg.norm(ground_x_axis)
    ground_y_axis = np.cross(ground_z_axis, ground_x_axis)
    ground_y_axis = ground_y_axis / np.linalg.norm(ground_y_axis)
    c2g_trans = np.vstack([ground_x_axis, ground_y_axis, ground_z_axis])  # (3, 3)
    return c2g_trans


# 读取相机内参矩阵
def read_kitti_cal(calfile):
    with open(calfile) as f:
        text_file = json.load(f)

    # cam_K 是内参矩阵,P 是带畸变参数的矩阵
    # p2 = np.array(text_file["P"]).reshape(3, 4)
    p2 = np.array(text_file["cam_K"]).reshape(3, 3)
    return p2


# 读取相机外参矩阵
def load_transfer_data(extrifile):
    with open(extrifile) as f:
        extri = json.load(f)
    R = extri["rotation"]
    t = extri["translation"]
    return R, t


# 获得像素坐标系下的8个角点
def project_3d_ground(p2, Object_center, w3d, h3d, l3d, ry3d, c2g_trans, isCenter=True):
    """
    Args:
        p2 (nparray): 相机内参矩阵 不带畸变（3*3），带畸变（3*4），试了带畸变的有问题
        Object_center: 标签文件中物体的三维坐标点(x,y,z)，可能是底部中心，可能是正中心，取决于数据集
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: 虚拟LiDAR坐标系下，物体绕Z轴旋转到x轴的角度；相机坐标系下物体绕y轴旋转到x轴的角度
        c2g_trans: camera_to_ground translation
    Return:
        verts2d: 8个角点在像素坐标系的坐标
        verts3d: 8个角点在相机坐标系的坐标
    """
    Object_center_Ground = c2g_trans.dot(
        Object_center
    )  # 将相机坐标系中的物体坐标转换为地面坐标系
    Object_center_Ground = np.array(Object_center_Ground)
    Object_center_Ground = Object_center_Ground.reshape((3, 1))

    # 将相机坐标系中的物体航向角转换为地面坐标系
    theta0 = np.matrix(data=[math.cos(ry3d), 0, math.sin(ry3d)]).reshape(3, 1)
    theta0 = c2g_trans[:3, :3] @ theta0
    yaw_world_res = math.atan2(theta0[1], theta0[0])

    g2c_trans = np.linalg.inv(c2g_trans)
    verts2d, verts3d = get_camera_3d_8points_g2c(
        w3d,
        h3d,
        l3d,
        yaw_world_res,
        Object_center_Ground,
        g2c_trans,
        p2,
        isCenter,
    )
    verts2d = np.array(verts2d)
    verts3d = np.array(verts3d)
    return verts2d, verts3d


def get_camera_3d_8points_g2c(
    w3d, h3d, l3d, yaw_ground, center_ground, g2c_trans, p2, isCenter
):
    """
    Args:
        w3d: width of object
        h3d: height of object
        l3d: length of object
        yaw_ground: yaw angle in world coordinate
        center_ground: the center or the bottom-center of the object in world-coord
        g2c_trans: ground2camera / world2camera transformation
        p2: projection matrix of size 4x3 (camera intrinsics)
        isCenter: 标签文件中物体的三维坐标
            1: 物体的正中心,
            0: 物体的底部中心
    Returns:
        _corners_2d_: 像素坐标系下物体的8个角点
    """
    ground_r = np.matrix(
        [
            [math.cos(yaw_ground), -math.sin(yaw_ground), 0],
            [math.sin(yaw_ground), math.cos(yaw_ground), 0],
            [0, 0, 1],
        ]
    )  # 地面坐标系下物体的旋转矩阵
    w = w3d
    l = l3d
    h = h3d

    if isCenter:  # True
        corners_3d_ground = np.matrix(
            [
                [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
            ]
        )
    else:  # bottom center, ground: z axis is up
        corners_3d_ground = np.matrix(
            [
                [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                [0, 0, 0, 0, h, h, h, h],
            ]
        )

    corners_3d_ground = np.matrix(ground_r) * np.matrix(corners_3d_ground) + np.matrix(
        center_ground
    )  # [3, 8] 地面坐标系下物体的8个角点坐标
    if g2c_trans.shape[0] == 4:  # world2camera transformation
        ones = np.ones(8).reshape(1, 8).tolist()
        corners_3d_cam = g2c_trans * np.matrix(corners_3d_ground.tolist() + ones)
        corners_3d_cam = corners_3d_cam[:3, :]
    else:  # only consider the rotation 相机坐标系下物体的8个角点坐标
        corners_3d_cam = np.matrix(g2c_trans) * corners_3d_ground  # [3, 8]

    pt = p2[:3, :3] * corners_3d_cam
    corners_2d = pt / pt[2]
    corners_2d_all = corners_2d.reshape(-1)  # 像素坐标系下物体的8个角点坐标
    if True in np.isnan(corners_2d_all):
        print("Invalid projection")
        return None

    corners_2d = corners_2d[0:2].T.tolist()  # (8,2)
    corners_3d_cam = corners_3d_cam.T  # (8,3)
    for i in range(8):
        corners_2d[i][0] = int(corners_2d[i][0])
        corners_2d[i][1] = int(corners_2d[i][1])
    return corners_2d, corners_3d_cam


# 使用一个平面内的三个点得出平面方程
def compute_plane_equation(p1, p2, p3):
    """
    通过三个点计算平面方程系数 (α, β, γ, d)
    Args:
        p1, p2, p3 (np.array): 三个点的坐标，形状为 (3,)
    Returns:
        tuple: 平面方程的系数 (α, β, γ, d)
    """

    # 向量 v1 和 v2
    v1 = p2 - p1
    v2 = p3 - p1
    # 计算法向量 (α, β, γ)
    normal = np.cross(v1, v2)
    alpha, beta, gamma = normal
    # 计算 d
    d = -np.dot(normal, p1)
    return alpha, beta, gamma, d


# 计算像素 (u, v) 对应的深度 z
def compute_depth(alpha, beta, gamma, d, u, v, fx, fy, cx, cy):
    """
    计算像素 (u, v) 对应的深度 z
    Args:
        alpha, beta, gamma, d (float): 平面方程系数
        u, v (int): 像素坐标
        f (float): 相机焦距
        cx, cy (float): 相机主点坐标
    Returns:
        float: 深度 z
    """
    # 由论文中的方程可以得到：
    # x = (u - cx) * z / fx
    # y = (v - cy) * z / fy
    # 将x和y代入论文中方程：
    # alpha * (u - cx) * z / fx + beta * (v - cy) * z / fy + gamma * z + d = 0
    # 解出 z：
    denominator = alpha * (u - cx) / fx + beta * (v - cy) / fy + gamma
    if denominator == 0:
        print("denominator == 0")
        return None  # 无解
    z = -d / denominator
    if z <= 0:
        print("z <= 0")
        return None  # 无效深度
    return z


# 判断点 (u, v) 是否在多边形 polygon 内
def is_point_inside_polygon(u, v, polygon):
    """
    判断点 (u, v) 是否在多边形 polygon 内
    Args:
        u, v (int): 像素坐标
        polygon (list of tuples): 多边形顶点列表 [(u1, v1), (u2, v2), ...]
    Returns:
        bool: True 表示在多边形内，False 表示在外部
    """
    point = (u, v)
    polygon_np = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(polygon_np, point, False)
    return result >= 0  # 包含在内或在边界上


# 生成并保存深度图
def generate_depth_map(name, width, height, depth_values, save_dir):
    """
    生成并保存深度图
    Args:
        name (str): 图像名称
        width (int): 图像宽度
        height (int): 图像高度
        depth_values (dict): 像素坐标到深度值的映射 {(u, v): z}
        save_dir (str): 深度图保存目录
    """
    depth_map = np.zeros((height, width), dtype=np.float32)
    for (u, v), z in depth_values.items():
        depth_map[v, u] = z
    # 归一化到0-255
    # depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    # depth_map_normalized = depth_map_normalized.astype(np.uint8)
    # 保存深度图
    cv2.imwrite(os.path.join(save_dir, f"{name}_depth.png"), depth_map)


# 画出深度立方体
def show_Depthbox(name_list, thresh=0.5):
    # 确保保存目录存在
    if os.path.exists(SAVE_DIR):
        # shutil.rmtree(SAVE_DIR)
        print("Data exists, exiting Program")
        sys.exit()  # 我已经生成了文件，所以加了停止，防止被删除
        logging.info(f"Deleted existing save directory: {SAVE_DIR}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    logging.info(f"Created save directory: {SAVE_DIR}")

    for name in tqdm(name_list, desc="Draw BBox"):
        name = name.split(".")[0]
        label_file = os.path.join(LABEL_DIR, f"{name}.json")
        denorm_file = os.path.join(DENORM_DIR, f"{name}.json")
        calfile = os.path.join(CALIB_DIR, f"{name}.json")
        extrifile = os.path.join(EXTRINSIC_DIR, f"{name}.json")

        # 检查文件是否存在
        if not all(
            os.path.exists(p) for p in [label_file, denorm_file, calfile, extrifile]
        ):
            logging.warning(f"Missing files for {name}, skipping.")
            continue

        result = load_detect_data(label_file)
        de_norm = load_denorm_data(denorm_file)
        c2g_trans = compute_c2g_trans(de_norm)
        p2 = read_kitti_cal(calfile)
        R, T = load_transfer_data(extrifile)
        H, W = 1080, 1920

        depth_map = np.zeros((H, W))

        for t in result:
            if t.score < thresh:
                continue
            if t.obj_type not in COLOR_LIST:
                continue
            if t.w <= 0.05 and t.l <= 0.05 and t.h <= 0.05:  # 无效标注
                continue
            try:
                cam_center, t.yaw = transfer_Lidar2Camera([t.X, t.Y, t.Z], R, T, t.yaw)
                verts2d, verts3d = project_3d_ground(
                    p2,
                    np.array(cam_center),
                    t.w,
                    t.h,
                    t.l,
                    t.yaw,
                    c2g_trans,
                    True,  # V2X-Seq 中标签文件中物体位置为物体正中心
                )
                # if verts2d is None:
                #     continue
                verts2d = verts2d.astype(np.int32)
                # verts3d = verts3d.astype(np.int32)

                # 计算平面方程并获取表面投影顶点
                surfaces = [
                    [0, 1, 2, 3],  # 底面
                    [4, 5, 6, 7],  # 顶面
                    [0, 1, 5, 4],  # 前面
                    [1, 2, 6, 5],  # 右面
                    [2, 3, 7, 6],  # 后面
                    [3, 0, 4, 7],  # 左面
                ]
                plane_equations = []
                surface_vertices_2d = []
                for surface in surfaces:
                    p1, p2_p, p3 = (
                        verts3d[surface[0]],
                        verts3d[surface[1]],
                        verts3d[surface[2]],
                    )
                    alpha, beta, gamma, d = compute_plane_equation(p1, p2_p, p3)
                    plane_equations.append((alpha, beta, gamma, d))
                    # 获取表面在图像上的顶点
                    surface_2d = [tuple(verts2d[idx]) for idx in surface]
                    surface_vertices_2d.append(surface_2d)

                # 获取相机内参
                fx, fy = p2[0, 0], p2[1, 1]
                cx, cy = p2[0, 2], p2[1, 2]

                # 遍历2D边界框内的每个像素
                for u in range(max(t.x1, 0), min(t.x2 + 1, W)):
                    for v in range(max(t.y1, 0), min(t.y2 + 1, H)):
                        depths = []
                        for plane_equation, surface in zip(
                            plane_equations, surface_vertices_2d
                        ):
                            alpha, beta, gamma, d = plane_equation
                            # 检查像素点是否在该表面的投影多边形内
                            if is_point_inside_polygon(u, v, surface):
                                z = compute_depth(
                                    alpha, beta, gamma, d, u, v, fx, fy, cx, cy
                                )
                                if z is not None:
                                    depths.append(z)
                        if depths:
                            z = min(depths)  # 选择最近的深度
                            if depth_map[(v, u)] != 0:
                                depth_map[(v, u)] = min(
                                    depth_map[(v, u)], z
                                )  # 后面的会被前面的遮挡
                            else:
                                depth_map[(v, u)] = z

                # 绘制3D边界框的线条
                # for start, end in [
                #     (2, 1),
                #     (1, 0),
                #     (0, 3),
                #     (2, 3),
                #     (7, 4),
                #     (4, 5),
                #     (5, 6),
                #     (6, 7),
                #     (7, 3),
                #     (1, 5),
                #     (0, 4),
                #     (2, 6),
                # ]:
                #     cv2.line(
                #         depth_map, tuple(verts3d[start]), tuple(verts3d[end]), color_type, 2
                #     )
            except Exception as e:
                logging.error(f"Error processing box in {name}: {e}")
                continue

        # 生成并保存深度图
        # generate_depth_map(name, W, H, depth_values, SAVE_DIR)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{name}_obstacle.png"), depth_map)

        # 保存带有绘制边界框的图像
        # cv2.imwrite(os.path.join(SAVE_DIR, f"{name}.jpg"), img)


if __name__ == "__main__":
    name_list = os.listdir(LABEL_DIR)
    show_Depthbox(name_list)
