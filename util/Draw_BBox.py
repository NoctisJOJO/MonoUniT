### 绘制框，现在有问题，转向角还没调好
import json
import os
import cv2
import numpy as np
import math
import sys

# import config
import glob as gb

# import pdb
# import yaml
import shutil
from pyquaternion import Quaternion

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# pwd = os.getcwd()
# img_dir = ".\\datasets\\testing\\image"
# label_dir = ".\\datasets\\training\\label"
# calib_dir = ".\\datasets\\testing\\calib"
# denorm_dir = ".\\datasets\\testing\\denorm_1"
# save_dir = ".\\datasets\\testing\\visualize_original"
img_dir = os.path.join("/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/image")
label_dir = os.path.join(
    "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/label/camera"
)
calib_dir = os.path.join(
    "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/calib/camera_intrinsic"
)
extrinsic = os.path.join(
    "/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/calib/virtuallidar_to_camera"
)
denorm_dir = os.path.join("/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/denorm")
save_dir = os.path.join("/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/final_2")

color_list = {
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


def progress(count, total, status=""):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + ">" + "-" * (bar_len - filled_len)
    sys.stdout.write("[%s] %s%s ...%s\r" % (bar, percents, "%", status))
    sys.stdout.flush()


# 从label文件中加载检测内容
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
            )  # # yaw angle [rad] ; V2X-Seq中是虚拟LiDAR坐标系下（前左上），物体的全局方向角（物体前进方向与虚拟LiDAR坐标系x轴的夹角），范围：-pi~pi
            if len(obj) >= 16:
                t_data.score = float(obj[15])  # detection score
            else:
                t_data.score = 1
            t_data.detect_id = index
            data.append(t_data)
            index = index + 1
    return data


# 读取v2x denorm文件内容
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


# 标签文件中的三维目标框信息在虚拟激光雷达坐标系下，所以需要转至相机坐标系
def transfer_Lidar2Camera(Lidar_pos, R, T):
    Lidar_pos = np.array(Lidar_pos).T
    R = np.array(R)
    T = np.array(T).flatten()
    camera_pos = Lidar_pos.dot(R) + T
    return camera_pos


# 计算相机坐标系到地面的转化
def compute_c2g_trans(de_norm):
    ground_z_axis = de_norm
    cam_xaxis = np.array([1.0, 0.0, 0.0])
    ground_x_axis = cam_xaxis - cam_xaxis.dot(ground_z_axis) * ground_z_axis
    ground_x_axis = ground_x_axis / np.linalg.norm(ground_x_axis)
    ground_y_axis = np.cross(ground_z_axis, ground_x_axis)
    ground_y_axis = ground_y_axis / np.linalg.norm(ground_y_axis)
    c2g_trans = np.vstack([ground_x_axis, ground_y_axis, ground_z_axis])  # (3, 3)
    return c2g_trans


# 读取相机校准文件P2
def read_kitti_cal(calfile):
    with open(calfile) as f:
        text_file = json.load(f)
    # for line in text_file:
    #     parsed = line.split("\n")[0].split(" ")
    #     # cls x y w h occ x y w h ign ang
    #     if parsed is not None and parsed[0] == "P2:":
    #         p2 = np.zeros([4, 4], dtype=float)
    #         p2[0, 0] = parsed[1]
    #         p2[0, 1] = parsed[2]
    #         p2[0, 2] = parsed[3]
    #         p2[0, 3] = parsed[4]
    #         p2[1, 0] = parsed[5]
    #         p2[1, 1] = parsed[6]
    #         p2[1, 2] = parsed[7]
    #         p2[1, 3] = parsed[8]
    #         p2[2, 0] = parsed[9]
    #         p2[2, 1] = parsed[10]
    #         p2[2, 2] = parsed[11]
    #         p2[2, 3] = parsed[12]
    #         p2[3, 3] = 1
    p2 = np.array(text_file["P"]).reshape(3, 4)
    return p2


# 从外参文件中获得旋转矩阵和平移矩阵
def load_transfer_data(extrifile):
    with open(extrifile) as f:
        extri = json.load(f)
    R = extri["rotation"]
    t = extri["translation"]
    return R, t


# Projects a 3D box into 2D vertices using the camera2ground tranformation
def project_3d_ground(p2, de_bottom_center, w3d, h3d, l3d, ry3d, de_norm, c2g_trans):
    """
    Args:
        p2 (nparray): projection matrix of size 4x3
        de_bottom_center: bottom center XYZ-coord of the object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
        de_norm: [a,b,c] denotes the ground equation plane
        c2g_trans: camera_to_ground translation
    """
    de_bottom_center_in_ground = c2g_trans.dot(
        de_bottom_center
    )  # convert de_bottom_center in Camera coord into Ground coord
    bottom_center_ground = np.array(de_bottom_center_in_ground)
    bottom_center_ground = bottom_center_ground.reshape((3, 1))
    theta = np.matrix([math.cos(ry3d), 0, -math.sin(ry3d)]).reshape(3, 1)
    theta0 = c2g_trans[:3, :3] * theta  # first column
    yaw_world_res = math.atan2(theta0[1], theta0[0])
    g2c_trans = np.linalg.inv(c2g_trans)
    verts3d = get_camera_3d_8points_g2c(
        w3d,
        h3d,
        l3d,
        yaw_world_res,
        bottom_center_ground,
        g2c_trans,
        p2,
        isCenter=False,
    )
    verts3d = np.array(verts3d)
    return verts3d


def get_camera_3d_8points_g2c(
    w3d, h3d, l3d, yaw_ground, center_ground, g2c_trans, p2, isCenter=True
):
    """
    function: projection 3D to 2D
    w3d: width of object
    h3d: height of object
    l3d: length of object
    yaw_world: yaw angle in world coordinate
    center_world: the center or the bottom-center of the object in world-coord
    g2c_trans: ground2camera / world2camera transformation
    p2: projection matrix of size 4x3 (camera intrinsics)
    isCenter:
        1: center,
        0: bottom
    """
    ground_r = np.matrix(
        [
            [math.cos(yaw_ground), -math.sin(yaw_ground), 0],
            [math.sin(yaw_ground), math.cos(yaw_ground), 0],
            [0, 0, 1],
        ]
    )
    # l, w, h = obj_size
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
    )  # [3, 8]
    if g2c_trans.shape[0] == 4:  # world2camera transformation
        ones = np.ones(8).reshape(1, 8).tolist()
        corners_3d_cam = g2c_trans * np.matrix(corners_3d_ground.tolist() + ones)
        corners_3d_cam = corners_3d_cam[:3, :]
    else:  # only consider the rotation
        corners_3d_cam = np.matrix(g2c_trans) * corners_3d_ground  # [3, 8]

    ###### 这里有问题，算出来的值是负的
    pt = p2[:3, :3] * corners_3d_cam
    corners_2d = pt / pt[2]
    corners_2d_all = corners_2d.reshape(-1)
    if True in np.isnan(corners_2d_all):
        print("Invalid projection")
        return None

    corners_2d = corners_2d[0:2].T.tolist()
    for i in range(8):
        corners_2d[i][0] = int(corners_2d[i][0])
        corners_2d[i][1] = int(corners_2d[i][1])
    return corners_2d


# 显示2D和3D框
def show_box_with_roll(name_list, thresh=0.5):
    for i, name in enumerate(name_list):
        name = name.strip()
        name = name.split("/")
        name = name[-1].split(".")[0]
        img_path = os.path.join(img_dir, "%s.jpg" % (name))
        progress(i, len(name_list))
        detection_file = os.path.join(label_dir, "%s.json" % (name))
        result = load_detect_data(detection_file)
        denorm_file = os.path.join(denorm_dir, "%s.json" % (name))
        de_norm = load_denorm_data(denorm_file)  # [ax+by+cz+d=0]
        c2g_trans = compute_c2g_trans(de_norm)

        calfile = os.path.join(calib_dir, "%s.json" % (name))
        p2 = read_kitti_cal(calfile)
        extrifile = os.path.join(extrinsic, "%s.json" % (name))
        R, T = load_transfer_data(extrifile)

        img = cv2.imread(img_path)
        h, w, c = img.shape

        for result_index in range(len(result)):
            t = result[result_index]
            if t.score < thresh:
                continue
            if t.obj_type not in color_list.keys():
                continue
            color_type = color_list[t.obj_type]
            cv2.rectangle(
                img, (t.x1, t.y1), (t.x2, t.y2), (255, 255, 255), 1
            )  # 2D标注框
            if t.w <= 0.05 and t.l <= 0.05 and t.h <= 0.05:  # 无效的annos
                continue
            t.X, t.Y, t.Z = transfer_Lidar2Camera([t.X, t.Y, t.Z], R, T)
            cam_bottom_center = [t.X, t.Y, t.Z]  # 相机坐标系的底面中心
            verts3d = project_3d_ground(
                p2,
                np.array(cam_bottom_center),
                t.w,
                t.h,
                t.l,
                t.yaw,
                de_norm,
                c2g_trans,
            )
            if verts3d is None:
                continue
            verts3d = verts3d.astype(np.int32)

            # draw projection
            cv2.line(img, tuple(verts3d[2]), tuple(verts3d[1]), color_type, 2)
            cv2.line(img, tuple(verts3d[1]), tuple(verts3d[0]), color_type, 2)
            cv2.line(img, tuple(verts3d[0]), tuple(verts3d[3]), color_type, 2)
            cv2.line(img, tuple(verts3d[2]), tuple(verts3d[3]), color_type, 2)
            cv2.line(img, tuple(verts3d[7]), tuple(verts3d[4]), color_type, 2)
            cv2.line(img, tuple(verts3d[4]), tuple(verts3d[5]), color_type, 2)
            cv2.line(img, tuple(verts3d[5]), tuple(verts3d[6]), color_type, 2)
            cv2.line(img, tuple(verts3d[6]), tuple(verts3d[7]), color_type, 2)
            cv2.line(img, tuple(verts3d[7]), tuple(verts3d[3]), color_type, 2)
            cv2.line(img, tuple(verts3d[1]), tuple(verts3d[5]), color_type, 2)
            cv2.line(img, tuple(verts3d[0]), tuple(verts3d[4]), color_type, 2)
            cv2.line(img, tuple(verts3d[2]), tuple(verts3d[6]), color_type, 2)
        cv2.imwrite("%s/%s.jpg" % (save_dir, name), img)


if __name__ == "__main__":
    name_list = gb.glob(label_dir + "/*")
    # print(name_list)
    if os.path.exists(save_dir):
        os.system(f"rm -r {save_dir}")
    os.mkdir(save_dir)
    show_box_with_roll(name_list)
