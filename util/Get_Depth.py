import numpy as np

# 雷达坐标系到相机坐标系的旋转矩阵 (3x3) 和平移向量 (3x1)
rotation = np.array(
    [
        [0.10110970060354674, -1.000711521547571, 0.011220729934179371],
        [-0.2107882792537585, -0.02776873858533773, -0.7986380231662247],
        [0.9730329919664971, 0.08789024621230597, -0.1750508608779963],
    ]
)

translation = np.array(
    [[-2.473586406852653], [5.591948549790895], [1.3769298336996387]]
)

# 物体在雷达坐标系下的坐标 (x, y, z)
lidar_position = np.array([68.043482, 35.090888, -1.23219])

# 将雷达坐标系的 3D 坐标转换为齐次坐标
lidar_position_homogeneous = np.append(lidar_position, 1)

# 构建完整的 4x4 转换矩阵
# [R | t] 和 [0, 0, 0 | 1] 是为了齐次坐标变换
T_lidar_to_camera = np.eye(4)
T_lidar_to_camera[:3, :3] = rotation
T_lidar_to_camera[:3, 3] = translation.flatten()

# 计算物体在相机坐标系下的位置 (4x1 矩阵)
camera_position_homogeneous = T_lidar_to_camera @ lidar_position_homogeneous

# 物体在相机坐标系下的 3D 坐标
camera_position = camera_position_homogeneous[:3]

# 输出物体在相机坐标系下的 3D 坐标
print(f"物体在相机坐标系下的坐标: {camera_position}")

# 深度值是物体在相机坐标系下的 Z 坐标
depth = camera_position[2]

# 输出深度值
print(f"物体在相机坐标系下的深度值 (Z 坐标): {depth}")
