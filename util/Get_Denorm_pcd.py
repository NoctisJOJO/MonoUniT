import open3d as o3d
import numpy as np
import os


def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd


def plane_fitting(pcd):
    # 使用RANSAC算法拟合平面
    # print(f"点云点数: {len(pcd.points)}")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.02, ransac_n=3, num_iterations=1000
    )
    a, b, c, d = plane_model
    # print(f"平面参数: a={a},b={b},c={c},d={d}")
    return a, b, c, d, inliers


def visualize_point_cloud(pcd, inliers):
    # 可视化点云数据和拟合的平面
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    # 显示点云和拟合的平面
    inlier_cloud.paint_uniform_color([0.1, 0.9, 0.1])  # 地面点云绿色
    outlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # 其余点云红色
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def main():
    # 加载点云文件
    # file_path= "path_to_your_point_cloud.pcd"  # 点云路径
    for i in range(10):
        file_path = (
            f"/root/MonoUniT/dataset/V2X-Seq/infrastructure-side/velodyne/00000{i}.pcd"
        )
        pcd = load_point_cloud(file_path)
        a, b, c, d, inliers = plane_fitting(pcd)
        # visualize_point_cloud(pcd, inliers)
        print(f"地平面方程{i}：{a}x + {b}y + {c}z + {d} = 0")


if __name__ == "__main__":
    main()
