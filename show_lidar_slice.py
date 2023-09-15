import os
import math
import json
import torch
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_pts_xz(x_range, y_range, z_range, ppm, ppm_y, y_res=None):
    x_res = abs(int((x_range[1] - x_range[0]) * ppm))
    z_res = abs(int((z_range[1] - z_range[0]) * ppm))
    if y_res is None:
        y_res = abs(int((y_range[1] - y_range[0]) * ppm_y))

    x = torch.linspace(x_range[0], x_range[1], x_res).view(
        1, 1, x_res).expand(y_res, z_res, -1)
    z = torch.linspace(z_range[0], z_range[1], z_res).view(
        1, z_res, 1).expand(y_res, -1, x_res)
    if y_res == 1:
        y = torch.tensor([y_range[0] * .5 + y_range[1] * .5]
                         ).view(y_res, 1, 1).expand(-1, z_res, x_res)
    else:
        y = torch.linspace(y_range[0], y_range[1], y_res).view(
            y_res, 1, 1).expand(-1, z_res, x_res)
    xyz = torch.stack((x, y, z), dim=-1)

    return xyz, (x_res, y_res, z_res)


def get_pts_xy(x_range, y_range, z_range, ppm, ppm_z, z_res=None):
    x_res = abs(int((x_range[1] - x_range[0]) * ppm))
    y_res = abs(int((y_range[1] - y_range[0]) * ppm))
    if z_res is None:
        z_res = abs(int((z_range[1] - z_range[0]) * ppm_z))

    x = torch.linspace(x_range[0], x_range[1], x_res).view(
        1, 1, x_res).expand(-1, y_res, z_res)
    y = torch.linspace(y_range[0], y_range[1], y_res).view(
        1, y_res, 1).expand(x_res, -1, z_res)
    if z_res == 1:
        z = torch.tensor([z_range[0] * .5 + z_range[1] * .5]
                         ).view(1, 1, z_res).expand(-1, y_res, x_res)
    else:
        z = torch.linspace(z_range[0], z_range[1], z_res).view(
            z_res, 1, 1).expand(-1, y_res, x_res)
    xyz = torch.stack((x, y, z), dim=-1)

    return xyz, (x_res, y_res, z_res)


def get_pts_yxz(x_range, y_range, z_range, ppm, ppm_z, z_res=None):
    x_res = abs(int((x_range[1] - x_range[0]) * ppm))
    y_res = abs(int((y_range[1] - y_range[0]) * ppm))
    if z_res is None:
        z_res = abs(int((z_range[1] - z_range[0]) * ppm_z))

    y = torch.linspace(y_range[0], y_range[1], y_res).view(
        y_res, 1, 1).expand(-1, x_res, z_res)
    x = torch.linspace(x_range[0], x_range[1], x_res).view(
        1, x_res, 1).expand(y_res, -1, z_res)
    if z_res == 1:
        z = torch.tensor([z_range[0] * .5 + z_range[1] * .5]
                         ).view(1, 1, z_res).expand(y_res, x_res, -1)
    else:
        z = torch.linspace(z_range[0], z_range[1], z_res).view(
            1, 1, z_res).expand(y_res, x_res, -1)
        
    xyz = torch.stack((x, y, z), dim=-1)

    return xyz, (x_res, y_res, z_res)


# This function takes all points between min_y and max_y and projects them into the x-z plane.
# To avoid cases where there are no points at the top end, we consider also points that are beyond the maximum z distance.
# The points are then converted to polar coordinates and sorted by angle.
def get_lidar_slices(point_clouds, velo_poses, y_range, y_res, max_dist):
    slices = []
    ys = torch.linspace(y_range[0], y_range[1], y_res)
    if y_res > 1:
        slice_height = ys[1] - ys[0]
    else:
        slice_height = 0
    n_bins = 360

    for y in ys:
        if y_res == 1:
            min_y = y
            max_y = y_range[-1]
        else:
            min_y = y - slice_height / 2
            max_y = y + slice_height / 2

        slice = []

        for pc, velo_pose in zip(point_clouds, velo_poses):
            pc_world = (velo_pose @ pc.T).T

            mask = ((pc_world[:, 2] >= min_y) & (pc_world[:, 2] <= max_y)) | (
                torch.norm(pc_world[:, :3], dim=-1) >= max_dist)

            slice_points = pc[mask, :2]
            # show_2d_point(slice_points)

            # convert to polar cor
            angles = torch.atan2(slice_points[:, 1], slice_points[:, 0])
            dists = torch.norm(slice_points, dim=-1)
            # plot_polar(angles, dists)
            slice_points_polar = torch.stack((angles, dists), dim=1)
            # Sort by angles for fast lookup
            slice_points_polar = slice_points_polar[torch.sort(angles)[
                1], :]   # alpha, d

            slice_points_polar_binned = torch.zeros_like(
                slice_points_polar[:n_bins, :])
            bin_borders = torch.linspace(-math.pi, math.pi,
                                         n_bins+1, device=slice_points_polar.device)  # size 361. I guess the design make sure to inclue the slice_points_polar and take value for floor and ceil.

            dist = slice_points_polar[0, 1]

            # To reduce noise, we bin the lidar points into bins of 1deg and then take the minimum distance per bin.
            border_is = torch.searchsorted(
                slice_points_polar[:, 0], bin_borders)

            # assign alpha, d to 360 eg.slice_height It's the S
            for i in range(n_bins):
                # correspond to floor and ceil
                left_i, right_i = border_is[i], border_is[i+1]
                angle = (bin_borders[i] + bin_borders[i+1]) * .5
                if right_i > left_i:
                    dist = torch.min(slice_points_polar[left_i:right_i, 1])

                slice_points_polar_binned[i, 0] = angle
                slice_points_polar_binned[i, 1] = dist

            slice_points_polar = slice_points_polar_binned
            # plot_polar(slice_points_polar[:, 0], slice_points_polar[:, 1])
            # Append first element to last to have full 360deg coverage
            slice_points_polar = torch.cat((torch.tensor([[slice_points_polar[-1, 0] - math.pi * 2, slice_points_polar[-1, 1]]], device=slice_points_polar.device),
                                           slice_points_polar,
                                           torch.tensor([[slice_points_polar[0, 0] + math.pi * 2, slice_points_polar[0, 1]]], device=slice_points_polar.device)), dim=0)

            slice.append(slice_points_polar)

        slices.append(slice)

    return slices


def check_occupancy(pts, slices, velo_poses, min_dist=3):
    is_occupied = torch.ones_like(pts[:, 0])
    is_visible = torch.zeros_like(pts[:, 0], dtype=torch.bool)

    # inconsistent with article. instead of 0.5
    thresh = (len(slices[0]) - 2) / len(slices[0])

    pts = torch.cat((pts, torch.ones_like(pts[:, :1])), dim=-1)

    world_to_velos = torch.inverse(velo_poses)

    step = pts.shape[0] // len(slices)

    for i, slice in enumerate(slices):
        for j, (lidar_polar, world_to_velo) in enumerate(zip(slice, world_to_velos)):
            pts_velo = (world_to_velo @ pts[i*step: (i+1)*step, :].T).T

            # Convert query points to polar coordinates in velo space
            angles = torch.atan2(pts_velo[:, 1], pts_velo[:, 0])
            dists = torch.norm(pts_velo, dim=-1)

            indices = torch.searchsorted(
                lidar_polar[:, 0].contiguous(), angles)

            left_angles = lidar_polar[indices-1, 0]
            right_angles = lidar_polar[indices, 0]

            left_dists = lidar_polar[indices-1, 1]
            right_dists = lidar_polar[indices, 1]

            interp = (angles - left_angles) / (right_angles - left_angles)
            surface_dist = left_dists * (1 - interp) + right_dists * interp
            # plot_polar(angles, dists)
            # plot_polar(lidar_polar[:, 0], lidar_polar[:, 1])
            # plot_polar(angles, surface_dist)
            is_occupied_velo = (dists > surface_dist) | (dists < min_dist)

            is_occupied[i*step: (i+1)*step] += is_occupied_velo.float()

            if j == 0:
                is_visible[i*step: (i+1)*step] |= ~is_occupied_velo

    is_occupied /= len(slices[0])

    is_occupied = is_occupied > thresh

    return is_occupied, is_visible


def xyz2pcd(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.colors = o3d.utility.Vector3dVector(np.ones((xyz.shape[0], 3)) * [1, 0.55, 0])
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            np.ones((xyz.shape[0], 3)) * color)

    return pcd


def show_point(pcd: list):
    o3d.visualization.draw_geometries(pcd)


def show_2d_point(x, y):
    plt.scatter(x, y)
    plt.show()


def plot_polar(theta: np.array, r: np.array):
    ax = plt.subplot(111, polar=True)
    ax.scatter(theta, r)
    plt.show()


if __name__ == '__main__':
    y_range = [-3.5, 5]
    x_range = [4.5, 30]
    z_range = [-0.5, 1.2]
    max_dist = (x_range[1] ** 2 + y_range[1] ** 2) ** .5

    z_res = 1
    ppm = 7
    ppm_y = 2

    # img_folder = "/kube/home/DATASET/processed/guangwei/20230726/停车场_奥林匹克公园南西入停车场/20230726113754_passby/CAMERA/FW"
    point_folder = "/home/xufengxue/Desktop/pyproject/point"
    pose_config_file = "/home/xufengxue/Desktop/pyproject/lidar.json"

    with open(pose_config_file, 'r') as file:
        config_dict = json.load(file)

    pose_dict = config_dict['pose_array'][:20]
    point_file_list = sorted(os.listdir(point_folder))[:20]

    # Load lidar pointclouds
    points_all = []
    velo_poses = []
    world_points = []
    for point, pose in zip(point_file_list, pose_dict):
        points = np.fromfile(os.path.join(point_folder, point),
                             dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0  # convert to homo cord
        points = torch.tensor(points)
        pose = torch.tensor(pose["lidar2world"],
                            dtype=torch.float32).view(-1, 4)
        world_point = (pose @ points.T).T
        points_all.append(points)
        velo_poses.append(pose)
        world_points.append(world_point)

    world_points = torch.cat(world_points, dim=0)
    lidar_points = torch.cat(points_all, dim=0)
    velo_poses = torch.stack(velo_poses, dim=0)

    # Get pts
    q_pts, (xd, yd, zd) = get_pts_yxz(x_range, y_range, z_range, ppm, ppm_y, z_res)
    q_pts = q_pts.to(velo_poses.device).view(-1, 3)
    slices = get_lidar_slices(points_all, velo_poses, z_range, zd, max_dist)
    is_occupied, is_visible = check_occupancy(q_pts, slices, velo_poses)
    # Only not visible points can be occupied
    is_occupied &= ~is_visible

    # visual 3d origin
    lidar_points_pcd = xyz2pcd(lidar_points[:,:3].numpy(), color=[0., 0., 0.7])
    world_points_pcd = xyz2pcd(world_points[:,:3].numpy(), color=[0.7, 0., 0.7])
    show_point([world_points_pcd, lidar_points_pcd])

    # visual 3d with mask
    mask = (
        (world_points[:, 2] >= z_range[0]) & (world_points[:, 2] <= z_range[1]) &
        (world_points[:, 0] >= x_range[0]) & (world_points[:, 0] <= x_range[1]) &
        (world_points[:, 1] >= y_range[0]) & (world_points[:, 1] <= y_range[1])
    )
    world_points_masked_3d = world_points[mask, :3]
    lidar_points_masked_3d = lidar_points[mask, :3]
    world_masked_pcd = xyz2pcd(world_points_masked_3d[:, :3].numpy(), color=[1., 0.7, 0.])
    lidar_masked_pcd = xyz2pcd(lidar_points_masked_3d[:, :3].numpy(), color=[0.7, 0.7, 0.7])
    show_point([world_masked_pcd, lidar_masked_pcd])

    # visual 2d with mask
    world_points_masked_2d = world_points[mask, :2]
    lidar_points_masked_2d = lidar_points[mask, :2]

    # visual query points wether occ and mask in 2D
    show_2d_point(q_pts[:,0], q_pts[:,1])
    show_2d_point(q_pts[is_occupied][:,0], q_pts[is_occupied][:,1])
    show_2d_point(world_points_masked_2d[:,0], world_points_masked_2d[:,1])

    # visual query points in 3D
    q_pts_masked_3d = q_pts[is_occupied, :3]
    q_pts_masked_pcd = xyz2pcd(q_pts_masked_3d[:, :3].numpy(), color=[0, 0.8, 0.8])
    show_point([world_masked_pcd, q_pts_masked_pcd, world_points_pcd])
    
