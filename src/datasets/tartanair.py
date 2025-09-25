import glob
import os
from typing import Optional

import numpy as np
import torch
from natsort import natsorted
from scipy.spatial.transform import Rotation

from src.datasets.basedataset import GradSLAMDataset

class TartanAirSemanticDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        print("Load tartanair dataset!!!")
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "pose_left.txt")
        self.pose_path_right = os.path.join(self.input_folder, "pose_right.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/image_left/00*.png"))
        color_paths_right = natsorted(glob.glob(f"{self.input_folder}/image_right/00*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth_left/00*.npy"))
        depth_paths_right = natsorted(glob.glob(f"{self.input_folder}/depth_right/00*.npy"))
        object_paths = natsorted(glob.glob(f"{self.input_folder}/seg_left/00*.npy"))
        object_paths_right = natsorted(glob.glob(f"{self.input_folder}/seg_right/00*.npy"))

        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, color_paths_right, depth_paths, depth_paths_right, object_paths, object_paths_right, embedding_paths
    
    def load_poses(self):
        poses = []
        poses_right = []

        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            # 解析4元组形式 (x, y, z, qx, qy, qz, qw)
            values = list(map(float, line.split()))
            if len(values) == 7:
                # 位置部分 (x, y, z)
                position = values[:3]
                # 四元数部分 (qx, qy, qz, qw)
                quaternion = values[3:]
            else:
                raise ValueError(f"Pose data format error: expected 7 values, got {len(values)}")
            
            # 将四元数转换为旋转矩阵
            rotation = Rotation.from_quat(quaternion).as_matrix()
            
            # 构建4x4变换矩阵
            c2w = np.eye(4)
            c2w[:3, :3] = rotation
            c2w[:3, 3] = position
            
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        
        with open(self.pose_path_right, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            # 解析4元组形式 (x, y, z, qx, qy, qz, qw)
            values = list(map(float, line.split()))
            if len(values) == 7:
                # 位置部分 (x, y, z)
                position = values[:3]
                # 四元数部分 (qx, qy, qz, qw)
                quaternion = values[3:]
            else:
                raise ValueError(f"Pose data format error: expected 7 values, got {len(values)}")
            
            # 将四元数转换为旋转矩阵
            rotation = Rotation.from_quat(quaternion).as_matrix()
            
            # 构建4x4变换矩阵
            c2w = np.eye(4)
            c2w[:3, :3] = rotation
            c2w[:3, 3] = position
            
            c2w = torch.from_numpy(c2w).float()
            poses_right.append(c2w)

        return poses, poses_right
    
    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)
    
