import glob
import os
from typing import Optional

import numpy as np
import torch
from natsort import natsorted
from scipy.spatial.transform import Rotation

from src.datasets.basedataset import GradSLAMDataset

class KITTI360SemanticDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 376,
        desired_width: Optional[int] = 1408,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        print("Load kitti-360 dataset!!!")
        self.input_folder = basedir
        self.sequence = sequence
        self.pose_path = os.path.join(self.input_folder, "data_2d_raw", self.sequence, "cam0_to_world.txt")
        self.pose_path_right = os.path.join(self.input_folder, "data_2d_raw", self.sequence, "cam1_to_world.txt")
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
        color_paths = natsorted(glob.glob(f"{self.input_folder}/data_2d_raw/{self.sequence}/image_00/data_rect/*.png"))
        color_paths_right = natsorted(glob.glob(f"{self.input_folder}/data_2d_raw/{self.sequence}/image_01/data_rect/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/data_2d_depth/{self.sequence}/image_00/data_depth/*.npy"))
        depth_paths_right = natsorted(glob.glob(f"{self.input_folder}/data_2d_depth/{self.sequence}/image_00/data_depth/*.npy"))
        object_paths = natsorted(glob.glob(f"{self.input_folder}/data_2d_semantics/{self.sequence}/image_00/semantic/*.png"))
        object_paths_right = natsorted(glob.glob(f"{self.input_folder}/data_2d_semantics/{self.sequence}/image_01/semantic/*.png"))

        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, color_paths_right, depth_paths, depth_paths_right, object_paths, object_paths_right, embedding_paths
    
    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            values = list(map(float, line.split()[1:]))  # 跳过帧号
            if len(values) == 16:
                c2w = np.array(values).reshape(4, 4)
            elif len(values) == 12:
                c2w = np.eye(4)
                c2w[:3, :4] = np.array(values).reshape(3, 4)
            else:
                raise ValueError(f"Pose data format error: expected 16 values, got {len(values)}")
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)

        poses_right = []
        with open(self.pose_path_right, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            values = list(map(float, line.split()[1:]))  # 跳过帧号
            if len(values) == 16:
                c2w = np.array(values).reshape(4, 4)
            elif len(values) == 12:
                c2w = np.eye(4)
                c2w[:3, :4] = np.array(values).reshape(3, 4)
            else:
                raise ValueError(f"Pose data format error: expected 16 values, got {len(values)}")
            c2w = torch.from_numpy(c2w).float()
            poses_right.append(c2w)

        return poses, poses_right
    
    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)
    
