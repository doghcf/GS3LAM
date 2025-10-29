import glob
import os
from typing import Optional

import numpy as np
import torch
from natsort import natsorted

from  src.datasets.basedataset import GradSLAMDataset

class EuRoCSemanticDataset(GradSLAMDataset):
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
        print("Load EuRoC dataset!!!")
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "mav0/state_groundtruth_estimate0/data.csv")
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
        color_paths = natsorted(glob.glob(f"{self.input_folder}/mav0/cam0/data/*.png"))
        color_paths_r = natsorted(glob.glob(f"{self.input_folder}/mav0/cam1/data/*.png"))
        object_paths = natsorted(glob.glob(f"{self.input_folder}/mav0/seg0/data/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, color_paths_r, object_paths, embedding_paths

    def load_poses(self):
        # 首先获取图像时间戳
        color_paths = natsorted(glob.glob(f"{self.input_folder}/mav0/cam0/data/*.png"))
        image_timestamps = []
        for path in color_paths:
            timestamp = int(os.path.basename(path).replace('.png', ''))
            image_timestamps.append(timestamp)
         
        # 读取所有位姿数据
        all_poses_data = []
        all_pose_timestamps = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()

        start_index = 1 if len(lines) > 0 and not lines[0].replace(',', '').replace('.', '').strip().isdigit() else 0
        for i in range(start_index, len(lines)):
            line = lines[i]
            data = line.strip().split(',')
            timestamp = int(data[0])  # 第一列是时间戳
            all_pose_timestamps.append(timestamp)
            all_poses_data.append(data)
        
        # 为每个图像时间戳找到最接近的位姿
        poses = []
        for img_ts in image_timestamps:
            # 找到最接近的位姿时间戳
            closest_idx = np.argmin(np.abs(np.array(all_pose_timestamps) - img_ts))
            data = all_poses_data[closest_idx]
            
            tx, ty, tz = map(float, data[1:4])
            qw, qx, qy, qz = map(float, data[4:8])
            R = self.quaternion_to_rotation_matrix(qw, qx, qy, qz)
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = [tx, ty, tz]
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        
        return poses

    def quaternion_to_rotation_matrix(self, w, x, y, z):
        """
        Convert a quaternion into a rotation matrix.
        """
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        return R

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)