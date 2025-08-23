import glob
import os
from typing import Optional
import numpy as np
import torch

from src.datasets.basedataset import GradSLAMDataset

class KITTISemanticDataset(GradSLAMDataset):
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
        load_stereo: Optional[bool] = True,
        **kwargs,
    ):
        print("Load KITTI dataset!!!")
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "cam0_to_world.txt")
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
            load_stereo=load_stereo,
            **kwargs,
        )

    def get_filepaths(self):
        left_color_paths = sorted(glob.glob(f"{self.input_folder}/image_00/data_rect/*.png"))
        right_color_paths = sorted(glob.glob(f"{self.input_folder}/image_01/data_rect/*.png"))
        object_paths = sorted(glob.glob(f"{self.input_folder}/object_mask_00/*.png"))
        
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = sorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        
        return left_color_paths, right_color_paths, object_paths, embedding_paths

    def load_poses(self):
        poses = [torch.eye(4) for _ in range(self.num_imgs)]
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            elems = line.split()
            idx = int(elems[0])
            c2w = np.array(list(map(float, elems[1:17]))).reshape(4, 4)
            c2w = torch.from_numpy(c2w).float()
            poses[idx] = c2w
        
        return poses