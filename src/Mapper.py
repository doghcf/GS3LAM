import os
import sys
import random
import cv2
import numpy as np
import torch

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from src.utils.sh_utils import RGB2SH
from src.utils.gaussian_utils import transform_to_frame, build_rotation
from src.Render import get_rasterizationSettings, transformed_params2rendervar
from gaussian_semantic_rasterization import GaussianRasterizer

sys.path.append(os.path.expanduser('/home/fu/GS3LAM/submodules/RAFT-Stereo'))
from types import SimpleNamespace
from argparse import Namespace
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder

def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective", random_select=False):
    """
    将彩色图像与深度图像转换为三维点云，并可选地进行坐标变换、高斯尺度估计（均方距离）、掩码筛选和随机采样
    """
    if len(color.shape) == 3:
        width, height = color.shape[2], color.shape[1]  # [C, H, W]格式
    else:
        height, width = color.shape  # [H, W]格式

    if not isinstance(depth, torch.Tensor):
        depth = torch.from_numpy(depth).cuda().float()
    elif not depth.is_cuda:
        depth = depth.cuda().float()

    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth.reshape(-1)

    # points in camera coordinates
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)

    # transform points to world coordinates if required
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    if len(color.shape) == 3 and color.shape[0] in [1, 3]:  # [C, H, W]格式
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    else:  # [H, W]格式
        cols = color.reshape(-1, 1).repeat(1, 3)  # 转换为3通道
    point_cld = torch.cat((pts, cols), -1)

    if random_select:
        num_pts = point_cld.shape[0]
        random_mask = random.sample(range(num_pts), int(0.005 * num_pts))
        # random_mask = random.sample(range(num_pts), 1024)
        point_cld = point_cld[random_mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[random_mask]

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution, num_objects=16):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    
    fused_objects = RGB2SH(torch.rand((num_pts, num_objects), device="cuda"))
    fused_objects = fused_objects[:,:,None]
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
        "obj_dc": fused_objects.transpose(1, 2)
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    # Every Frame pose initialization
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables

def remove_duplicate_points_single_array(point_data, mean3_sq_dist=None, threshold=0.001):
    """
    处理单个数组格式的点云去重
    假设数组结构为: [X, Y, Z, R, G, B, ...] (可能还有其他属性)
    """
    # 提取坐标和颜色
    points = point_data[:, :3]  # 前3列为坐标
    colors = point_data[:, 3:6]  # 接下来的3列为颜色
    
    # 创建体素网格进行去重
    voxel_size = threshold
    voxel_grid = {}
    
    for i, point in enumerate(points):
        voxel_idx = tuple(np.floor(point / voxel_size).astype(int))
        if voxel_idx not in voxel_grid:
            voxel_grid[voxel_idx] = []
        voxel_grid[voxel_idx].append(i)
    
    # 为每个体素保留一个点（取平均值）
    new_points = []
    new_colors = []
    new_mean3_sq_dist = []
    
    for voxel_idx, indices in voxel_grid.items():
        if indices:
            # 取体素内所有点的平均位置和颜色
            avg_point = np.mean(points[indices], axis=0)
            avg_color = np.mean(colors[indices], axis=0)
            new_points.append(avg_point)
            new_colors.append(avg_color)

            # 处理 mean3_sq_dist
            if mean3_sq_dist is not None:
                avg_mean3_sq_dist = torch.mean(mean3_sq_dist[indices], dim=0)
                new_mean3_sq_dist.append(avg_mean3_sq_dist)
    
    # 确保始终是二维数组
    if len(new_points) > 0:
        new_points = np.array(new_points)
        new_colors = np.array(new_colors)
    else:
        # 如果没有点，创建空的二维数组
        new_points = np.empty((0, 3))
        new_colors = np.empty((0, 3))
    
    # 重新组合数据
    result = np.concatenate([new_points, new_colors], axis=1)
    
    if mean3_sq_dist is not None:
        if len(new_mean3_sq_dist) > 0:
            new_mean3_sq_dist = torch.stack(new_mean3_sq_dist, dim=0)
        else:
            new_mean3_sq_dist = torch.empty((0,))
        return result, new_mean3_sq_dist
    else:
        return result

def estimate_disparity_confidence(disparity, left_image, right_image, kernel_size=5):
    """
    基于视差一致性和图像边缘的简单置信度估计
    """
    
    # 转换为numpy数组（如果是tensor）
    if isinstance(disparity, torch.Tensor):
        disparity_np = disparity.cpu().numpy()
    else:
        disparity_np = disparity
        
    if isinstance(left_image, torch.Tensor):
        left_img = left_image.cpu().numpy()
        if left_img.ndim == 4:
            left_img = left_img[0]  # 去除batch维度
        if left_img.shape[0] in [1, 3]:  # CHW格式
            left_img = np.transpose(left_img, (1, 2, 0))
    else:
        left_img = left_image
        
    # 转换为灰度图
    if left_img.ndim == 3 and left_img.shape[2] in [1, 3]:
        gray_img = cv2.cvtColor(left_img if left_img.shape[2] == 3 else 
                               np.repeat(left_img, 3, axis=2), cv2.COLOR_RGB2GRAY)
    else:
        gray_img = left_img
        
    # 1. 基于图像边缘的置信度（边缘区域置信度较低）
    edges = cv2.Canny((gray_img * 255).astype(np.uint8), 50, 150)
    edge_distances = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    edge_confidence = np.clip(edge_distances / 20.0, 0, 1)  # 归一化到[0,1]
    
    # 2. 基于局部视差一致性的置信度
    # 计算视差梯度
    grad_x = cv2.Sobel(disparity_np, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(disparity_np, cv2.CV_64F, 0, 1, ksize=3)
    disparity_gradients = np.sqrt(grad_x**2 + grad_y**2)
    
    # 梯度越小，置信度越高
    gradient_confidence = 1.0 - np.clip(disparity_gradients / np.percentile(disparity_gradients, 90), 0, 1)
    
    # 3. 组合置信度
    combined_confidence = edge_confidence * gradient_confidence
    
    # 转换回tensor并移到GPU
    confidence_tensor = torch.from_numpy(combined_confidence).float()
    if isinstance(disparity, torch.Tensor) and disparity.is_cuda:
        confidence_tensor = confidence_tensor.cuda()
        
    return confidence_tensor

def initialize_first_timestep(dataset, num_frames, mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None, num_objects=16):
    # Get Image Data & Camera Parameters
    color, color_right, _, intrinsics, pose, gt_objects = dataset[0]

    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    color_right = color_right.permute(2, 0, 1) / 255
    # depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

    def process_image(img):
        if len(img.shape) == 3 and img.shape[0] in [1, 3]:  # 已经是 [C, H, W] 格式
            img_np = img.permute(1, 2, 0).cpu().numpy()  # 转换为 [H, W, C] 以便处理
        else:
            img_np = img.cpu().numpy()
            
        # 如果是3通道图像，转换为灰度图
        if len(img_np.shape) == 3:
            if img_np.shape[2] in [1, 3]:
                img_np = img_np.mean(axis=2)
                
        # 模拟RGB三通道输入
        img_np = np.stack([img_np, img_np, img_np], axis=2)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return img_tensor[None].cuda()  # 添加batch维度并移到GPU

    # 处理左右图像
    color_processed = process_image(color)
    color_right_processed = process_image(color_right)
        
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = get_rasterizationSettings(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    args = Namespace(
        focal_length=320.0,
        baseline=0.25,
        valid_iters=7)
    model_args = SimpleNamespace(
        restore_ckpt="submodules/RAFT-Stereo/models/raftstereo-sceneflow.pth",
        corr_implementation = 'alt',
        shared_backbone = False,
        n_downsample = 2,
        n_gru_layers = 3,
        slow_fast_gru = False,
        hidden_dims = [128, 128, 128],
        context_dims = [128, 128, 128],
        corr_levels = 4,
        corr_radius = 4,
        num_frames = 2,
        query_feat_dim = 128,
        context_norm = "batch",
        dropout = 0.0,
        mixed_precision = True,
        alternate_corr = False
    )

    model = torch.nn.DataParallel(RAFTStereo(model_args))
    state_dict = torch.load(model_args.restore_ckpt)
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    left_image = color_processed
    right_image = color_right_processed

    with torch.no_grad():
        _, disparity = model(left_image, right_image, iters=args.valid_iters, test_mode=True)

    disparity = disparity.cpu().numpy()[0, 0]
    disparity = np.abs(disparity)
    confidence = estimate_disparity_confidence(disparity, left_image, right_image)
    print(f"Confidence range: {confidence.min()} to {confidence.max()}")
    # print(f"Disparity range: {disparity.min()} to {disparity.max()}")
    # print(f"Disparity mean: {disparity.mean()}")

    # 根据公式 depth = (f * B) / disparity 计算深度
    valid_disparity = np.where(disparity > 0.1, disparity, 0.1)
    depth = (args.focal_length * args.baseline) / valid_disparity
    # depth = 1.0 / depth
    # print(f"Depth range: {depth.min()} to {depth.max()}")
    # print(f"Depth mean: {depth.mean()}")

    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, intrinsics, w2c, 
                                            compute_mean_sq_dist=True, 
                                            mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution, num_objects)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    pt_cld_tensor = torch.from_numpy(init_pt_cld).cuda() if isinstance(init_pt_cld, np.ndarray) else init_pt_cld.clone().detach()
    max_depth_from_points = torch.max(pt_cld_tensor[:, 2])  # Z坐标即深度
    depth_tensor = torch.from_numpy(depth).cuda() if isinstance(depth, np.ndarray) else depth
    variables['scene_radius'] = torch.max(depth_tensor) / max_depth_from_points
    return params, variables, intrinsics, w2c, cam
    
def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution, num_objects=16):
    """
    初始化新点云中每个点对应的高斯参数
    """
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    
    # random init obj_id
    fused_objects = RGB2SH(torch.rand((num_pts, num_objects), device="cuda"))
    fused_objects = fused_objects[:,:,None]
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales':  log_scales,
        "obj_dc": fused_objects.transpose(1, 2)
    }
   
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians_alpha(params, variables, curr_data, densify_thres, time_idx, mean_sq_dist_method, gaussian_distribution, num_objects=16):
    # Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_pts)
    rendered_image, rendered_objects, radii, render_depth, rendered_alpha = GaussianRasterizer(raster_settings=curr_data["cam"])(**rendervar)

    # alpha mask
    non_presence_alpha_mask = (rendered_alpha < densify_thres)
    
    show_add_mask = False
    if show_add_mask:
        from PIL import Image
        os.makedirs("./logs/plots", exist_ok=True)
        masked_image_data = rendered_image * (~non_presence_alpha_mask)
        masked_image = Image.fromarray((masked_image_data.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8'))
        masked_image.save(f'./logs/plots/add_alpha_{time_idx}.png')

    # depth error mask
    gt_depth = curr_data['depth'][0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50 * depth_error.median())
    
    non_presence_mask = non_presence_alpha_mask | non_presence_depth_mask   # 逻辑或操作，得到非存在点的掩码，即需要补充新点的地方

    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
    curr_cam_tran = params['cam_trans'][..., time_idx].detach()
    curr_w2c = torch.eye(4).cuda().float()
    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
    curr_w2c[:3, 3] = curr_cam_tran

    # Get the new pointcloud in the world frame
    new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                mean_sq_dist_method=mean_sq_dist_method, random_select=False)
    
    # new_pt_cld_right, mean3_sq_dist_right = get_pointcloud(curr_data['im_right'], curr_data['depth_right'], curr_data['intrinsics'],
    #                             curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
    #                             mean_sq_dist_method=mean_sq_dist_method, random_select=False)
    # new_pt_cld = np.concatenate([new_pt_cld.cpu().numpy(), new_pt_cld_right.cpu().numpy()], axis=0)
    # mean3_sq_dist = torch.cat([mean3_sq_dist, mean3_sq_dist_right], dim=0)

    # new_pt_cld, mean3_sq_dist = remove_duplicate_points_single_array(new_pt_cld, mean3_sq_dist, threshold=0.005)

    new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution, num_objects)
    
    for k, v in new_params.items():
        params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))

    num_pts = params['means3D'].shape[0]
    variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
    variables['denom'] = torch.zeros(num_pts, device="cuda").float()
    variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
    new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
    variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    # ========== 新增：动态更新 scene_radius ==========
    # 获取当前帧的有效深度值（去除0值）
    valid_depths = curr_data['depth'][curr_data['depth'] > 0]
    
    if len(valid_depths) > 0:
        # 方法1：使用深度中值作为场景半径估计（对异常值更鲁棒）
        depth_median = torch.median(valid_depths)
        
        # 方法2：使用深度值的百分位数（避免极端值影响）
        depth_90_percentile = torch.quantile(valid_depths, 0.9)
        
        # 方法3：结合中值和范围进行动态调整
        depth_min = torch.min(valid_depths)
        depth_max = torch.max(valid_depths)
        
        # 动态计算场景半径，考虑深度范围的变化
        if depth_max > 100:  # 如果深度范围很大，使用对数缩放
            scene_radius = torch.log(depth_median + 1) * 2.0
        else:  # 正常范围，使用线性关系
            scene_radius = depth_median * 1.5
            
        # 限制场景半径的变化范围，避免剧烈波动
        current_radius = variables.get('scene_radius', torch.tensor(1.0).cuda())
        if 'scene_radius' in variables:
            # 平滑更新：新值 = 0.7 * 旧值 + 0.3 * 新计算值
            smoothed_radius = 0.7 * current_radius + 0.3 * scene_radius
            variables['scene_radius'] = smoothed_radius
        else:
            variables['scene_radius'] = scene_radius
            
        # 可选：打印调试信息
        if time_idx % 10 == 0:  # 每10帧打印一次
            print(f"Frame {time_idx}: Depth range [{depth_min:.2f}, {depth_max:.2f}], "
                  f"Scene radius: {variables['scene_radius']:.2f}")

    return params, variables