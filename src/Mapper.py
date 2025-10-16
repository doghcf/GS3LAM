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

def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective", random_select=False):
    """
    将彩色图像与深度图像转换为三维点云，并可选地进行坐标变换、高斯尺度估计（均方距离）、掩码筛选和随机采样
    """
    width, height = color.shape[2], color.shape[1]
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
    depth_z = depth[0].reshape(-1)

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
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
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

def generate_right_pointcloud_from_left(depth_left, color_right, intrinsics, T_left_to_right):
    """
    根据左目深度图和右目RGB生成右目点云
    """
    H, W = depth_left.shape[-2:]
    device = depth_left.device

    # 1. 构建像素坐标网格
    u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    ones = torch.ones_like(u)
    pixels = torch.stack((u, v, ones), dim=-1).reshape(-1, 3).T  # (3, N)

    # 2. 反投影到左目相机坐标系
    K_inv = torch.linalg.inv(intrinsics)
    z = depth_left.reshape(-1)
    pts_cam_left = (K_inv @ pixels) * z  # (3, N)

    # 3. 转换到右目坐标系
    pts_cam_left_h = torch.cat([pts_cam_left, torch.ones((1, pts_cam_left.shape[1]), device=device)], dim=0)
    pts_cam_right_h = T_left_to_right @ pts_cam_left_h
    pts_cam_right = pts_cam_right_h[:3, :].T  # (N, 3)

    # 4. 投影到右目像素坐标系
    proj = intrinsics @ pts_cam_right.T
    proj[:2, :] /= proj[2, :]  # 归一化除以z
    u_r = proj[0, :].reshape(H, W)
    v_r = proj[1, :].reshape(H, W)

    # 5. 从右目图像采样颜色（双线性插值）
    color_right = color_right.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    grid_x = 2 * (u_r / (W - 1)) - 1
    grid_y = 2 * (v_r / (H - 1)) - 1
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    sampled_color = torch.nn.functional.grid_sample(color_right, grid, align_corners=True)
    sampled_color = sampled_color.squeeze().permute(1, 2, 0)  # (H, W, C)

    # 6. 有效点掩码（在图像内且z>0）
    valid = (proj[2, :] > 0) & (u_r >= 0) & (u_r < W) & (v_r >= 0) & (v_r < H)
    pts_cam_right = pts_cam_right[valid]
    sampled_color = sampled_color.reshape(-1, 3)[valid]

    return pts_cam_right, sampled_color

def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None, num_objects=16):
    # Get RGB-D Data & Camera Parameters
    color, color_right, depth, depth_right, intrinsics, pose, pose_right, gt_objects, gt_objects_right = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    color_right = color_right.permute(2, 0, 1) / 255
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    if depth_right is not None:
        depth_right = depth_right.permute(2, 0, 1)
        
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)
    w2c_right = torch.linalg.inv(pose_right)

    # Setup Camera
    cam = get_rasterizationSettings(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    pt_cld_left, mean3_sq_dist = get_pointcloud(color, depth, intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    pt_cld_right = None
    if depth_right is not None:
        mask_right = (depth_right > 0)
        mask_right = mask_right.reshape(-1)
        pt_cld_right, mean3_sq_dist_right = get_pointcloud(color_right, depth_right, intrinsics, w2c_right,
                                        mask=mask_right, compute_mean_sq_dist=True,
                                        mean_sq_dist_method=mean_sq_dist_method)
    else:
        T_left_to_right = torch.linalg.inv(pose_right) @ pose
        pt_cld_right, color_right_pts = generate_right_pointcloud_from_left(
            depth.squeeze(0), color_right, intrinsics, T_left_to_right
        )

    init_pt_cld = np.concatenate([pt_cld_left.cpu().numpy(), pt_cld_right.cpu().numpy()], axis=0)
    mean3_sq_dist = torch.cat([mean3_sq_dist, mean3_sq_dist_right], dim=0)
    init_pt_cld, mean3_sq_dist = remove_duplicate_points_single_array(init_pt_cld, mean3_sq_dist, threshold=0.005)

    # Initialize Parameters
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution, num_objects)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    pt_cld_tensor = torch.tensor(init_pt_cld, device='cuda')
    max_depth_from_points = torch.max(pt_cld_tensor[:, 2])  # Z坐标即深度
    variables['scene_radius'] = torch.max(depth) / max_depth_from_points

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
    
    new_pt_cld_right, mean3_sq_dist_right = get_pointcloud(curr_data['im_right'], curr_data['depth_right'], curr_data['intrinsics'],
                                curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                mean_sq_dist_method=mean_sq_dist_method, random_select=False)
    new_pt_cld = np.concatenate([new_pt_cld.cpu().numpy(), new_pt_cld_right.cpu().numpy()], axis=0)
    mean3_sq_dist = torch.cat([mean3_sq_dist, mean3_sq_dist_right], dim=0)

    new_pt_cld, mean3_sq_dist = remove_duplicate_points_single_array(new_pt_cld, mean3_sq_dist, threshold=0.005)

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