import torch
import torchvision
import os

from src.utils.gaussian_utils import transform_to_frame
from src.Render import transformed_params2rendervar
from src.utils.metric_utils import calc_ssim, l1_loss_v1
from gaussian_semantic_rasterization import GaussianRasterizer

def initialize_optimizer(params, lrs_dict, tracking):
    param_groups = [{'params': [v], 'name': k, 'lr': lrs_dict[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
    

def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, 
             use_l1, ignore_outlier_depth_loss, tracking=False,
             mapping=False, do_ba=False, use_reg_loss=False,
             semantic_decoder=None,
             use_semantic_for_tracking=True,
             use_semantic_for_mapping=True,
             use_alpha_for_loss=False,
             alpha_thres=0.99,
             num_classes=256):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    
    # Rendering
    rendervar['means2D'].retain_grad()
    rendered_image, rendered_objects, radii, rendered_depth, rendered_alpha = GaussianRasterizer(raster_settings=curr_data["cam"])(**rendervar)

    variables['means2D'] = rendervar['means2D']  # Gradient only accum from color render for densification

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(rendered_depth))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - rendered_depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10 * depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask

    if tracking and use_alpha_for_loss:
        presence_alpha_mask = (rendered_alpha > alpha_thres)
        mask = mask & presence_alpha_mask

    # 计算观测区域掩码 (对应论文中的 M_obs 和 M_unobs)
    if tracking:
        # 对于跟踪: 使用可见性掩码和立体一致性掩码
        observable_mask = (rendered_alpha > 0.5)  # 简化的可见性掩码
        if ignore_outlier_depth_loss:
            stereo_consistency_error = torch.abs(curr_data['depth'] - rendered_depth)
            stereo_mask = (stereo_consistency_error < stereo_consistency_error.median() * 2)
            observable_mask = observable_mask & stereo_mask
        mask = mask & observable_mask
    else:
        # 对于映射: 使用不可观测区域掩码
        opacity_threshold = 0.5
        unobservable_by_opacity = (rendered_alpha < opacity_threshold)
        
        depth_error = torch.abs(rendered_depth - curr_data['depth'])
        median_depth_error = torch.median(depth_error[depth_error > 0]) if torch.sum(depth_error > 0) > 0 else 1.0
        unobservable_by_depth = (depth_error > 50 * median_depth_error)
        
        unobservable_mask = unobservable_by_opacity | unobservable_by_depth
        mask = mask & (~unobservable_mask)  # 只保留可观测区域

    # Color/Photometric Loss (L_color)
    if tracking:
        if ignore_outlier_depth_loss:
            color_mask = torch.tile(mask, (3, 1, 1))
            color_mask = color_mask.detach()
            losses['color'] = torch.abs(curr_data['im'] - rendered_image)[color_mask].sum()
        else:
            losses['color'] = torch.abs(curr_data['im'] - rendered_image).sum()
    else:
        if rendered_image.dim() == 5:
            rendered_image = rendered_image.squeeze(1)
            curr_data['im'] = curr_data['im'].squeeze(1)
        elif rendered_image.dim() == 2:
            rendered_image = rendered_image.unsqueeze(0).unsqueeze(0)
            curr_data['im'] = curr_data['im'].unsqueeze(0).unsqueeze(0)
        elif rendered_image.dim() == 3:
            rendered_image = rendered_image.unsqueeze(0)
            curr_data['im'] = curr_data['im'].unsqueeze(0)

        # 确保两个图像具有相同的通道数
        if rendered_image.shape[-3] != curr_data['im'].shape[-3]:
            min_channels = min(rendered_image.shape[-3], curr_data['im'].shape[-3])
            if rendered_image.shape[-3] > min_channels:
                rendered_image = rendered_image[:, :min_channels, :, :]
            if curr_data['im'].shape[-3] > min_channels:
                curr_data['im'] = curr_data['im'][:, :min_channels, :, :]
        
        # 结合L1和SSIM损失 (论文中的 photometric loss)
        l1_component = l1_loss_v1(rendered_image, curr_data['im'])
        ssim_component = 1.0 - calc_ssim(rendered_image, curr_data['im'])
        losses['color'] = 0.8 * l1_component + 0.2 * ssim_component
    
    # Geometric Loss with Uncertainty (L_geo)
    # 实现论文中的公式: L_geo = L1(Î\hat{D}, D) / Ï\sigma_{depth}^2 + log(Ï\sigma_{depth}^2)
    if 'depth_uncertainty' in curr_data:
        # 如果提供了深度不确定性
        depth_uncertainty = curr_data['depth_uncertainty']
        depth_error = torch.abs(curr_data['depth'] - rendered_depth)
        # 论文公式: L_geo = L1(Î\hat{D}, D) / Ï\sigma_{depth}^2 + log(Ï\sigma_{depth}^2)
        uncertainty_weighted_error = (depth_error / (depth_uncertainty ** 2 + 1e-6))[mask].mean()
        uncertainty_regularization = torch.log(depth_uncertainty ** 2 + 1e-6)[mask].mean()
        losses['geo'] = uncertainty_weighted_error + uncertainty_regularization
    else:
        # 简化版本，只使用L1损失
        if tracking:
            losses['geo'] = torch.abs(curr_data['depth'] - rendered_depth)[mask].sum()
        else:
            losses['geo'] = torch.abs(curr_data['depth'] - rendered_depth)[mask].mean()
    
    # Semantic Loss (L_sem)
    if semantic_decoder is not None:
        gt_obj = curr_data["obj"].long()
        logits = semantic_decoder(rendered_objects) # type: ignore
        cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        if tracking and use_semantic_for_tracking:
            if ignore_outlier_depth_loss:
                obj_mask = mask.detach().squeeze(0) if mask.dim() > 2 else mask
                loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze()
                if obj_mask.dim() == 0:
                    obj_mask = obj_mask.unsqueeze(0)
                loss_obj = loss_obj[obj_mask].sum() if loss_obj.numel() > 1 else loss_obj.sum()
            else:
                loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().sum()
            losses['sem'] = loss_obj / torch.log(torch.tensor(num_classes))
            
        elif mapping and use_semantic_for_mapping:
            loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
            losses['sem'] = loss_obj / torch.log(torch.tensor(num_classes))

    # Stereo Loss (仅用于跟踪)
    if tracking and 'stereo_im' in curr_data and 'stereo_cam' in curr_data:
        # 实现立体匹配损失 L_stereo = L1(Î\hat{C}_{left} - W(Î\hat{C}_{right}, Î\hat{D}_{left}))
        # 这里简化实现，实际需要根据视差进行图像变换
        if 'stereo_rendered_image' in locals():
            stereo_loss = torch.abs(rendered_image - stereo_rendered_image).sum()
            losses['stereo'] = stereo_loss

    # Regularization Losses
    if mapping and use_reg_loss:
        scaling = torch.exp(params['log_scales'])
        mean_scale = scaling.mean()
        std_scale = scaling.std()
        upper_limit = mean_scale + 2 * std_scale
        lower_limit = mean_scale - 2 * std_scale
        
        # regularize very big Gaussian
        if upper_limit < scaling.max():
            losses["big_gaussian_reg"] = torch.mean(scaling[torch.where(scaling > upper_limit)])
        else:
            losses["big_gaussian_reg"] = 0.0
        # regularize very small Gaussian
        if lower_limit > scaling.min():
            losses["small_gaussian_reg"] = torch.mean(-torch.log(scaling[torch.where(scaling < lower_limit)]))
        else:
            losses["small_gaussian_reg"] = 0.0

    # 应用权重并计算总损失
    weighted_losses = {}
    
    # 根据是跟踪还是映射应用不同的损失函数
    if tracking:
        # L_tracking = M_obs (λ_c^t L_color^t + λ_g^t L_geo^t + λ_s^t L_sem^t)
        # 损失已经在前面使用了正确的掩码(M_obs)，这里只需要应用权重
        for k, v in losses.items():
            weight_key = f"{k}_tracking" if f"{k}_tracking" in loss_weights else k
            if weight_key in loss_weights:
                weighted_losses[k] = v * loss_weights[weight_key]
    else:
        # L_mapping = M_unobs (λ_c^m L_color^t + λ_g^m L_geo^t + λ_s^m L_sem^t)
        # 损失已经在前面使用了正确的掩码(M_unobs)，这里只需要应用权重
        for k, v in losses.items():
            weight_key = f"{k}_mapping" if f"{k}_mapping" in loss_weights else k
            if weight_key in loss_weights:
                weighted_losses[k] = v * loss_weights[weight_key]

    loss = sum(weighted_losses.values())

    seen = radii > 0
    variables['max_2D_radius'][seen] = torch.max(radii[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses