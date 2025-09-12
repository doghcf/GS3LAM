import cv2
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from sklearn.decomposition import PCA
import copy

from src.Render import get_rasterizationSettings, transformed_params2rendervar
from src.utils.gaussian_utils import build_rotation, transform_to_frame
from src.utils.metric_utils import calc_psnr, evaluate_ate


from gaussian_semantic_rasterization import GaussianRasterizer

from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()


def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask

def eval(dataset, final_params, num_frames, eval_dir,
         mapping_iters, add_new_gaussians, wandb_run=None, wandb_save_qual=False, eval_every=1, save_frames=False, use_semantic=False, load_stereo=False, classifier=None):
    print("Evaluating Final Parameters ...")
    # Initialize metric lists
    psnr_list, rmse_list, l1_list, fps_list, lpips_list, ssim_list = [], [], [], [], [], []
    gt_w2c_list = []
    
    # Setup directories
    plot_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    save_frames=True
    if save_frames:
        if load_stereo:
            render_rgb_dir = os.path.join(eval_dir, "left_rendered_rgb")
            render_depth_dir = os.path.join(eval_dir, "left_rendered_depth")
            right_render_rgb_dir = os.path.join(eval_dir, "right_rendered_rgb")
            rgb_dir = os.path.join(eval_dir, "left_rgb")
            depth_dir = os.path.join(eval_dir, "left_depth")
            right_rgb_dir = os.path.join(eval_dir, "right_rgb")
            os.makedirs(render_rgb_dir, exist_ok=True)
            os.makedirs(render_depth_dir, exist_ok=True)
            os.makedirs(right_render_rgb_dir, exist_ok=True)
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(right_rgb_dir, exist_ok=True)
        else:
            render_rgb_dir = os.path.join(eval_dir, "rendered_rgb")
            render_depth_dir = os.path.join(eval_dir, "rendered_depth")
            rgb_dir = os.path.join(eval_dir, "rgb")
            depth_dir = os.path.join(eval_dir, "depth")
            os.makedirs(render_rgb_dir, exist_ok=True)
            os.makedirs(render_depth_dir, exist_ok=True)
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            
        if use_semantic:
            object_dir = os.path.join(eval_dir, "object_mask")
            os.makedirs(object_dir, exist_ok=True)
            render_object_dir = os.path.join(eval_dir, "rendered_object")
            os.makedirs(render_object_dir, exist_ok=True)
            objects_feature16_dir = os.path.join(eval_dir, "objects_feature16")
            os.makedirs(objects_feature16_dir, exist_ok=True)
            # mIoU
            gt_mask_array_path = os.path.join(eval_dir, "gt_mask_array")
            os.makedirs(gt_mask_array_path, exist_ok=True)
            pred_mask_array_path = os.path.join(eval_dir, "pred_mask_array")
            os.makedirs(objects_feature16_dir, exist_ok=True)
            os.makedirs(gt_mask_array_path, exist_ok=True)
            os.makedirs(pred_mask_array_path, exist_ok=True)
    
    # Initialize timing events
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    
    for time_idx in tqdm(range(num_frames)):
        # Get data for current frame
        if load_stereo:
            if use_semantic:
                left_color, right_color, left_depth, right_depth, intrinsics, pose, gt_objects, gt_objects_right = dataset[time_idx]
            else:
                left_color, right_color, left_depth, right_depth, intrinsics, pose = dataset[time_idx]
        else:
            if use_semantic:
                color, depth, intrinsics, pose, gt_objects = dataset[time_idx]
            else:
                color, depth, intrinsics, pose = dataset[time_idx]
            
        gt_w2c = torch.linalg.inv(pose)
        gt_w2c_list.append(gt_w2c)
        intrinsics = intrinsics[:3, :3]
        
        # Process color and depth
        if load_stereo:
            left_color = left_color.permute(2, 0, 1) / 255
            right_color = right_color.permute(2, 0, 1) / 255
            left_depth = left_depth.permute(2, 0, 1)
            right_depth = right_depth.permute(2, 0, 1)
        else:
            color = color.permute(2, 0, 1) / 255
            depth = depth.permute(2, 0, 1)
        
        if time_idx == 0:
            first_frame_w2c = torch.linalg.inv(pose)
            if load_stereo:
                # For stereo, we'll use the left camera as reference
                cam = get_rasterizationSettings(left_color.shape[2], left_color.shape[1], 
                                               intrinsics.cpu().numpy(), 
                                               first_frame_w2c.detach().cpu().numpy())
            else:
                cam = get_rasterizationSettings(color.shape[2], color.shape[1], 
                                               intrinsics.cpu().numpy(), 
                                               first_frame_w2c.detach().cpu().numpy())
        
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue
            
        # Prepare current data dictionary
        if load_stereo:
            curr_data = {
                'cam': cam,
                'im': left_color,
                'im_right': right_color,
                'depth': left_depth,
                'depth_right': right_depth,
                'id': time_idx,
                'intrinsics': intrinsics,
                'w2c': first_frame_w2c
            }
            if use_semantic:
                curr_data['obj'] = gt_objects
                curr_data['obj_right'] = gt_objects_right
        else:
            curr_data = {
                'cam': cam,
                'im': color,
                'depth': depth,
                'id': time_idx,
                'intrinsics': intrinsics,
                'w2c': first_frame_w2c
            }
            if use_semantic:
                curr_data['obj'] = gt_objects
        
        # Transform Gaussians and render
        transformed_gaussians = transform_to_frame(final_params, time_idx, 
                                                  gaussians_grad=False,
                                                  camera_grad=False)
        rendervar = transformed_params2rendervar(final_params, transformed_gaussians)
        
        # Time the rendering
        iter_start.record()
        im, rendered_objects, radius, rendered_depth, rendered_alpha = \
            GaussianRasterizer(raster_settings=curr_data["cam"])(**rendervar)
        iter_end.record()
        
        torch.cuda.synchronize()
        iter_time = iter_start.elapsed_time(iter_end) / 1000.0
        fps_list.append(1.0 / iter_time)
        
        # Compute metrics
        valid_depth_mask = (curr_data['depth'] > 0)
        rastered_depth_viz = rendered_depth.detach()
        weighted_im = im * valid_depth_mask
        weighted_gt_im = curr_data['im'] * valid_depth_mask
        
        psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
        ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                        data_range=1.0, size_average=True)
        lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                    torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()
        
        # Depth metrics
        diff_depth = rendered_depth - curr_data['depth']
        if mapping_iters == 0 and not add_new_gaussians:
            rmse = torch.sqrt((diff_depth ** 2) * valid_depth_mask).sum() / valid_depth_mask.sum()
            l1 = (torch.abs(diff_depth) * valid_depth_mask).sum() / valid_depth_mask.sum()
        else:
            rmse = torch.sqrt((diff_depth ** 2) * valid_depth_mask).sum() / valid_depth_mask.sum()
            l1 = (torch.abs(diff_depth) * valid_depth_mask).sum() / valid_depth_mask.sum()
        
        psnr_list.append(psnr.cpu().numpy() if hasattr(psnr, 'cpu') else psnr)
        ssim_list.append(ssim.cpu().numpy() if hasattr(ssim, 'cpu') else ssim)
        lpips_list.append(lpips_score)
        rmse_list.append(rmse.cpu().numpy() if hasattr(rmse, 'cpu') else rmse)
        l1_list.append(l1.cpu().numpy() if hasattr(l1, 'cpu') else l1)
        
        # Handle semantic segmentation
        if use_semantic:
            
            rendered_objects_for_semantic = rendered_objects
                
            logits = classifier(rendered_objects_for_semantic)
            pred_obj = torch.argmax(logits, dim=0)
            
            # Save masks for mIoU calculation
            gt_mask_array = curr_data['obj'].cpu().numpy().astype(np.uint8)
            pred_mask_array = pred_obj.cpu().numpy().astype(np.uint8)
            view_prefix = "left_" if load_stereo else ""
            np.save(os.path.join(gt_mask_array_path, f"{view_prefix}gt_{time_idx:04d}.npy"), gt_mask_array)
            np.save(os.path.join(pred_mask_array_path, f"{view_prefix}pred_{time_idx:04d}.npy"), pred_mask_array)
            
            # Visualize and save masks
            pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
            gt_rgb_mask = visualize_obj(curr_data['obj'].cpu().numpy().astype(np.uint8))
            rgb_mask = feature_to_rgb(rendered_objects_for_semantic)
            
            cv2.imwrite(os.path.join(object_dir, f"gt_{time_idx:04d}.png"), gt_rgb_mask)
            cv2.imwrite(os.path.join(render_object_dir, f"gs_{time_idx:04d}.png"), pred_obj_mask)
            cv2.imwrite(os.path.join(objects_feature16_dir, f"{view_prefix}{time_idx:04d}.png"), rgb_mask)
        
        # Save frames if required
        if save_frames:
            # Save Rendered RGB and Depth
            viz_render_im = torch.clamp(im, 0, 1)
            viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
            vmin, vmax = 0, 6
            viz_render_depth = rendered_depth[0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_render_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(render_rgb_dir, f"gs_{time_idx:04d}.png"), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(render_depth_dir, f"gs_{time_idx:04d}.png"), depth_colormap)
            
            # Save GT RGB and Depth
            viz_gt_im = torch.clamp(curr_data['im'], 0, 1)
            viz_gt_im = viz_gt_im.detach().cpu().permute(1, 2, 0).numpy()
            viz_gt_depth = curr_data['depth'][0].detach().cpu().numpy()
            normalized_depth = np.clip((viz_gt_depth - vmin) / (vmax - vmin), 0, 1)
            depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(rgb_dir, f"gt_{time_idx:04d}.png"), cv2.cvtColor(viz_gt_im*255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(depth_dir, f"gt_{time_idx:04d}.png"), depth_colormap)
    
    # Compute ATE RMSE
    try:
        num_frames = final_params['cam_unnorm_rots'].shape[-1]
        latest_est_w2c = first_frame_w2c
        latest_est_w2c_list = [latest_est_w2c]
        valid_gt_w2c_list = [gt_w2c_list[0]]
        
        for idx in range(1, num_frames):
            if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                continue
            interm_cam_rot = F.normalize(final_params['cam_unnorm_rots'][..., idx].detach())
            interm_cam_trans = final_params['cam_trans'][..., idx].detach()
            intermrel_w2c = torch.eye(4).cuda().float()
            intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
            intermrel_w2c[:3, 3] = interm_cam_trans
            latest_est_w2c = intermrel_w2c
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[idx])
            
        ate_rmse = evaluate_ate(valid_gt_w2c_list, latest_est_w2c_list)
        ate_status = "Success"
        print(f"Final Average ATE RMSE: {ate_rmse*100:.2f} cm")
    except Exception as e:
        ate_rmse = 100.0
        ate_status = "Failed"
        print(f'Failed to evaluate trajectory with alignment: {e}')
    
    # Calculate average metrics
    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    fps_list = np.array(fps_list)
    
    avg_psnr = psnr_list.mean()
    avg_rmse = rmse_list.mean()
    avg_l1 = l1_list.mean()
    avg_ssim = ssim_list.mean()
    avg_lpips = lpips_list.mean()
    avg_fps = fps_list.mean()
    
    print(f"Average PSNR: {avg_psnr:.3f}")
    print(f"Average Depth RMSE: {avg_rmse*100:.3f} cm")
    print(f"Average Depth L1: {avg_l1*100:.3f} cm")
    print(f"Average MS-SSIM: {avg_ssim:.3f}")
    print(f"Average LPIPS: {avg_lpips:.3f}")
    
    # Prepare summary metrics
    metric_summary = [
        f"Rendering FPS: {avg_fps:.5f}",
        f"Final Average ATE RMSE: {ate_rmse*100:.2f} cm" if ate_status == "Success" else "Failed to evaluate trajectory with alignment.",
        f"Average PSNR: {avg_psnr:.3f}",
        f"Average Depth RMSE: {avg_rmse*100:.3f} cm",
        f"Average Depth L1: {avg_l1*100:.3f} cm",
        f"Average MS-SSIM: {avg_ssim:.3f}",
        f"Average LPIPS: {avg_lpips:.3f}"
    ]
    
    # Save metric lists
    np.savetxt(os.path.join(eval_dir, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_dir, "rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_dir, "l1.txt"), l1_list)
    np.savetxt(os.path.join(eval_dir, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(eval_dir, "lpips.txt"), lpips_list)
    np.savetxt(os.path.join(eval_dir, "avg_metric.txt"), metric_summary, fmt='%s')
    
    # Create and save plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(np.arange(len(psnr_list)), psnr_list)
    axs[0].set_title("RGB PSNR")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("PSNR")
    
    axs[1].plot(np.arange(len(l1_list)), np.array(l1_list)*100)
    axs[1].set_title("Depth L1")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("L1 (cm)")
    
    title = f"Average PSNR: {avg_psnr:.2f}, Average Depth L1: {avg_l1*100:.2f} cm, ATE RMSE: {ate_rmse*100:.2f} cm"
    fig.suptitle(title, y=1.05, fontsize=16)
    plt.savefig(os.path.join(eval_dir, "metrics.png"), bbox_inches='tight')
    
    if wandb_run is not None:
        wandb_run.log({"Eval/Metrics": fig})
    plt.close()