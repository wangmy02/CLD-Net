from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


def bilateral_filter_depth(depth_map, d=5, sigma_color=50, sigma_space=50):
    """
    对深度图应用双边滤波（保边平滑）
    
    注意：OpenCV 的 bilateralFilter 仅支持 8-bit 或 32-bit float，
    这里直接在 float32 深度图上操作。
    
    Args:
        depth_map: [H, W] 深度图（float32）
        d: 滤波器直径
        sigma_color: 颜色空间标准差
        sigma_space: 坐标空间标准差
    
    Returns:
        filtered_depth: [H, W] 滤波后的深度图（float32）
    """
    # 确保是 float32
    depth_f32 = depth_map.astype(np.float32)
    
    # 双边滤波（直接在 float32 上进行）
    filtered = cv2.bilateralFilter(depth_f32, d, sigma_color, sigma_space)
    
    return filtered.astype(np.float32)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    abs_diff=np.mean(np.abs(gt - pred))
    
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_diff,abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    # Set depth range based on dataset
    # Both SCARED and Hamlyn use millimeters as the unit (no conversion)
    MIN_DEPTH = 1e-3   # 0.001mm (to handle near-zero values)
    MAX_DEPTH = 150    # 150mm
    
    if opt.eval_split == "hamlyn":
        MAX_DEPTH = 300  # Hamlyn: 1-300mm valid range
        print(f"-> Using Hamlyn depth range: [{MIN_DEPTH}, {MAX_DEPTH}] mm")
    else:
        print(f"-> Using SCARED depth range: [{MIN_DEPTH}, {MAX_DEPTH}] mm")

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)
        
        # Load filenames only for datasets that need them (e.g., SCARED)
        if opt.eval_split == "endovis":
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
            dataset = datasets.SCAREDRAWDataset(opt.data_path, filenames,
                                            encoder_dict['height'], encoder_dict['width'],
                                            [0], 4, is_train=False)
            dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                    pin_memory=True, drop_last=False)
        elif opt.eval_split == "hamlyn":
            filenames = None  # Hamlyn doesn't use test_files.txt
            dataset = datasets.HamlynDataset(opt.data_path,
                                            encoder_dict['height'], encoder_dict['width'],
                                            [0], 4, is_train=False)
            dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                    pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        
        # Load decoder weights first to check the type
        decoder_dict = torch.load(decoder_path)
        
        # Check if it's IntrinsicGuidedDepthDecoder or standard DepthDecoder
        is_intrinsic_guided = any('intrinsic_encoder' in key or 'refinement_blocks' in key 
                                   for key in decoder_dict.keys())
        
        if is_intrinsic_guided:
            print("-> Detected IntrinsicGuidedDepthDecoder (Stage A model)")
            depth_decoder = networks.IntrinsicGuidedDepthDecoder(encoder.num_ch_enc, scales=range(4))
        else:
            print("-> Detected standard DepthDecoder (Baseline model)")
            depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(decoder_dict)

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()
        
        # Load decompose network if using IntrinsicGuidedDepthDecoder
        if is_intrinsic_guided:
            decompose_encoder_path = os.path.join(opt.load_weights_folder, "decompose_encoder.pth")
            decompose_path = os.path.join(opt.load_weights_folder, "decompose.pth")
            
            if os.path.exists(decompose_encoder_path) and os.path.exists(decompose_path):
                print("-> Loading decompose network for intrinsic features")
                decompose_encoder = networks.ResnetEncoder(opt.num_layers, False)
                decompose_decoder = networks.decompose_decoder(decompose_encoder.num_ch_enc, 
                                                               scales=range(4))
                
                decompose_encoder_dict = torch.load(decompose_encoder_path)
                decompose_encoder.load_state_dict({k: v for k, v in decompose_encoder_dict.items() 
                                                   if k in decompose_encoder.state_dict()})
                decompose_decoder.load_state_dict(torch.load(decompose_path))
                
                decompose_encoder.cuda()
                decompose_encoder.eval()
                decompose_decoder.cuda()
                decompose_decoder.eval()
            else:
                print("⚠️  Warning: Decompose network not found, using dummy intrinsic features")
                decompose_encoder = None
                decompose_decoder = None
        else:
            decompose_encoder = None
            decompose_decoder = None

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                # === TTA: Test-Time Augmentation ===
                use_tta = getattr(opt, 'use_tta', False)
                
                if use_tta:
                    # TTA: 原图 + 水平翻转
                    pred_disps_tta = []
                    
                    # 1. 原图预测
                    features = encoder(input_color)
                    if is_intrinsic_guided:
                        if decompose_encoder is not None and decompose_decoder is not None:
                            decompose_features = decompose_encoder(input_color)
                            reflectance, shading = decompose_decoder(decompose_features)
                        else:
                            B, _, H, W = input_color.shape
                            reflectance = torch.ones(B, 3, H, W).cuda()
                            shading = torch.ones(B, 1, H, W).cuda()
                        output = depth_decoder(features, reflectance, shading)
                    else:
                        output = depth_decoder(features)
                    
                    pred_disp1, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                    pred_disps_tta.append(pred_disp1)
                    
                    # 2. 水平翻转预测
                    input_flipped = torch.flip(input_color, [3])
                    features_flip = encoder(input_flipped)
                    if is_intrinsic_guided:
                        if decompose_encoder is not None and decompose_decoder is not None:
                            decompose_features_flip = decompose_encoder(input_flipped)
                            reflectance_flip, shading_flip = decompose_decoder(decompose_features_flip)
                        else:
                            reflectance_flip = torch.ones_like(reflectance)
                            shading_flip = torch.ones_like(shading)
                        output_flip = depth_decoder(features_flip, reflectance_flip, shading_flip)
                    else:
                        output_flip = depth_decoder(features_flip)
                    
                    pred_disp2, _ = disp_to_depth(output_flip[("disp", 0)], opt.min_depth, opt.max_depth)
                    pred_disp2 = torch.flip(pred_disp2, [3])  # 翻转回来
                    pred_disps_tta.append(pred_disp2)
                    
                    # 平均 TTA 结果
                    pred_disp = torch.stack(pred_disps_tta, dim=0).mean(dim=0)
                    pred_disp = pred_disp.cpu()[:, 0].numpy()
                
                else:
                    # 原始评估流程
                    if opt.post_process:
                        # Post-processed results require each image to have two forward passes
                        input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                    # Get depth features
                    features = encoder(input_color)
                    
                    # For IntrinsicGuidedDepthDecoder, we need reflectance and shading
                    if is_intrinsic_guided:
                        if decompose_encoder is not None and decompose_decoder is not None:
                            # Get intrinsic decomposition
                            decompose_features = decompose_encoder(input_color)
                            # decompose_decoder returns (reflectance, shading) directly
                            reflectance, shading = decompose_decoder(decompose_features)
                        else:
                            # Use dummy features if decompose not available
                            B, _, H, W = input_color.shape
                            reflectance = torch.ones(B, 3, H, W).cuda()
                            shading = torch.ones(B, 1, H, W).cuda()
                        
                        output = depth_decoder(features, reflectance, shading)
                    else:
                        output = depth_decoder(features)
                    
                    pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                    pred_disp = pred_disp.cpu()[:, 0].numpy()

                    if opt.post_process:
                        N = pred_disp.shape[0] // 2
                        pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    # Load GT depths
    if opt.eval_split == "hamlyn":
        # Check if npz file exists for Hamlyn
        gt_npz_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        if os.path.exists(gt_npz_path):
            # Use npz file (faster and more memory efficient for subset)
            print(f"-> Loading Hamlyn GT depths from npz: {gt_npz_path}")
            gt_depths = np.load(gt_npz_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
            hamlyn_depth_info = None
            print(f"-> Loaded {len(gt_depths)} GT depth maps from npz")
        else:
            # Fall back to on-the-fly loading (for full dataset or if npz not available)
            print("-> Hamlyn npz not found, using on-the-fly GT loading from dataset")
            gt_depths = None
            hamlyn_depth_info = [(scan["depth01"], scan["sequence"]) for scan in dataset.scans]
            print(f"-> Prepared {len(hamlyn_depth_info)} GT depth paths from Hamlyn dataset")
    else:
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
        hamlyn_depth_info = None

    # Limit evaluation samples if specified
    if opt.max_eval_samples is not None:
        num_samples = min(opt.max_eval_samples, pred_disps.shape[0])
        print(f"-> Limiting evaluation to first {num_samples} samples (out of {pred_disps.shape[0]})")
        pred_disps = pred_disps[:num_samples]
        if opt.eval_split == "hamlyn" and hamlyn_depth_info is not None:
            # On-the-fly loading mode
            hamlyn_depth_info = hamlyn_depth_info[:num_samples]
        else:
            # npz loading mode (SCARED or Hamlyn with npz)
            gt_depths = gt_depths[:num_samples]

    print("-> Mono evaluation - using median scaling")
    
    # 检查是否使用后处理
    use_bilateral_filter = getattr(opt, 'use_bilateral_filter', False)
    if use_bilateral_filter:
        print("-> Using bilateral filter for depth post-processing")

    errors = []
    ratios = []

    save_dir = os.path.join(opt.load_weights_folder, "depth_predictions")
    if opt.max_save_samples is not None:
        num_to_save = min(opt.max_save_samples, pred_disps.shape[0])
        print("-> Saving first {} predicted depth maps to {} (out of {} total)".format(
            num_to_save, save_dir, pred_disps.shape[0]))
    else:
        print("-> Saving out benchmark predictions to {}".format(save_dir))
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # Get filenames for saving predictions with proper names
    # Try to get filenames if not already defined (e.g., when ext_disp_to_eval is used)
    try:
        # Check if filenames exists in current scope
        _ = filenames
    except NameError:
        # filenames not defined, try to load it
        test_files_path = os.path.join(splits_dir, opt.eval_split, "test_files.txt")
        if os.path.exists(test_files_path):
            try:
                filenames = readlines(test_files_path)
            except:
                filenames = None
        else:
            filenames = None

    for i in range(pred_disps.shape[0]):

        # Load GT depth
        if opt.eval_split == "hamlyn" and hamlyn_depth_info is not None:
            # On-the-fly loading for Hamlyn (when npz not available)
            depth_path, sequence = hamlyn_depth_info[i]
            gt_depth = np.array(Image.open(depth_path), dtype=np.float32)
            # Apply cropping for sequence > 13
            if sequence > 13:
                gt_depth = gt_depth[:, 180:590]
            # Keep in millimeters (no conversion)
        else:
            # Load from npz (SCARED or Hamlyn with npz)
            gt_depth = gt_depths[i]
            # Keep in millimeters (no conversion)
        
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1/pred_disp
        
        # Convert predicted depth from meters to millimeters to match GT unit
        pred_depth = pred_depth * 1000.0
        
        # === 后处理：双边滤波 ===
        if use_bilateral_filter:
            # 归一化到 [0, 1] 范围
            pred_depth_norm = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-8)
            # 双边滤波
            pred_depth_filtered = bilateral_filter_depth(pred_depth_norm, d=5, sigma_color=50, sigma_space=50)
            # 还原到原始范围
            pred_depth = pred_depth_filtered * (pred_depth.max() - pred_depth.min()) + pred_depth.min()

        # Debug: Print statistics for first 10 samples to diagnose issues
        if i < 10:
            print(f"\n=== Debug Info for First Sample ===")
            print(f"GT depth: min={gt_depth.min():.4f}, max={gt_depth.max():.4f}, mean={gt_depth.mean():.4f}")
            print(f"GT depth non-zero: {np.count_nonzero(gt_depth)} / {gt_depth.size}")
            print(f"Pred depth: min={pred_depth.min():.4f}, max={pred_depth.max():.4f}, mean={pred_depth.mean():.4f}")
            print(f"Depth range: MIN_DEPTH={MIN_DEPTH}, MAX_DEPTH={MAX_DEPTH}")
        
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        
        # Debug: Print mask statistics for first 10 samples
        if i < 10:
            print(f"Valid pixels in mask: {np.count_nonzero(mask)} / {mask.size}")
            if np.count_nonzero(mask) > 0:
                print(f"GT depth (masked): min={gt_depth[mask].min():.4f}, max={gt_depth[mask].max():.4f}")
                print(f"Pred depth (masked): min={pred_depth[mask].min():.4f}, max={pred_depth[mask].max():.4f}")
            print(f"===================================\n")
        
        pred_depth_my=pred_depth

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        # Skip samples with no valid pixels
        if len(gt_depth) == 0:
            if i < 10:  # Print warning for first 10 samples
                print(f"⚠️  WARNING: Sample {i} has no valid pixels, skipping...")
            continue

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio
            pred_depth_my *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        pred_depth_my[pred_depth_my < MIN_DEPTH] = MIN_DEPTH
        pred_depth_my[pred_depth_my > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))
        
        # === Save predicted depth map for DEHS calculation ===
        # Only save if within max_save_samples limit (to save disk space)
        should_save = (opt.max_save_samples is None) or (i < opt.max_save_samples)
        
        if should_save:
            # Generate filename from original filename or use index
            if filenames is not None and i < len(filenames):
                # Extract base filename from the path
                base_name = os.path.basename(filenames[i])
                # Remove extension and add .npy
                depth_filename = os.path.splitext(base_name)[0] + "_depth.npy"
            else:
                # Fallback to index-based naming
                depth_filename = "{:06d}_depth.npy".format(i)
            
            depth_save_path = os.path.join(save_dir, depth_filename)
            np.save(depth_save_path, pred_depth_my)

    num_evaluated = len(errors)
    num_total = pred_disps.shape[0]
    num_skipped = num_total - num_evaluated
    
    if num_skipped > 0:
        print(f"\n⚠️  Skipped {num_skipped} samples (out of {num_total}) due to no valid pixels")
    
    print(f"-> Evaluated {num_evaluated} samples successfully")
    
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 8).format("abs_diff","abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
