"""
Composite Lighting Loss Functions

Implements region-adaptive supervision strategy for Composite Lighting FDA-Net.

Key innovations:
1. Region-adaptive reconstruction: different supervision in specular vs non-specular regions
2. Decoupling constraints: ensure L_diffuse and L_specular are orthogonal
3. Physical constraints: smoothness for diffuse, sparsity for specular
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLightingLoss(nn.Module):
    """
    Loss function for Composite Lighting FDA-Net.
    
    Implements region-adaptive supervision:
    - In non-specular regions: enforce L_diffuse reconstruction
    - In specular regions: enforce L_specular contribution
    - Global: ensure L_diffuse + L_specular = L_total
    """
    
    def __init__(self, opt):
        super(CompositeLightingLoss, self).__init__()
        self.opt = opt
        
        # Loss weights from config
        self.recon_global_weight = getattr(opt, 'composite_recon_global_weight', 0.5)
        self.recon_diffuse_region_weight = getattr(opt, 'composite_recon_diffuse_region_weight', 0.3)
        self.recon_specular_region_weight = getattr(opt, 'composite_recon_specular_region_weight', 0.3)
        self.diffuse_smooth_weight = getattr(opt, 'composite_diffuse_smooth_weight', 0.1)
        self.diffuse_freq_prior_weight = getattr(opt, 'composite_diffuse_freq_prior_weight', 0.2)
        self.specular_sparse_weight = getattr(opt, 'composite_specular_sparse_weight', 0.2)
        self.specular_consistency_weight = getattr(opt, 'composite_specular_consistency_weight', 0.1)
        self.orthogonal_weight = getattr(opt, 'composite_orthogonal_weight', 0.1)
        self.alpha_guidance_weight = getattr(opt, 'composite_alpha_guidance_weight', 0.1)

        # Direct mean constraint on L_diffuse (Stage 2 "surgical" guidance)
        # Encourages L_diffuse_mean to move towards a target value instead of staying locked
        self.diffuse_mean_constraint_weight = getattr(
            opt, 'composite_diffuse_mean_constraint_weight', 0.0
        )
        self.diffuse_target_mean = getattr(
            opt, 'composite_diffuse_target_mean', 0.2
        )

        # Activation guidance on L_specular (Stage 2 "revival" guidance)
        # Encourages L_specular_mean to stay in a reasonable non-zero range
        self.specular_activation_weight = getattr(
            opt, 'composite_specular_activation_weight', 0.0
        )
        self.specular_target_activation = getattr(
            opt, 'composite_specular_target_activation', 0.025
        )
        
        # Region definition threshold
        self.specular_threshold = getattr(opt, 'composite_specular_threshold', 0.3)
    
    def forward(self, rgb, R, L_total, aux_outputs):
        """
        Compute composite lighting loss.
        
        Args:
            rgb: (B, 3, H, W) original RGB image
            R: (B, 3, H, W) predicted reflectance
            L_total: (B, 3, H, W) total illumination
            aux_outputs: dict of auxiliary outputs from CompositeLightingFDANet
            
        Returns:
            total_loss: scalar tensor
            losses: dict of individual loss components for logging
        """
        L_diffuse = aux_outputs['L_diffuse']
        L_specular = aux_outputs['L_specular']
        L_diffuse_1ch = aux_outputs['L_diffuse_1ch']
        L_specular_1ch = aux_outputs['L_specular_1ch']
        mask_specular = aux_outputs['mask_specular']
        specular_detection = aux_outputs['specular_detection']
        alpha = aux_outputs['alpha']
        
        # Define regions
        specular_region = (mask_specular > self.specular_threshold).float()
        non_specular_region = 1.0 - specular_region
        
        # Avoid division by zero
        specular_count = specular_region.sum() + 1e-6
        non_specular_count = non_specular_region.sum() + 1e-6
        
        # === 1. Global reconstruction loss ===
        reconstruction = rgb - R * L_total
        loss_recon_global = reconstruction.abs().mean()
        
        # === 2. Region-adaptive reconstruction ===
        # In non-specular regions, L_diffuse should be sufficient
        recon_diffuse = rgb - R * L_diffuse
        loss_recon_diffuse_region = (recon_diffuse.abs() * non_specular_region).sum() / non_specular_count
        
        # In specular regions, L_total (diffuse + specular) should reconstruct well
        recon_specular = rgb - R * L_total
        loss_recon_specular_region = (recon_specular.abs() * specular_region).sum() / specular_count
        
        # === 3. L_diffuse constraints ===
        # Smoothness
        loss_diffuse_smooth = self.smooth_loss(L_diffuse_1ch)
        
        # Frequency prior (low frequency)
        loss_diffuse_freq_prior = self.frequency_prior_loss(
            aux_outputs['mask_diffuse'], 
            favor='low'
        )

        # Direct mean constraint: push L_diffuse_mean towards a target value
        if self.diffuse_mean_constraint_weight > 0.0:
            L_diffuse_mean = L_diffuse_1ch.mean()
            loss_diffuse_mean = (L_diffuse_mean - self.diffuse_target_mean) ** 2
        else:
            # Keep graph-connected scalar for logging even if weight is zero
            loss_diffuse_mean = L_diffuse_1ch.mean() * 0.0

        # Activation guidance: keep L_specular_mean around a target activation level
        if self.specular_activation_weight > 0.0:
            L_specular_mean = L_specular_1ch.mean()
            loss_specular_activation = (L_specular_mean - self.specular_target_activation) ** 2
        else:
            loss_specular_activation = L_specular_1ch.mean() * 0.0
        
        # === 4. L_specular constraints ===
        # Sparsity (encourage sparse specular mask)
        # Option A (current): Simple mean - encourages overall small values
        # Option B (advanced): True sparsity - encourages few non-zero elements
        use_advanced_sparsity = getattr(self.opt, 'composite_use_advanced_sparsity', False)
        
        if use_advanced_sparsity:
            # Advanced: (mask^2).sum() / (mask.sum() + ε) - true sparsity
            loss_specular_sparse = (mask_specular ** 2).sum() / (mask_specular.sum() + 1e-6)
        else:
            # Simple: mask.mean() - encourages small values (more stable initially)
            loss_specular_sparse = mask_specular.mean()
        
        # Consistency with detection
        loss_specular_consistency = F.mse_loss(mask_specular, specular_detection)
        
        # === 5. Decoupling constraint ===
        # L_diffuse and L_specular should be orthogonal (non-overlapping)
        loss_orthogonal = (L_diffuse_1ch * L_specular_1ch).mean()
        
        # === 6. Fusion weight guidance ===
        # Alpha should be high in specular regions
        loss_alpha_guidance = F.mse_loss(alpha, mask_specular)
        
        # === Total loss ===
        total_loss = (
            self.recon_global_weight * loss_recon_global +
            self.recon_diffuse_region_weight * loss_recon_diffuse_region +
            self.recon_specular_region_weight * loss_recon_specular_region +
            self.diffuse_smooth_weight * loss_diffuse_smooth +
            self.diffuse_freq_prior_weight * loss_diffuse_freq_prior +
            self.diffuse_mean_constraint_weight * loss_diffuse_mean +
            self.specular_activation_weight * loss_specular_activation +
            self.specular_sparse_weight * loss_specular_sparse +
            self.specular_consistency_weight * loss_specular_consistency +
            self.orthogonal_weight * loss_orthogonal +
            self.alpha_guidance_weight * loss_alpha_guidance
        )
        
        # Return detailed loss dictionary for logging
        losses = {
            'composite/total': total_loss,
            'composite/recon_global': loss_recon_global,
            'composite/recon_diffuse_region': loss_recon_diffuse_region,
            'composite/recon_specular_region': loss_recon_specular_region,
            'composite/diffuse_smooth': loss_diffuse_smooth,
            'composite/diffuse_freq_prior': loss_diffuse_freq_prior,
            'composite/diffuse_mean': loss_diffuse_mean,
            'composite/specular_activation': loss_specular_activation,
            'composite/specular_sparse': loss_specular_sparse,
            'composite/specular_consistency': loss_specular_consistency,
            'composite/orthogonal': loss_orthogonal,
            'composite/alpha_guidance': loss_alpha_guidance,
            # Statistics
            'composite/specular_region_ratio': specular_region.mean(),
            'composite/alpha_mean': alpha.mean(),
            'composite/L_diffuse_mean': L_diffuse_1ch.mean(),
            'composite/L_specular_mean': L_specular_1ch.mean()
        }
        
        return total_loss, losses
    
    def smooth_loss(self, img):
        """
        Smoothness loss (total variation).
        
        Args:
            img: (B, 1, H, W) image
            
        Returns:
            scalar loss
        """
        grad_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        grad_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        return grad_x.mean() + grad_y.mean()
    
    def frequency_prior_loss(self, mask, favor='low'):
        """
        Frequency prior loss.
        
        Args:
            mask: (B, 1, H, W_freq) frequency mask
            favor: 'low' or 'high'
            
        Returns:
            scalar loss
        """
        B, C, H, W_freq = mask.shape
        
        # Create radial frequency prior
        cy, cx = H // 2, W_freq // 2
        y = torch.arange(H, device=mask.device).float() - cy
        x = torch.arange(W_freq, device=mask.device).float() - cx
        # Use compatible meshgrid (indexing='ij' is default in older PyTorch versions)
        try:
            yy, xx = torch.meshgrid(y, x, indexing='ij')
        except TypeError:
            # Fallback for older PyTorch versions (< 1.10.0)
            yy, xx = torch.meshgrid(y, x)
        radius = torch.sqrt(yy**2 + xx**2)
        radius_normalized = radius / (radius.max() + 1e-8)
        
        if favor == 'low':
            # Low frequency prior: encourage high mask values at center (low radius)
            prior = 1.0 - radius_normalized
        else:
            # High frequency prior: encourage high mask values at edges (high radius)
            prior = radius_normalized
        
        prior = prior.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W_freq)
        
        # Compute loss: encourage mask to align with prior
        loss = torch.abs(mask - prior).mean()
        
        return loss


class SpecularDetectionLoss(nn.Module):
    """
    Loss for Stage 1: Specular detection pretraining.
    
    This is used to pretrain the spatial branch before joint training.
    """
    
    def __init__(self, opt):
        super(SpecularDetectionLoss, self).__init__()
        self.opt = opt
        
        self.detection_weight = getattr(opt, 'stage1_detection_weight', 1.0)
        self.sparse_weight = getattr(opt, 'stage1_sparse_weight', 0.2)
        self.localization_weight = getattr(opt, 'stage1_localization_weight', 0.5)
    
    def forward(self, rgb, mask_specular, specular_detection):
        """
        Stage 1 loss for specular detection.
        
        Args:
            rgb: (B, 3, H, W) input image
            mask_specular: (B, 1, H, W) refined specular mask
            specular_detection: (B, 1, H, W) raw detection result
            
        Returns:
            total_loss: scalar tensor
            losses: dict of loss components
        """
        # 1. Detection accuracy: mask should match detection
        loss_detection = F.mse_loss(mask_specular, specular_detection)
        
        # 2. Sparsity: specular highlights should be sparse
        loss_sparse = mask_specular.mean()
        
        # 3. Localization: encourage high confidence in detected regions
        # Use pseudo-labels from intensity-based heuristic
        intensity = rgb.mean(dim=1, keepdim=True)
        threshold = torch.quantile(intensity.view(rgb.size(0), -1), 0.90, dim=1)
        pseudo_label = (intensity > threshold.view(-1, 1, 1, 1)).float()
        loss_localization = F.binary_cross_entropy(mask_specular, pseudo_label)
        
        # Total loss
        total_loss = (
            self.detection_weight * loss_detection +
            self.sparse_weight * loss_sparse +
            self.localization_weight * loss_localization
        )
        
        losses = {
            'stage1/total': total_loss,
            'stage1/detection': loss_detection,
            'stage1/sparse': loss_sparse,
            'stage1/localization': loss_localization,
            'stage1/mask_mean': mask_specular.mean()
        }
        
        return total_loss, losses


if __name__ == "__main__":
    # Test loss functions
    print("Testing CompositeLightingLoss...")
    
    # Mock options
    class MockOpt:
        composite_recon_global_weight = 0.5
        composite_recon_diffuse_region_weight = 0.3
        composite_recon_specular_region_weight = 0.3
        composite_diffuse_smooth_weight = 0.1
        composite_diffuse_freq_prior_weight = 0.2
        composite_specular_sparse_weight = 0.2
        composite_specular_consistency_weight = 0.1
        composite_orthogonal_weight = 0.1
        composite_alpha_guidance_weight = 0.1
        composite_specular_threshold = 0.3
    
    opt = MockOpt()
    loss_fn = CompositeLightingLoss(opt)
    
    # Create dummy data
    B, C, H, W = 2, 3, 256, 320
    rgb = torch.rand(B, C, H, W)
    R = torch.rand(B, C, H, W)
    L_total = torch.rand(B, C, H, W)
    
    aux_outputs = {
        'L_diffuse': torch.rand(B, C, H, W),
        'L_specular': torch.rand(B, C, H, W),
        'L_diffuse_1ch': torch.rand(B, 1, H, W),
        'L_specular_1ch': torch.rand(B, 1, H, W),
        'mask_diffuse': torch.rand(B, 1, H, W // 2 + 1),
        'mask_specular': torch.rand(B, 1, H, W),
        'specular_detection': torch.rand(B, 1, H, W),
        'alpha': torch.rand(B, 1, H, W)
    }
    
    # Forward pass
    total_loss, losses = loss_fn(rgb, R, L_total, aux_outputs)
    
    print(f"✓ Loss computation successful")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Number of loss components: {len(losses)}")
    
    # Test backward
    total_loss.backward()
    print(f"✓ Backward pass successful")
    
    print("\n✅ All tests passed!")

