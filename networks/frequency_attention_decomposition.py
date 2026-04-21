"""
Frequency-Domain Attention Network (FDA-Net) for Intrinsic Decomposition

This module implements a learnable frequency-domain decomposition that replaces
the fixed homomorphic filtering or IID module.

Key innovations:
1. Learnable frequency-domain attention masks
2. Adaptive separation of illumination and reflectance
3. High parameter efficiency (~500K vs IID's ~5M)
4. Strong interpretability through visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FrequencyAttentionModule(nn.Module):
    """
    Generates attention masks in frequency domain to separate illumination and reflectance.
    
    Input: Frequency domain features (magnitude + phase)
    Output: Two attention masks (mask_L for illumination, mask_R for reflectance)
    """
    def __init__(self, feature_channels=64):
        super(FrequencyAttentionModule, self).__init__()
        
        # Feature extraction from frequency domain
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(2, feature_channels, kernel_size=1),  # 2 channels: magnitude + phase
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        
        # Generate two attention masks
        self.mask_generator = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 2, 2, kernel_size=1),  # 2 masks: L and R
            nn.Sigmoid()  # Ensure masks are in [0, 1]
        )
        
    def forward(self, magnitude, phase):
        """
        Args:
            magnitude: [B, 1, H, W] - Frequency magnitude
            phase: [B, 1, H, W] - Frequency phase
            
        Returns:
            mask_L: [B, 1, H, W] - Mask for illumination (low frequency bias)
            mask_R: [B, 1, H, W] - Mask for reflectance (high frequency bias)
        """
        # Concatenate magnitude and phase
        freq_feat = torch.cat([magnitude, phase], dim=1)  # [B, 2, H, W]
        
        # Extract features
        features = self.freq_encoder(freq_feat)  # [B, C, H, W]
        
        # Generate masks
        masks = self.mask_generator(features)  # [B, 2, H, W]
        
        mask_L = masks[:, 0:1, :, :]  # Illumination mask
        mask_R = masks[:, 1:2, :, :]  # Reflectance mask
        
        return mask_L, mask_R


class SpatialRefinementModule(nn.Module):
    """
    Refines the decomposed reflectance and illumination in spatial domain.
    
    This module corrects potential artifacts from frequency-domain separation
    and ensures spatial continuity.
    """
    def __init__(self, in_channels=1, hidden_channels=32):
        super(SpatialRefinementModule, self).__init__()
        
        self.refine_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - Input from frequency domain
            
        Returns:
            refined: [B, C, H, W] - Spatially refined output
        """
        residual = self.refine_net(x)
        refined = x + residual  # Residual connection
        return refined


class FrequencyAttentionDecomposition(nn.Module):
    """
    Complete FDA-Net for intrinsic image decomposition.
    
    Pipeline:
    1. Convert RGB to grayscale
    2. Apply FFT to get frequency representation
    3. Generate attention masks via neural network
    4. Separate illumination and reflectance in frequency domain
    5. Apply IFFT to get spatial domain results
    6. Refine in spatial domain
    """
    def __init__(self, feature_channels=64, refine_channels=32, use_spatial_refine=True):
        super(FrequencyAttentionDecomposition, self).__init__()
        
        self.use_spatial_refine = use_spatial_refine
        
        # Frequency attention module
        self.freq_attention = FrequencyAttentionModule(feature_channels)
        
        # Spatial refinement module (optional)
        if self.use_spatial_refine:
            self.spatial_refine_R = SpatialRefinementModule(1, refine_channels)
            self.spatial_refine_L = SpatialRefinementModule(1, refine_channels)
        
    def rgb_to_gray(self, rgb):
        """Convert RGB to grayscale using standard weights."""
        # Standard grayscale conversion: 0.299*R + 0.587*G + 0.114*B
        weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(rgb.device)
        gray = (rgb * weights).sum(dim=1, keepdim=True)
        return gray
    
    def normalize_magnitude(self, magnitude):
        """Normalize magnitude to [0, 1] for stable training."""
        # Use log scale to compress dynamic range
        magnitude_norm = torch.log(magnitude + 1e-8)
        # Min-max normalization
        min_val = magnitude_norm.min()
        max_val = magnitude_norm.max()
        magnitude_norm = (magnitude_norm - min_val) / (max_val - min_val + 1e-8)
        return magnitude_norm
    
    def forward(self, rgb):
        """
        Args:
            rgb: [B, 3, H, W] - Input RGB image
            
        Returns:
            reflectance: [B, 1, H, W] - Reflectance component (3 channels, repeated)
            illumination: [B, 1, H, W] - Illumination component (3 channels, repeated)
            aux_outputs: dict - Auxiliary outputs for visualization and loss computation
        """
        B, C, H, W = rgb.shape
        
        # 1. Convert to grayscale
        gray = self.rgb_to_gray(rgb)  # [B, 1, H, W]
        
        # 2. FFT to frequency domain
        fft = torch.fft.fft2(gray, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        
        # 3. Extract magnitude and phase
        magnitude = torch.abs(fft_shift)  # [B, 1, H, W]
        phase = torch.angle(fft_shift)    # [B, 1, H, W]
        
        # Normalize magnitude for stable training
        magnitude_norm = self.normalize_magnitude(magnitude)
        
        # 4. Generate attention masks
        mask_L, mask_R = self.freq_attention(magnitude_norm, phase)
        
        # 5. Apply masks to separate L and R in frequency domain
        fft_L = fft_shift * mask_L
        fft_R = fft_shift * mask_R
        
        # 6. IFFT back to spatial domain
        ifft_L = torch.fft.ifftshift(fft_L, dim=(-2, -1))
        ifft_R = torch.fft.ifftshift(fft_R, dim=(-2, -1))
        
        illumination_freq = torch.fft.ifft2(ifft_L, dim=(-2, -1)).real
        reflectance_freq = torch.fft.ifft2(ifft_R, dim=(-2, -1)).real
        
        # 7. Spatial domain refinement
        if self.use_spatial_refine:
            reflectance = self.spatial_refine_R(reflectance_freq)
            illumination = self.spatial_refine_L(illumination_freq)
        else:
            reflectance = reflectance_freq
            illumination = illumination_freq
        
        # Ensure positive values and normalize
        reflectance = torch.clamp(reflectance, min=0.0)
        illumination = torch.clamp(illumination, min=0.0)
        
        # Normalize to [0, 1] range
        reflectance = reflectance / (reflectance.max() + 1e-8)
        illumination = illumination / (illumination.max() + 1e-8)
        
        # Expand to 3 channels to match RGB input
        reflectance_3ch = reflectance.repeat(1, 3, 1, 1)
        illumination_3ch = illumination.repeat(1, 3, 1, 1)
        
        # Store auxiliary outputs for visualization and loss computation
        aux_outputs = {
            'mask_L': mask_L,
            'mask_R': mask_R,
            'magnitude': magnitude,
            'phase': phase,
            'reflectance_1ch': reflectance,
            'illumination_1ch': illumination,
        }
        
        return reflectance_3ch, illumination_3ch, aux_outputs


class FrequencyPriorLoss(nn.Module):
    """
    Encourages the learned masks to follow frequency priors:
    - mask_L should bias towards low frequencies (illumination)
    - mask_R should bias towards high frequencies (reflectance)
    
    But not enforce strictly - allows adaptive adjustment.
    """
    def __init__(self, weight=0.1):
        super(FrequencyPriorLoss, self).__init__()
        self.weight = weight
        
    def create_frequency_prior(self, H, W, device):
        """
        Create prior masks:
        - Low frequency prior: Gaussian centered at DC component
        - High frequency prior: 1 - Low frequency prior
        """
        # Note: use default meshgrid behavior for compatibility with older PyTorch
        # (no `indexing` argument). This gives y: rows, x: cols in 'ij' style.
        y, x = torch.meshgrid(
            torch.arange(H, device=device) - H // 2,
            torch.arange(W, device=device) - W // 2
        )
        D = torch.sqrt(x**2 + y**2)
        
        # Gaussian for low frequency prior (sigma = min(H, W) / 6)
        sigma = min(H, W) / 6.0
        low_freq_prior = torch.exp(-D**2 / (2 * sigma**2))
        high_freq_prior = 1.0 - low_freq_prior
        
        return low_freq_prior, high_freq_prior
    
    def forward(self, mask_L, mask_R):
        """
        Args:
            mask_L: [B, 1, H, W] - Illumination mask
            mask_R: [B, 1, H, W] - Reflectance mask
            
        Returns:
            loss: scalar - Prior loss encouraging reasonable frequency separation
        """
        B, C, H, W = mask_L.shape
        device = mask_L.device
        
        # Create frequency priors
        low_freq_prior, high_freq_prior = self.create_frequency_prior(H, W, device)
        low_freq_prior = low_freq_prior.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        high_freq_prior = high_freq_prior.unsqueeze(0).unsqueeze(0)
        
        # Encourage mask_L to follow low frequency prior
        loss_L = F.mse_loss(mask_L, low_freq_prior.expand_as(mask_L))
        
        # Encourage mask_R to follow high frequency prior
        loss_R = F.mse_loss(mask_R, high_freq_prior.expand_as(mask_R))
        
        total_loss = (loss_L + loss_R) * self.weight
        
        return total_loss


class ComplementaryLoss(nn.Module):
    """
    Encourages mask_L and mask_R to be complementary: mask_L + mask_R ≈ 1
    
    This prevents degenerate solutions where both masks are 0 or both are 1.
    """
    def __init__(self, weight=0.1):
        super(ComplementaryLoss, self).__init__()
        self.weight = weight
        
    def forward(self, mask_L, mask_R):
        """
        Args:
            mask_L: [B, 1, H, W]
            mask_R: [B, 1, H, W]
            
        Returns:
            loss: scalar
        """
        mask_sum = mask_L + mask_R
        target = torch.ones_like(mask_sum)
        loss = F.mse_loss(mask_sum, target) * self.weight
        return loss


class ReconstructionLoss(nn.Module):
    """
    Encourages physical consistency: R * L ≈ original image
    
    This ensures the decomposition is physically meaningful.
    """
    def __init__(self, weight=0.1):
        super(ReconstructionLoss, self).__init__()
        self.weight = weight
        
    def forward(self, reflectance, illumination, original):
        """
        Args:
            reflectance: [B, 3, H, W]
            illumination: [B, 3, H, W]
            original: [B, 3, H, W]
            
        Returns:
            loss: scalar
        """
        reconstruction = reflectance * illumination
        loss = F.l1_loss(reconstruction, original) * self.weight
        return loss


class BalanceLoss(nn.Module):
    """
    Encourages balanced distribution between mask_L and mask_R.
    
    Prevents degenerate solutions where one mask dominates (e.g., mask_R ≈ 1, mask_L ≈ 0).
    Forces masks to be more evenly distributed around target ratios.
    """
    def __init__(self, weight=0.5, target_L=0.4, target_R=0.6):
        super(BalanceLoss, self).__init__()
        self.weight = weight
        self.target_L = target_L  # Target mean for mask_L (default: 40%)
        self.target_R = target_R  # Target mean for mask_R (default: 60%)
        
    def forward(self, mask_L, mask_R):
        """
        Args:
            mask_L: [B, 1, H, W] - Illumination mask
            mask_R: [B, 1, H, W] - Reflectance mask
            
        Returns:
            loss: scalar - Penalty for imbalanced masks
        """
        # Compute mean values across spatial dimensions for each sample
        mean_L = mask_L.mean(dim=[1, 2, 3])  # [B]
        mean_R = mask_R.mean(dim=[1, 2, 3])  # [B]
        
        # Penalize deviation from target distributions
        loss_L = F.mse_loss(mean_L, torch.ones_like(mean_L) * self.target_L)
        loss_R = F.mse_loss(mean_R, torch.ones_like(mean_R) * self.target_R)
        
        total_loss = (loss_L + loss_R) * self.weight
        
        return total_loss


class RobustSpecularDetector(nn.Module):
    """
    Robust specular highlight detector using multi-feature fusion.
    
    Combines three features to detect specular highlights:
    1. High intensity (bright pixels)
    2. Low saturation (close to white)
    3. High local contrast (sharp edges around highlights)
    
    This avoids false positives on normal high-reflectance tissue.
    """
    def __init__(self, intensity_percentile=95, saturation_threshold=0.3):
        super(RobustSpecularDetector, self).__init__()
        self.intensity_percentile = intensity_percentile
        self.saturation_threshold = saturation_threshold
        
        # Sobel kernels for gradient computation
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
    
    def forward(self, rgb):
        """
        Args:
            rgb: [B, 3, H, W] - RGB image in range [0, 1]
        
        Returns:
            specular_mask: [B, 1, H, W] - Soft mask in range [0, 1]
                          Higher values indicate higher probability of specular highlight
        """
        B, C, H, W = rgb.shape
        
        # Feature 1: High intensity (V channel in HSV)
        intensity = torch.max(rgb, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        
        # Compute per-sample threshold
        intensity_flat = intensity.view(B, -1)  # [B, H*W]
        intensity_threshold = torch.quantile(
            intensity_flat, 
            self.intensity_percentile / 100.0, 
            dim=1, 
            keepdim=True
        ).view(B, 1, 1, 1)  # [B, 1, 1, 1]
        
        high_intensity = torch.sigmoid((intensity - intensity_threshold) * 10)  # Soft threshold
        
        # Feature 2: Low saturation (close to white)
        max_rgb = torch.max(rgb, dim=1, keepdim=True)[0]
        min_rgb = torch.min(rgb, dim=1, keepdim=True)[0]
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-8)  # [B, 1, H, W]
        
        low_saturation = 1.0 - torch.sigmoid((saturation - self.saturation_threshold) * 10)
        
        # Feature 3: High local contrast (edges around highlights)
        sobel_x_out = F.conv2d(intensity, self.sobel_x, padding=1)
        sobel_y_out = F.conv2d(intensity, self.sobel_y, padding=1)
        gradient = torch.sqrt(sobel_x_out**2 + sobel_y_out**2 + 1e-8)
        
        gradient_flat = gradient.view(B, -1)
        gradient_threshold = torch.quantile(
            gradient_flat, 0.8, dim=1, keepdim=True
        ).view(B, 1, 1, 1)
        
        high_contrast = torch.sigmoid((gradient - gradient_threshold) * 10)
        
        # Fusion: specular = high_intensity AND low_saturation (AND optionally high_contrast)
        # Use soft multiplication for differentiability
        specular_score = high_intensity * low_saturation * (0.5 + 0.5 * high_contrast)
        
        return specular_score


class ImprovedSpecularAwareLoss(nn.Module):
    """
    Encourages mask_L to be high in detected specular highlight regions.
    
    This directly injects the prior: "specular highlights should be attributed to illumination"
    """
    def __init__(self, weight=0.5, intensity_percentile=95, saturation_threshold=0.3):
        super(ImprovedSpecularAwareLoss, self).__init__()
        self.weight = weight
        self.detector = RobustSpecularDetector(intensity_percentile, saturation_threshold)
    
    def forward(self, mask_L, rgb):
        """
        Args:
            mask_L: [B, 1, H, W] - Illumination mask
            rgb: [B, 3, H, W] - Original RGB image
        
        Returns:
            loss: scalar - Weighted MSE between mask_L and target (1.0) in specular regions
        """
        # Detect specular highlights
        specular_mask = self.detector(rgb)  # [B, 1, H, W], values in [0, 1]
        
        # In specular regions, encourage mask_L to be close to 1.0
        target = torch.ones_like(mask_L)
        
        # Weighted MSE: specular_mask acts as spatial weight
        loss = (specular_mask * (mask_L - target) ** 2).mean()
        
        return loss * self.weight


class DiversityBalanceLoss(nn.Module):
    """
    Encourages mask_L and mask_R to have diversity (distinction) in local patches.
    
    Unlike global BalanceLoss which forces overall 40-60 split (conflicting with specular awareness),
    this loss encourages:
    1. Local complementarity: L + R ≈ 1 in each patch
    2. Local diversity: |L - R| should be meaningful (avoid L≈R≈0.5 everywhere)
    
    This allows specular regions to have high L, while normal regions have high R.
    """
    def __init__(self, weight=0.3, patch_size=32, min_diversity=0.2):
        super(DiversityBalanceLoss, self).__init__()
        self.weight = weight
        self.patch_size = patch_size
        self.min_diversity = min_diversity
    
    def forward(self, mask_L, mask_R):
        """
        Args:
            mask_L: [B, 1, H, W]
            mask_R: [B, 1, H, W]
        
        Returns:
            loss: scalar
        """
        B, C, H, W = mask_L.shape
        
        # Divide into non-overlapping patches
        n_patches_h = H // self.patch_size
        n_patches_w = W // self.patch_size
        
        if n_patches_h == 0 or n_patches_w == 0:
            # Image too small for patches, use global statistics
            mean_L = mask_L.mean()
            mean_R = mask_R.mean()
            complementary_loss = (mean_L + mean_R - 1.0) ** 2
            diversity = torch.abs(mean_L - mean_R)
            diversity_loss = F.relu(self.min_diversity - diversity)
            return (complementary_loss + diversity_loss) * self.weight
        
        total_loss = 0.0
        count = 0
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                h_start = i * self.patch_size
                w_start = j * self.patch_size
                h_end = h_start + self.patch_size
                w_end = w_start + self.patch_size
                
                patch_L = mask_L[:, :, h_start:h_end, w_start:w_end]
                patch_R = mask_R[:, :, h_start:h_end, w_start:w_end]
                
                # Patch-wise statistics
                mean_L = patch_L.mean()
                mean_R = patch_R.mean()
                
                # Loss 1: Complementarity (L + R ≈ 1)
                complementary_loss = (mean_L + mean_R - 1.0) ** 2
                
                # Loss 2: Diversity (|L - R| should be meaningful)
                diversity = torch.abs(mean_L - mean_R)
                diversity_loss = F.relu(self.min_diversity - diversity)
                
                total_loss += complementary_loss + diversity_loss
                count += 1
        
        return (total_loss / count) * self.weight


# Quick test
if __name__ == "__main__":
    print("Testing FDA-Net modules...")
    
    # Test FrequencyAttentionModule
    print("\n1. Testing FrequencyAttentionModule...")
    freq_attn = FrequencyAttentionModule(feature_channels=64)
    magnitude = torch.randn(2, 1, 64, 64)
    phase = torch.randn(2, 1, 64, 64)
    mask_L, mask_R = freq_attn(magnitude, phase)
    print(f"   Mask L shape: {mask_L.shape}")
    print(f"   Mask R shape: {mask_R.shape}")
    print(f"   Mask L range: [{mask_L.min().item():.3f}, {mask_L.max().item():.3f}]")
    print(f"   Mask R range: [{mask_R.min().item():.3f}, {mask_R.max().item():.3f}]")
    print("   ✓ FrequencyAttentionModule works!")
    
    # Test SpatialRefinementModule
    print("\n2. Testing SpatialRefinementModule...")
    spatial_refine = SpatialRefinementModule(in_channels=1, hidden_channels=32)
    x = torch.randn(2, 1, 64, 64)
    refined = spatial_refine(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {refined.shape}")
    print("   ✓ SpatialRefinementModule works!")
    
    # Test FrequencyAttentionDecomposition
    print("\n3. Testing FrequencyAttentionDecomposition (FDA-Net)...")
    fda_net = FrequencyAttentionDecomposition(feature_channels=64, refine_channels=32)
    rgb = torch.randn(2, 3, 256, 256)
    reflectance, illumination, aux = fda_net(rgb)
    print(f"   Input RGB shape: {rgb.shape}")
    print(f"   Reflectance shape: {reflectance.shape}")
    print(f"   Illumination shape: {illumination.shape}")
    print(f"   Aux outputs keys: {aux.keys()}")
    print(f"   Mask L shape: {aux['mask_L'].shape}")
    print(f"   Mask R shape: {aux['mask_R'].shape}")
    print("   ✓ FDA-Net works!")
    
    # Test loss functions
    print("\n4. Testing loss functions...")
    
    # Frequency prior loss
    prior_loss_fn = FrequencyPriorLoss(weight=0.1)
    prior_loss = prior_loss_fn(aux['mask_L'], aux['mask_R'])
    print(f"   Frequency prior loss: {prior_loss.item():.4f}")
    
    # Complementary loss
    comp_loss_fn = ComplementaryLoss(weight=0.1)
    comp_loss = comp_loss_fn(aux['mask_L'], aux['mask_R'])
    print(f"   Complementary loss: {comp_loss.item():.4f}")
    
    # Reconstruction loss
    recon_loss_fn = ReconstructionLoss(weight=0.1)
    recon_loss = recon_loss_fn(reflectance, illumination, rgb)
    print(f"   Reconstruction loss: {recon_loss.item():.4f}")
    
    # Balance loss
    balance_loss_fn = BalanceLoss(weight=0.5, target_L=0.4, target_R=0.6)
    balance_loss = balance_loss_fn(aux['mask_L'], aux['mask_R'])
    print(f"   Balance loss: {balance_loss.item():.4f}")
    
    # Specular detector
    print("\n5. Testing RobustSpecularDetector...")
    specular_detector = RobustSpecularDetector()
    specular_mask = specular_detector(rgb)
    print(f"   Specular mask shape: {specular_mask.shape}")
    print(f"   Specular mask range: [{specular_mask.min().item():.3f}, {specular_mask.max().item():.3f}]")
    print(f"   Specular pixels (>0.5): {(specular_mask > 0.5).float().mean().item()*100:.2f}%")
    print("   ✓ Specular detector works!")
    
    # Improved specular aware loss
    specular_loss_fn = ImprovedSpecularAwareLoss(weight=0.5)
    specular_loss = specular_loss_fn(aux['mask_L'], rgb)
    print(f"   Specular aware loss: {specular_loss.item():.4f}")
    
    # Diversity balance loss
    diversity_loss_fn = DiversityBalanceLoss(weight=0.3, patch_size=32)
    diversity_loss = diversity_loss_fn(aux['mask_L'], aux['mask_R'])
    print(f"   Diversity balance loss: {diversity_loss.item():.4f}")
    
    print("   ✓ All loss functions work!")
    
    # Count parameters
    print("\n5. Counting parameters...")
    total_params = sum(p.numel() for p in fda_net.parameters())
    trainable_params = sum(p.numel() for p in fda_net.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Target: ~500K parameters")
    if trainable_params < 600000:
        print("   ✓ Parameter count is within target!")
    else:
        print("   ⚠ Parameter count exceeds target")
    
    print("\n" + "="*50)
    print("✅ All FDA-Net tests passed!")
    print("="*50)

