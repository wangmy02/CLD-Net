import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedSpecularDetector(nn.Module):
    """
    Enhanced specular highlight detector with multi-scale detection and learnable weights.
    
    Improvements over RobustSpecularDetector:
    - Multi-scale detection for better coverage
    - Learnable feature weights
    - More robust to varying highlight sizes
    """
    
    def __init__(self, intensity_percentile=85, saturation_threshold=0.2):
        super(EnhancedSpecularDetector, self).__init__()
        
        # Learnable feature weights
        self.intensity_weight = nn.Parameter(torch.tensor(1.0))
        self.saturation_weight = nn.Parameter(torch.tensor(1.0))
        self.gradient_weight = nn.Parameter(torch.tensor(0.5))
        
        # Multi-scale detection
        self.scales = [1.0, 0.5, 0.25]
        
        self.percentile = intensity_percentile
        self.sat_threshold = saturation_threshold
    
    def detect_specular_multiscale(self, rgb):
        """
        Multi-scale specular detection
        
        Args:
            rgb: (B, 3, H, W) RGB image
            
        Returns:
            specular_map: (B, 1, H, W) specular probability map [0, 1]
        """
        B, C, H, W = rgb.shape
        specular_maps = []
        
        for scale in self.scales:
            if scale != 1.0:
                h, w = int(H * scale), int(W * scale)
                rgb_scaled = F.interpolate(rgb, size=(h, w), mode='bilinear', align_corners=False)
            else:
                rgb_scaled = rgb
            
            # Single scale detection
            spec_map = self.detect_single_scale(rgb_scaled)
            
            # Restore to original size
            if scale != 1.0:
                spec_map = F.interpolate(spec_map, size=(H, W), mode='bilinear', align_corners=False)
            
            specular_maps.append(spec_map)
        
        # Multi-scale fusion (max pooling)
        specular_final = torch.stack(specular_maps, dim=0).max(dim=0)[0]
        
        return specular_final
    
    def detect_single_scale(self, rgb):
        """Single-scale specular detection"""
        B = rgb.size(0)
        
        # 1. Intensity feature
        intensity = rgb.mean(dim=1, keepdim=True)
        threshold_val = torch.quantile(intensity.view(B, -1), self.percentile / 100.0, dim=1)
        high_intensity = (intensity > threshold_val.view(B, 1, 1, 1).detach()).float()
        
        # 2. Saturation feature
        rgb_sum = rgb.sum(dim=1, keepdim=True) + 1e-8
        rgb_normalized = rgb / rgb_sum
        saturation = 1.0 - rgb_normalized.std(dim=1, keepdim=True) * 1.732  # sqrt(3)
        low_saturation = (saturation < self.sat_threshold).float()
        
        # 3. Gradient feature
        grad_x = torch.abs(rgb[:, :, :, 1:] - rgb[:, :, :, :-1])
        grad_y = torch.abs(rgb[:, :, 1:, :] - rgb[:, :, :-1, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        gradient = (grad_x + grad_y).mean(dim=1, keepdim=True)
        grad_mean = gradient.mean(dim=(2, 3), keepdim=True)
        high_contrast = (gradient > grad_mean).float()
        
        # Weighted fusion
        specular_score = (
            torch.abs(self.intensity_weight) * high_intensity * 
            torch.abs(self.saturation_weight) * low_saturation * 
            (0.5 + 0.5 * torch.abs(self.gradient_weight) * high_contrast)
        )
        
        return specular_score
    
    def forward(self, rgb):
        """Forward pass"""
        return self.detect_specular_multiscale(rgb)


class SpatialSpecularModule(nn.Module):
    """
    Spatial domain specular extraction module.
    
    This module:
    1. Detects specular highlights using EnhancedSpecularDetector
    2. Refines the detection using a lightweight CNN
    3. Extracts specular illumination component
    """
    
    def __init__(self, in_channels=3, hidden_channels=32):
        super(SpatialSpecularModule, self).__init__()
        
        self.detector = EnhancedSpecularDetector()
        
        # Refinement network
        self.refine_conv = nn.Sequential(
            nn.Conv2d(4, hidden_channels, 3, padding=1),  # RGB + detection
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, rgb):
        """
        Args:
            rgb: (B, 3, H, W) input image
            
        Returns:
            L_specular: (B, 1, H, W) specular illumination
            mask_specular: (B, 1, H, W) refined specular mask
            specular_detection: (B, 1, H, W) raw detection result
        """
        # Detect specular highlights
        specular_detection = self.detector(rgb)
        
        # Refine (combine with original image)
        combined = torch.cat([rgb, specular_detection], dim=1)
        mask_specular = self.refine_conv(combined)
        
        # Extract specular illumination
        intensity = rgb.mean(dim=1, keepdim=True)
        L_specular = intensity * mask_specular
        
        return L_specular, mask_specular, specular_detection


class AdaptiveFusionModule(nn.Module):
    """
    Adaptive fusion of L_diffuse and L_specular.
    
    L_total = L_diffuse + α * L_specular
    
    where α is spatially-adaptive fusion weight predicted by a lightweight CNN.
    """
    
    def __init__(self, in_channels=2, hidden_channels=16):
        super(AdaptiveFusionModule, self).__init__()
        
        # Spatial adaptive fusion weight predictor
        self.fusion_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, L_diffuse, L_specular):
        """
        Args:
            L_diffuse: (B, 1, H, W) diffuse illumination
            L_specular: (B, 1, H, W) specular illumination
            
        Returns:
            L_total: (B, 1, H, W) total illumination
            alpha: (B, 1, H, W) fusion weight map
        """
        # Concatenate two components
        combined = torch.cat([L_diffuse, L_specular], dim=1)
        
        # Predict fusion weight
        alpha = self.fusion_net(combined)
        
        # Fusion
        L_total = L_diffuse + alpha * L_specular
        
        return L_total, alpha


class FrequencyAttentionModule(nn.Module):
    """
    Frequency domain attention module (from original FDA-Net).
    Generates attention mask in frequency domain for diffuse illumination.
    """
    
    def __init__(self, feature_channels=64):
        super(FrequencyAttentionModule, self).__init__()
        
        # Feature extraction
        self.feature_extract = nn.Sequential(
            nn.Conv2d(2, feature_channels, 3, padding=1),  # magnitude + phase
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Mask prediction
        self.mask_predict = nn.Sequential(
            nn.Conv2d(feature_channels, 1, 1),
            nn.Sigmoid()
        )
        
        nn.init.normal_(self.mask_predict[0].weight, mean=0.0, std=0.01)  # 将权重初始化为接近0
        nn.init.constant_(self.mask_predict[0].bias, 0.1)                 # 将偏置初始化为小正值
    
    def forward(self, magnitude, phase):
        """
        Args:
            magnitude: (B, 1, H, W//2+1) frequency magnitude
            phase: (B, 1, H, W//2+1) frequency phase
            
        Returns:
            mask: (B, 1, H, W//2+1) attention mask
        """
        # Concatenate magnitude and phase
        freq_features = torch.cat([magnitude, phase], dim=1)
        
        # Extract features
        features = self.feature_extract(freq_features)
        
        # Predict mask
        mask = self.mask_predict(features)
        
        return mask


class SpatialRefinementModule(nn.Module):
    """Spatial refinement for illumination (from original FDA-Net)"""
    
    def __init__(self, channels=32):
        super(SpatialRefinementModule, self).__init__()
        
        self.refine = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.refine(x) * x


class CompositeLightingFDANet(nn.Module):
    """
    Composite Lighting FDA-Net: Main architecture
    
    This network explicitly separates diffuse and specular illumination:
    - Frequency domain branch: generates smooth, center-biased L_diffuse
    - Spatial domain branch: detects and extracts scattered L_specular
    - Fusion module: adaptively combines the two components
    
    Key advantages:
    1. Resolves the contradiction between frequency prior and specular awareness
    2. Physically interpretable (follows Phong lighting model)
    3. Allows L_diffuse to have center pattern (physically correct for diffuse)
    4. Captures scattered specular highlights separately
    """
    
    def __init__(self, 
                 feature_channels=64, 
                 refine_channels=32,
                 use_spatial_refine=True):
        super(CompositeLightingFDANet, self).__init__()
        
        # Frequency domain branch (Diffuse)
        self.freq_branch = FrequencyAttentionModule(feature_channels=feature_channels)
        self.use_spatial_refine = use_spatial_refine
        if use_spatial_refine:
            self.spatial_refine_L = SpatialRefinementModule(refine_channels)
        
        # Spatial domain branch (Specular)
        self.spatial_branch = SpatialSpecularModule(
            in_channels=3,
            hidden_channels=32
        )
        
        # Fusion module
        self.fusion = AdaptiveFusionModule(in_channels=2)
    
    def forward(self, rgb):
        """
        Args:
            rgb: (B, 3, H, W) input RGB image
            
        Returns:
            R: (B, 3, H, W) reflectance
            L_total: (B, 3, H, W) total illumination
            aux_outputs: dict of auxiliary outputs for loss computation and visualization
        """
        B, C, H, W = rgb.shape
        
        # ===== 架构消融实验3：仅频域分支 =====
        # 1. Frequency domain branch → L_diffuse
        # Convert to grayscale for frequency analysis
        intensity = rgb.mean(dim=1, keepdim=True)
        
        # FFT
        fft_result = torch.fft.rfft2(intensity, dim=(-2, -1))
        magnitude = torch.abs(fft_result)
        phase = torch.angle(fft_result)
        
        # Frequency attention
        mask_diffuse = self.freq_branch(magnitude, phase)
        
        # Apply mask and inverse transform
        masked_fft = fft_result * mask_diffuse
        illumination_freq = torch.fft.irfft2(masked_fft, s=(H, W), dim=(-2, -1))
        
        # Spatial refinement
        if self.use_spatial_refine:
            L_diffuse = self.spatial_refine_L(illumination_freq)
        else:
            L_diffuse = illumination_freq
        
        L_diffuse = torch.clamp(L_diffuse, min=0.0)
        
        # 2. Spatial domain branch → L_specular
        L_specular_1ch, mask_specular, specular_detection = self.spatial_branch(rgb)

        # 3. Fusion (keep single channel for efficiency)
        L_total_1ch, alpha = self.fusion(L_diffuse, L_specular_1ch)
        
        # 4. Compute reflectance
        # Expand to 3 channels only when needed
        L_total_3ch = L_total_1ch.repeat(1, 3, 1, 1)
        R = rgb / (L_total_3ch + 1e-6)
        R = torch.clamp(R, min=0.0, max=1.0)
        
        # Expand other channels for compatibility with existing code
        L_diffuse_3ch = L_diffuse.repeat(1, 3, 1, 1)
        L_specular_3ch = L_specular_1ch.repeat(1, 3, 1, 1)
        
        # Auxiliary outputs for loss computation and visualization
        aux_outputs = {
            'L_diffuse': L_diffuse_3ch,
            'L_specular': L_specular_3ch,
            'L_diffuse_1ch': L_diffuse,
            'L_specular_1ch': L_specular_1ch,
            'mask_diffuse': mask_diffuse,
            'mask_specular': mask_specular,
            'specular_detection': specular_detection,
            'alpha': alpha,
            'magnitude': magnitude,
            'phase': phase
        }
        
        return R, L_total_3ch, aux_outputs


if __name__ == "__main__":
    # Test the network
    print("Testing CompositeLightingFDANet...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CompositeLightingFDANet().to(device)
    
    # Random input
    rgb = torch.rand(2, 3, 256, 320).to(device)
    
    # Forward pass
    R, L_total, aux = model(rgb)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {rgb.shape}")
    print(f"  R shape: {R.shape}")
    print(f"  L_total shape: {L_total.shape}")
    print(f"  L_diffuse shape: {aux['L_diffuse'].shape}")
    print(f"  L_specular shape: {aux['L_specular'].shape}")
    print(f"  alpha shape: {aux['alpha'].shape}")
    
    # Test backward
    loss = (R - rgb).abs().mean()
    loss.backward()
    print(f"✓ Backward pass successful")
    
    print("\n✅ All tests passed!")

