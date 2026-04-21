from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):

        self.features = []
        # x = (input_image - 0.45) / 0.225
        x = input_image
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class ResnetEncoderForDecompose(nn.Module):
    """
    ResNet encoder specifically designed for intrinsic decomposition with
    homomorphic filtering enhancement.
    
    This encoder accepts 12-channel input:
        - Channels 0-2: Original RGB (augmented)
        - Channels 3-11: Multi-scale homomorphic features (9 channels from 3 scales)
    
    The first convolutional layer is modified to accept 12 channels, with
    intelligent weight initialization from pretrained 3-channel models.
    
    Weight Initialization Strategy (A+C):
        - Channels 0-2 (RGB): Direct copy from pretrained weights
        - Channels 3-11 (Homo): 1/3 copy of pretrained weights + random perturbation
    
    Args:
        num_layers: Number of ResNet layers (18, 34, 50, 101, 152)
        pretrained: Whether to use pretrained ImageNet weights
        num_homo_channels: Number of homomorphic feature channels (default: 9)
    """
    def __init__(self, num_layers, pretrained, num_homo_channels=9):
        super(ResnetEncoderForDecompose, self).__init__()
        
        self.num_layers = num_layers
        self.num_homo_channels = num_homo_channels
        self.num_input_channels = 3 + num_homo_channels  # RGB + homomorphic
        
        # Channel configuration for each ResNet layer
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        
        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}
        
        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
        
        # Create base encoder (pretrained on 3-channel input)
        if pretrained:
            self.encoder = resnets[num_layers](pretrained=True)
        else:
            self.encoder = resnets[num_layers](pretrained=False)
        
        # Adjust channel dimensions for deeper networks
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
        
        # Replace first conv layer to accept 12 channels
        self._modify_first_conv_layer(pretrained)
        
    def _modify_first_conv_layer(self, load_pretrained_weights):
        """
        Modify the first convolutional layer to accept 12 channels.
        
        Implements weight initialization strategy A+C:
        - First 3 channels: Copy pretrained RGB weights
        - Next 9 channels: 1/3 copy of pretrained weights + 1% random noise
        
        Args:
            load_pretrained_weights: Whether to initialize from pretrained weights
        """
        # Get original conv1 parameters
        original_conv1 = self.encoder.conv1
        out_channels = original_conv1.out_channels
        kernel_size = original_conv1.kernel_size
        stride = original_conv1.stride
        padding = original_conv1.padding
        bias = original_conv1.bias is not None
        
        # Create new conv1 with 12 input channels
        new_conv1 = nn.Conv2d(
            in_channels=self.num_input_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
        if load_pretrained_weights:
            with torch.no_grad():
                # Get pretrained weights (shape: [64, 3, 7, 7])
                pretrained_weight = original_conv1.weight.data.clone()
                
                # Initialize new weights (shape: [64, 12, 7, 7])
                new_weight = torch.zeros(
                    out_channels, 
                    self.num_input_channels,
                    kernel_size[0], 
                    kernel_size[1]
                )
                
                # Strategy A: First 3 channels - direct copy of RGB weights
                new_weight[:, :3, :, :] = pretrained_weight
                
                # Strategy A+C: Next 9 channels - 1/3 copy + random perturbation
                # Each homomorphic channel gets 1/3 of the original RGB weights
                for i in range(self.num_homo_channels):
                    # Base weight: average of RGB channels scaled by 1/3
                    base_weight = pretrained_weight / 3.0
                    
                    # Add 1% random Gaussian noise for diversity
                    noise = torch.randn_like(base_weight) * 0.01
                    
                    # Assign to corresponding channel
                    new_weight[:, 3+i, :, :] = base_weight.mean(dim=1) + noise.mean(dim=1)
                
                # Assign new weights to conv1
                new_conv1.weight.data = new_weight
                
                # Copy bias if exists
                if bias:
                    new_conv1.bias.data = original_conv1.bias.data.clone()
        else:
            # Random initialization (Kaiming)
            nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
            if bias:
                nn.init.constant_(new_conv1.bias, 0)
        
        # Replace encoder's conv1
        self.encoder.conv1 = new_conv1
        
        print(f"[ResnetEncoderForDecompose] Modified conv1: "
              f"{3} -> {self.num_input_channels} input channels")
        if load_pretrained_weights:
            print(f"[ResnetEncoderForDecompose] Initialized with pretrained weights "
                  f"(Strategy A+C: RGB copy + 1/3 copy + noise)")
    
    def forward(self, input_image):
        """
        Forward pass through the encoder
        
        Args:
            input_image: Tensor of shape (B, 12, H, W)
                - Channels 0-2: RGB (augmented)
                - Channels 3-11: Homomorphic features
        
        Returns:
            List of feature maps at different scales
        """
        assert input_image.shape[1] == self.num_input_channels, \
            f"Expected {self.num_input_channels} input channels, got {input_image.shape[1]}"
        
        self.features = []
        x = input_image
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        
        return self.features
