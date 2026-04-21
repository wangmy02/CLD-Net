from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder
from .pose_decoder import PoseDecoder
from .adjust_net import adjust_net
from .decompose_decoder import decompose_decoder

# Lightweight Encoder (shared by multiple modules)
from .lightweight_encoder import LightweightEncoder

# Lightweight OF-AF Module (方案C - Scheme C)
from .lightweight_flow_encoder import LightweightFlowEncoder
from .flow_decoder import FlowDecoder
from .lightweight_ofaf_module import (
    LightweightOFNet,
    ImplicitAFNet,
    LightweightOFAFModule
)

# Intrinsic-Guided Depth Estimation (Stage A)
from .intrinsic_guided_depth_decoder import IntrinsicGuidedDepthDecoder
from .lightweight_intrinsic_encoder import LightweightIntrinsicEncoder
from .intrinsic_depth_refinement import IntrinsicDepthRefinement

# IDICR Framework (Iterative Depth-Intrinsic Co-Refinement)
from .intrinsic_refiner import IntrinsicRefiner
from .depth_refiner import DepthRefiner
from .consistency_losses import (
    GeometryPhotometryConsistency,
    EdgeAlignmentLoss,
    FeatureContrastiveLoss,
    ConsistencyLosses
)

# FDA-Net (Frequency-Domain Attention Network for Intrinsic Decomposition)
from .frequency_attention_decomposition import (
    FrequencyAttentionDecomposition,
    FrequencyAttentionModule,
    SpatialRefinementModule,
    FrequencyPriorLoss,
    ComplementaryLoss,
    ReconstructionLoss,
    BalanceLoss,
    RobustSpecularDetector,
    ImprovedSpecularAwareLoss,
    DiversityBalanceLoss,
)

# Composite Lighting FDA-Net (Phase 3: Explicitly separate Diffuse and Specular)
from .composite_lighting_fda_net import (
    CompositeLightingFDANet,
    EnhancedSpecularDetector,
    SpatialSpecularModule,
    AdaptiveFusionModule,
)
from .composite_lighting_loss import (
    CompositeLightingLoss,
    SpecularDetectionLoss,
)
