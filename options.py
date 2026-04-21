from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import json

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
     def __init__(self):
          self.parser = argparse.ArgumentParser(description="IID_SFM options")

          # PATHS
          self.parser.add_argument("--data_path",
                                   type=str,
                                   help="path to the training data")
          self.parser.add_argument("--log_dir",
                                   type=str,
                                   help="log directory")

          # TRAINING options
          self.parser.add_argument("--model_name",
                                   type=str,
                                   help="the name of the folder to save the model in",
                                   default=time.strftime('%Y-%m-%d-%H-%M-%S'))
          self.parser.add_argument("--split",
                                   type=str,
                                   help="which training split to use",
                                   choices=["endovis", "hamlyn", "hk", "c3vd"],
                                   default="hk")
          self.parser.add_argument("--num_layers",
                                   type=int,
                                   help="number of resnet layers",
                                   default=18,
                                   choices=[18, 34, 50, 101, 152])
          self.parser.add_argument("--dataset",
                                   type=str,
                                   help="dataset to train on",
                                   default="hk",
                                   choices=["endovis", "hamlyn", "hk", "c3vd"])
          self.parser.add_argument("--height",
                                   type=int,
                                   help="input image height",
                                   default=256)
          self.parser.add_argument("--width",
                                   type=int,
                                   help="input image width",
                                   default=320)
          self.parser.add_argument("--disparity_smoothness", # Les
                                   type=float,
                                   help="disparity smoothness weight",
                                   default=0.01)
          self.parser.add_argument("--reconstruction_constraint", # Lds
                                   type=float,
                                   help="consistency constraint weight",
                                   default=0.2)
          self.parser.add_argument("--reflec_constraint", # La
                                   type=float,
                                   help="epipolar constraint weight",
                                   default=0.2)
          self.parser.add_argument("--reprojection_constraint", #Lms
                                   type=float,
                                   help="geometry constraint weight",
                                   default=1)
          self.parser.add_argument("--scales",
                                   nargs="+",
                                   type=int,
                                   help="scales used in the loss",
                                   default=[0, 1, 2, 3])
          self.parser.add_argument("--min_depth",
                                   type=float,
                                   help="minimum depth",
                                   default=0.1)
          self.parser.add_argument("--max_depth",
                                   type=float,
                                   help="maximum depth",
                                   default=150.0)
          self.parser.add_argument("--frame_ids",
                                   nargs="+",
                                   type=int,
                                   help="frames to load",
                                   default=[0, -1, 1])
          # added by rema for training

          self.parser.add_argument("--inpaint_pseudo_gt_dir", type=str,
                                   help='path to the pseudo gt directory',
                                   default=None)
          self.parser.add_argument("--flipping", help="if set, uses flipping",
                                   action="store_true")
          self.parser.add_argument("--rotating", help="if set, uses rotating",
                                   action="store_true")
          self.parser.add_argument("--light_in_depth", help="if set, uses light in depth",
                                   action="store_true")
          self.parser.add_argument("--input_mask_path", type=str,
                                   help='path to the input mask',
                                   default=None)
          self.parser.add_argument("--distorted", help="if set, uses distorted intrinsics",
                                   action="store_true")
          self.parser.add_argument("--config", type=str,
                                   help='path to the config file',
                                   default=None)
          self.parser.add_argument("--aug_type", type=str,
                                   help='type of data augmentation',
                                   default='',
                                   choices=['', 'add', 'rem', 'addrem'])
          self.parser.add_argument("--automasking",
                                   help="if set, does auto-masking",
                                   action="store_true")
          self.parser.add_argument("--png",
                                   help="if set, trains from raw KITTI png files (instead of jpgs)",
                                   action="store_true")
          self.parser.add_argument("--noadjust", help="if set, does not adjust the shading",
                                  action="store_true")
          self.parser.add_argument("--adjust_net_type",
                                  type=str,
                                  help="type of shading adjustment network",
                                  default="original",
                                  choices=["original", "of_af_final", "lightweight_ofaf"])
         
         # IDICR options
          self.parser.add_argument("--use_co_refinement",
                                  action="store_true",
                                  help="use iterative co-refinement (IDICR)")
          self.parser.add_argument("--num_refinement_iterations",
                                  type=int,
                                  default=2,
                                  help="number of refinement iterations (default: 2)")
          self.parser.add_argument("--geo_photo_consistency",
                                  type=float,
                                  default=0.1,
                                  help="weight for geometry-photometry consistency loss")
          self.parser.add_argument("--edge_alignment",
                                  type=float,
                                  default=0.05,
                                  help="weight for edge alignment loss")
          self.parser.add_argument("--feature_contrastive",
                                  type=float,
                                  default=0.02,
                                  help="weight for feature contrastive loss")
          
          # IDICR v2: Physical constraint weights (curriculum learning)
          self.parser.add_argument("--lambertian_weight_start",
                                  type=float,
                                  default=0.05,
                                  help="Lambertian consistency weight at start")
          self.parser.add_argument("--lambertian_weight_end",
                                  type=float,
                                  default=0.3,
                                  help="Lambertian consistency weight at end")
          self.parser.add_argument("--temporal_weight_start",
                                  type=float,
                                  default=0.0,
                                  help="Temporal consistency weight at start")
          self.parser.add_argument("--temporal_weight_end",
                                  type=float,
                                  default=0.15,
                                  help="Temporal consistency weight at end")
          self.parser.add_argument("--residual_weight_start",
                                  type=float,
                                  default=0.01,
                                  help="Residual regularization weight at start")
          self.parser.add_argument("--residual_weight_end",
                                  type=float,
                                  default=0.1,
                                  help="Residual regularization weight at end")
          
          # IDICR v2: Curriculum learning stages
          self.parser.add_argument("--curriculum_stage1_epochs",
                                  type=int,
                                  default=5,
                                  help="Number of epochs for stage 1 (foundation)")
          self.parser.add_argument("--curriculum_stage2_epochs",
                                  type=int,
                                  default=10,
                                  help="Number of epochs for stage 2 (progressive introduction)")
          
          self.parser.add_argument("--disparity_spatial_constraint", 
                                   help="disparity spatial constraint weight",
                                   type=float,
                                   default=0.0)

          # HOMOMORPHIC FILTERING options
          self.parser.add_argument("--use_homomorphic_fusion",
                                   help="if set, uses multi-scale homomorphic filtering to enhance "
                                        "intrinsic decomposition encoder input (12 channels)",
                                   action="store_true")
          self.parser.add_argument("--homo_scales",
                                   nargs="+",
                                   type=float,
                                   help="cutoff frequency ratios for multi-scale homomorphic filtering",
                                   default=[0.3, 0.5, 0.7])
          self.parser.add_argument("--homo_gamma_low",
                                   type=float,
                                   help="gain for low frequencies (illumination suppression)",
                                   default=0.5)
          self.parser.add_argument("--homo_gamma_high",
                                   type=float,
                                   help="gain for high frequencies (reflectance enhancement)",
                                   default=1.5)
          self.parser.add_argument("--homo_use_mild_aug",
                                  help="if set, uses milder data augmentation when homomorphic fusion is enabled",
                                  action="store_true",
                                  default=True)

         # FDA-NET (Frequency-Domain Attention Network) options
          self.parser.add_argument("--use_fda_net",
                                  help="if set, uses FDA-Net instead of IID for intrinsic decomposition",
                                  action="store_true")
          self.parser.add_argument("--fda_feature_channels",
                                  type=int,
                                  help="number of feature channels in FDA-Net attention module",
                                  default=64)
          self.parser.add_argument("--fda_refine_channels",
                                  type=int,
                                  help="number of channels in spatial refinement module",
                                  default=32)
          self.parser.add_argument("--fda_use_spatial_refine",
                               help="if set, uses spatial refinement after frequency decomposition",
                                  action="store_true",
                                  default=True)
          self.parser.add_argument("--fda_freq_prior_weight",
                                  type=float,
                                  help="weight for frequency prior loss",
                                  default=0.1)
          self.parser.add_argument("--fda_complementary_weight",
                                  type=float,
                                  help="weight for complementary mask loss",
                                  default=0.1)
          self.parser.add_argument("--fda_reconstruction_weight",
                                  type=float,
                                  help="weight for reconstruction loss (R*L≈I)",
                                  default=0.1)
          self.parser.add_argument("--fda_balance_weight",
                                  type=float,
                                  help="weight for balance loss (encourages mask_L/mask_R to be balanced)",
                                  default=0.5)
          self.parser.add_argument("--fda_balance_target_L",
                                  type=float,
                                  help="target mean for mask_L (default: 0.4 = 40%%)",
                                  default=0.4)
          self.parser.add_argument("--fda_balance_target_R",
                                  type=float,
                                  help="target mean for mask_R (default: 0.6 = 60%%)",
                                  default=0.6)
          
          # FDA-Net v3: Specular-aware losses
          self.parser.add_argument("--fda_specular_weight",
                                  type=float,
                                  help="weight for specular-aware loss (encourages high mask_L in specular regions)",
                                  default=0.5)
          self.parser.add_argument("--fda_specular_percentile",
                                  type=float,
                                  help="intensity percentile for specular detection (default: 95)",
                                  default=95)
          self.parser.add_argument("--fda_specular_saturation_threshold",
                                  type=float,
                                  help="saturation threshold for specular detection (default: 0.3)",
                                  default=0.3)
          self.parser.add_argument("--fda_diversity_balance_weight",
                                  type=float,
                                  help="weight for diversity balance loss (local patch-wise balance)",
                                  default=0.3)
          self.parser.add_argument("--fda_diversity_patch_size",
                                  type=int,
                                  help="patch size for diversity balance loss",
                                  default=32)
          self.parser.add_argument("--fda_diversity_min_diversity",
                                  type=float,
                                  help="minimum diversity (|mean_L - mean_R|) required in each patch",
                                  default=0.2)

         # COMPOSITE LIGHTING FDA-NET options (Phase 3)
          self.parser.add_argument("--use_composite_lighting",
                                  help="if set, uses Composite Lighting FDA-Net (Phase 3 architecture)",
                                  action="store_true")
          self.parser.add_argument("--training_stage",
                                  type=int,
                                  help="training stage: 1=specular detection pretrain, 2=joint training",
                                  choices=[1, 2],
                                  default=2)
          self.parser.add_argument("--composite_feature_channels",
                                  type=int,
                                  help="feature channels for composite lighting frequency branch",
                                  default=64)
          self.parser.add_argument("--composite_refine_channels",
                                  type=int,
                                  help="refine channels for composite lighting spatial refinement",
                                  default=32)
          self.parser.add_argument("--composite_use_spatial_refine",
                                  help="if set, uses spatial refinement in composite lighting",
                                  action="store_true",
                                  default=True)
          
          # Composite Lighting: Freezing options
          self.parser.add_argument("--freeze_freq_branch",
                                  help="if set, freezes frequency branch (for Stage 1)",
                                  action="store_true")
          self.parser.add_argument("--freeze_specular_detector",
                                  help="if set, freezes specular detector (for Stage 2)",
                                  action="store_true")
          self.parser.add_argument("--freeze_fusion",
                                  help="if set, freezes fusion module (for Stage 1)",
                                  action="store_true")
          
          # Composite Lighting: Stage 1 loss weights
          self.parser.add_argument("--stage1_detection_weight",
                                  type=float,
                                  help="Stage 1: detection accuracy weight",
                                  default=1.0)
          self.parser.add_argument("--stage1_sparse_weight",
                                  type=float,
                                  help="Stage 1: sparsity weight",
                                  default=0.2)
          self.parser.add_argument("--stage1_localization_weight",
                                  type=float,
                                  help="Stage 1: localization weight",
                                  default=0.5)
          
          # Composite Lighting: Stage 2 loss weights
          self.parser.add_argument("--composite_recon_global_weight",
                                  type=float,
                                  help="Stage 2: global reconstruction weight",
                                  default=0.5)
          self.parser.add_argument("--composite_recon_diffuse_region_weight",
                                  type=float,
                                  help="Stage 2: diffuse region reconstruction weight",
                                  default=0.3)
          self.parser.add_argument("--composite_recon_specular_region_weight",
                                  type=float,
                                  help="Stage 2: specular region reconstruction weight",
                                  default=0.3)
          self.parser.add_argument("--composite_diffuse_smooth_weight",
                                  type=float,
                                  help="Stage 2: diffuse smoothness weight",
                                  default=0.1)
          self.parser.add_argument("--composite_diffuse_freq_prior_weight",
                                  type=float,
                                  help="Stage 2: diffuse frequency prior weight",
                                  default=0.2)
          self.parser.add_argument("--composite_diffuse_mean_constraint_weight",
                                  type=float,
                                  help="Stage 2: direct mean constraint weight for L_diffuse",
                                  default=0.0)
          self.parser.add_argument("--composite_diffuse_target_mean",
                                  type=float,
                                  help="Stage 2: target mean value for L_diffuse (used by mean constraint)",
                                  default=0.2)
          self.parser.add_argument("--composite_specular_activation_weight",
                                  type=float,
                                  help="Stage 2: activation guidance weight for L_specular mean",
                                  default=0.15)
          self.parser.add_argument("--composite_specular_target_activation",
                                  type=float,
                                  help="Stage 2: target mean activation value for L_specular",
                                  default=0.025)
          self.parser.add_argument("--composite_specular_sparse_weight",
                                  type=float,
                                  help="Stage 2: specular sparsity weight",
                                  default=0.2)
          self.parser.add_argument("--composite_specular_consistency_weight",
                                  type=float,
                                  help="Stage 2: specular consistency weight",
                                  default=0.1)
          self.parser.add_argument("--composite_orthogonal_weight",
                                  type=float,
                                  help="Stage 2: orthogonality (decoupling) weight",
                                  default=0.1)
          self.parser.add_argument("--composite_alpha_guidance_weight",
                                  type=float,
                                  help="Stage 2: alpha fusion guidance weight",
                                  default=0.1)
          self.parser.add_argument("--composite_specular_threshold",
                                  type=float,
                                  help="Stage 2: threshold to define specular regions",
                                  default=0.3)
          self.parser.add_argument("--composite_use_advanced_sparsity",
                                  help="if set, uses advanced sparsity loss (true sparsity)",
                                  action="store_true")
          
          # Composite Lighting: Stage 1 weights loading
          self.parser.add_argument("--load_composite_stage1",
                                  help="if set, loads pretrained Stage 1 weights",
                                  action="store_true")
          self.parser.add_argument("--composite_stage1_weights",
                                  type=str,
                                  help="path to Stage 1 pretrained weights",
                                  default="")

         # INTRINSIC-GUIDED DEPTH ESTIMATION options (Stage A)
          self.parser.add_argument("--use_intrinsic_guidance",
                                   help="if set, uses IntrinsicGuidedDepthDecoder with R+S guidance",
                                   action="store_true")
          self.parser.add_argument("--smooth_aware_weight",
                                   type=float,
                                   help="weight for reflectance-aware smoothness loss",
                                   default=0.003)
          self.parser.add_argument("--intrinsic_consistency_weight",
                                   type=float,
                                   help="weight for intrinsic-space photometric consistency loss",
                                   default=0.2)

          # TWO-STAGE TRAINING options (Scheme C)
          self.parser.add_argument("--freeze_depth",
                                   help="if set, freezes depth network (for stage 2/3 training)",
                                   action="store_true")
          self.parser.add_argument("--freeze_pose",
                                   help="if set, freezes pose network (for stage 2/3 training)",
                                   action="store_true")
          self.parser.add_argument("--freeze_decompose",
                                   help="if set, freezes decompose network (for stage 3 training)",
                                   action="store_true")
          self.parser.add_argument("--decompose_lr_scale",
                                   type=float,
                                   help="learning rate scale for decompose network (for stage 2/3 fine-tuning)",
                                   default=1.0)
          self.parser.add_argument("--ofaf_lr_scale",
                                   type=float,
                                   help="learning rate scale for OF-AF module",
                                   default=1.0)
          self.parser.add_argument("--depth_lr_scale",
                                   type=float,
                                   help="learning rate scale for depth network (for stage 3 fine-tuning)",
                                   default=1.0)
          self.parser.add_argument("--pose_lr_scale",
                                   type=float,
                                   help="learning rate scale for pose network (for stage 3 fine-tuning)",
                                   default=1.0)
          self.parser.add_argument("--load_partial",
                                   help="if set, only loads matching weights (useful for stage transitions)",
                                   action="store_true")

          # OPTIMIZATION options
          self.parser.add_argument("--batch_size",
                                   type=int,
                                   help="batch size",
                                   default=8)
          self.parser.add_argument("--learning_rate",
                                   type=float,
                                   help="learning rate",
                                   default=1e-4)
          self.parser.add_argument("--num_epochs",
                                   type=int,
                                   help="number of epochs",
                                   default=20)
          self.parser.add_argument("--scheduler_step_size",
                                   type=int,
                                   help="step size of the scheduler",
                                   default=10)
          self.parser.add_argument("--scheduler_gamma",
                                   type=float,
                                   help="multiplicative factor of learning rate decay",
                                   default=0.1)
          self.parser.add_argument("--patience",
                                   type=int,
                                   help="early stopping patience (number of epochs without improvement)",
                                   default=0)
          self.parser.add_argument("--feature_alignment_weight",
                                   type=float,
                                   help="weight for feature alignment loss",
                                   default=0.0)

          # ABLATION options
          self.parser.add_argument("--weights_init",
                                   type=str,
                                   help="pretrained or scratch",
                                   default="pretrained",
                                   choices=["pretrained", "scratch"])
          self.parser.add_argument("--pose_model_input",
                                   type=str,
                                   help="how many images the pose network gets",
                                   default="pairs",
                                   choices=["pairs", "all"])

          # SYSTEM options
          self.parser.add_argument("--no_cuda",
                                   help="if set disables CUDA",
                                   action="store_true")
          self.parser.add_argument("--num_workers",
                                   type=int,
                                   help="number of dataloader workers",
                                   default=12)

          # LOADING options
          self.parser.add_argument("--load_weights_folder",
                                   type=str,
                                   help="path to pretrained model weights (for evaluation or stage 2/3 training)",
                                   default=None)
          self.parser.add_argument("--models_to_load",
                                   nargs="+",
                                   type=str,
                                   help="models to load")

          # LOGGING options
          self.parser.add_argument("--log_frequency",
                                   type=int,
                                   help="number of batches between each tensorboard log",
                                   default=200)
          self.parser.add_argument("--save_frequency",
                                   type=int,
                                   help="number of epochs between each save",
                                   default=10)

          # EVALUATION options
          self.parser.add_argument("--eval_stereo",
                                   help="if set evaluates in stereo mode",
                                   action="store_true")
          self.parser.add_argument("--eval_mono",
                                   help="if set evaluates in mono mode",
                                   action="store_true",
                                   default=True)
          self.parser.add_argument("--disable_median_scaling",
                                   help="if set disables median scaling in evaluation",
                                   action="store_true")
          self.parser.add_argument("--pred_depth_scale_factor",
                                   help="if set multiplies predictions by this number",
                                   type=float,
                                   default=1)
          self.parser.add_argument("--ext_disp_to_eval",
                                   type=str,
                                   help="optional path to a .npy disparities file to evaluate")
          self.parser.add_argument("--eval_split",
                                   type=str,
                                   default="endovis",
                                   choices=["endovis","hamlyn"],
                                   help="which split to run eval on")
          self.parser.add_argument("--save_pred_disps",
                                   help="if set saves predicted disparities",
                                   action="store_true")
          self.parser.add_argument("--no_eval",
                                   help="if set disables evaluation",
                                   action="store_true")
          self.parser.add_argument("--use_tta",
                                   help="if set, uses test-time augmentation (horizontal flip)",
                                   action="store_true")
          self.parser.add_argument("--use_bilateral_filter",
                                   help="if set, applies bilateral filter to depth predictions",
                                   action="store_true")
          self.parser.add_argument("--max_eval_samples",
                                   type=int,
                                   default=None,
                                   help="maximum number of samples to evaluate (None = all samples)")
          self.parser.add_argument("--max_save_samples",
                                   type=int,
                                   default=None,
                                   help="maximum number of predicted depth maps to save (None = save all)")
          self.parser.add_argument("--eval_eigen_to_benchmark",
                                   help="if set assume we are loading eigen results from npy but "
                                        "we want to evaluate using the new benchmark.",
                                   action="store_true")
          self.parser.add_argument("--eval_out_dir",
                                   help="if set will output the disparities to this folder",
                                   type=str)
          self.parser.add_argument("--post_process",
                                   help="if set will perform the flipping post processing "
                                        "from the original monodepth paper",
                                   action="store_true")

        
        
     def parse(self, args=None):
          """
          Parse command line options.
          
          - If `args` is None (default), uses `sys.argv` (standard training/evaluation flow)
          - If `args` is a list of strings, parses that list (used for unit tests)
          """
          if args is None:
               self.options = self.parser.parse_args()
          else:
               self.options = self.parser.parse_args(args)
          
          if self.options.config is not None:
               # Load options from the configuration file
               with open(self.options.config, 'r') as f:
                    config = json.load(f)
               
               # Update default options with options from the configuration file
               for key, value in config.items():
                    setattr(self.options, key, value)
          return self.options
