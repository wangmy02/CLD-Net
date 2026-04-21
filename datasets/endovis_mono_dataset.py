from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset import MonoDataset


class EndovisMonoDataset(MonoDataset):
    """Endovis单视图数据集加载器
    
    适配只有单视图（mono）的数据，不需要left_img/right_img子目录
    图像直接放在序列文件夹下
    
    数据集结构:
    data_path/
    ├── dataset1/
    │   ├── keyframe1/
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ...
    │   └── keyframe2/
    │       └── ...
    └── dataset2/
        └── ...
    
    Split文件格式（兼容3列或2列）:
    dataset2/keyframe2    626    l    # 3列格式（兼容）
    或
    dataset2/keyframe2    626         # 2列格式（更简洁）
    """
    
    def __init__(self, *args, **kwargs):
        super(EndovisMonoDataset, self).__init__(*args, **kwargs)
        
        # Endovis相机内参（归一化）
        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        self.full_res_shape = (1024, 1280)
        self.inpaint_pseudo_gt_dir = None  # 单视图数据暂不支持inpainting
    
    def check_depth(self):
        """检查是否有深度图"""
        return False
    
    def get_color(self, folder, frame_index, side, do_flip, do_rot):
        """加载颜色图像
        
        Args:
            folder: 文件夹路径（如 "dataset2/keyframe2"）
            frame_index: 帧编号
            side: 视图侧（单视图数据中忽略此参数，但保留以兼容基类接口）
            do_flip: 是否翻转
            do_rot: 是否旋转（单视图数据中通常不使用）
        """
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
        # 单视图数据通常不需要旋转增强（内窥镜图像方向固定）
        # 如果需要可以取消注释：
        # if do_rot:
        #     angle = np.random.choice([pil.ROTATE_90, pil.ROTATE_180, pil.ROTATE_270])
        #     color = color.transpose(angle)
        
        return color
    
    def get_image_path(self, folder, frame_index, side):
        """获取图像路径
        
        单视图数据：图像直接放在序列文件夹下，不需要left_img/right_img子目录
        
        Args:
            folder: 文件夹路径（如 "dataset2/keyframe2"）
            frame_index: 帧编号
            side: 视图侧（单视图数据中忽略）
        """
        # 使用10位数字格式（例如：0000000626.png）
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        
        # 单视图：图像直接放在folder下，不需要left_img子目录
        image_path = os.path.join(self.data_path, folder, f_str)
        
        return image_path
    
    def get_depth(self, folder, frame_index, side, do_flip):
        """加载深度图（如果可用）"""
        # 单视图数据通常没有深度图
        return None

