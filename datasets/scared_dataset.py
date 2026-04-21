from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset import MonoDataset


class SCAREDDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1280, 1024)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        
        return False

    def get_color(self, folder, frame_index, side, do_flip, do_rot=None):
        """加载颜色图像
        
        Args:
            folder: 文件夹路径
            frame_index: 帧编号
            side: 视图侧
            do_flip: 是否翻转
            do_rot: 是否旋转（可选，兼容接口）
        """
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
        # SCARED数据集通常不使用旋转增强
        # if do_rot:
        #     ...

        return color


class SCAREDRAWDataset(SCAREDDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        # 使用10位数字格式（例如：0000000626.png）
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        # 图片直接在keyframe文件夹下，没有left_img子文件夹
        image_path = os.path.join(
            self.data_path, folder, f_str)

        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "scene_points{:06d}.tiff".format(frame_index)

        depth_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data/groundtruth".format(self.side_map[side]),
            f_str)

        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = depth_gt[:, :, 0]
        depth_gt = depth_gt[0:1024, :]
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


