import copy
import platform
import random
from functools import partial

import numpy as np
import torch
import math
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader, DistributedSampler
import cv2
import mmcv


import json
import base64
from mmdet.datasets import PIPELINES
import os 
import os.path as osp
from io import BytesIO
from PIL import Image


########### 이미지가 base64 형식으로 json 파일에 있음으로 이를 받도록 수정진행 

@PIPELINES.register_module()
class CUSTOM_LoadImageFromFile:
    """Load an image from file.
    병변검출 DB SET에 맞춘 load 
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        
        #file name이 json 파일이고 json 파일의 'imageData'를  decode해야함 
        #img_bytes = self.file_client.get(filename) ### 기존 
        
        #json load 
        with open(filename, "r") as json_file:
                json_data = json.load(json_file)
        
        # load 이후 decoding 
        img_bytes = base64.b64decode(json_data['imageData'])
        
        #여기서 부터는 기존 소스 
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type,backend='pillow')
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


### 사람의 bodyparts는 붙어있기 때문에 flip되면 위치가 바뀌어야함
### pair로 두고 뒤집는 걸로 해결하기 
@PIPELINES.register_module()
class CustomRandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """
    def __init__(self, prob=None, direction='horizontal',flip_pair=None):
        self.prob = prob
        self.direction = direction
        self.flip_pair = flip_pair
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        th_val = random.random()
        flip = True if th_val < self.prob else False
        results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # mmcv.imwrite(results['img'],'ori.png')
            # mmcv.imwrite(results['gt_semantic_seg'],'ori_label.png')
            # flip image
            results['img'] = mmcv.imflip(
                results['img'], direction=results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction']).copy()
                # mmcv.imwrite(results['gt_semantic_seg'],'ori_label_flip.png')
                if self.flip_pair != None:
                    flip_gt = np.zeros_like(results[key])
                    for pair in self.flip_pair:
                        index1 = np.where(results[key]==pair[0])
                        flip_gt[index1] = pair[1]
                        
                        index2 = np.where(results[key]==pair[1])
                        flip_gt[index2] = pair[0]
                        
                        # print(pair[1])
                        # print(pair[0])
                        
                    index_body = np.where(results[key]==1)
                    flip_gt[index_body] = 1
                    
                    index_head = np.where(results[key]==14)
                    flip_gt[index_head] = 14
                        
                    results[key] = flip_gt
                        
            # mmcv.imwrite(results['img'],'t.png')
            # mmcv.imwrite(results['gt_semantic_seg'],'tt.png')
        
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'
