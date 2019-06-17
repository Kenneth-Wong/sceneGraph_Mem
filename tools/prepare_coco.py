import _init_paths
import os.path as osp
import numpy as np
import math
import json
from model.config import cfg
from scipy.misc import imread, imresize
import h5py


class COCOLoader:
    def __init__(self, split):
        self._split = split
        self._loadImgInfo()

    def _loadImgInfo(self):
        with open(osp.join(cfg.COCO_DIR, 'mscoco_'+self._split+'2017.json'), 'r') as f:
            info = json.load(f)
        self._img_names = [key for key in info['data']]
        self._img_paths = [osp.join(cfg.COCO_DIR, self._split+'2017', name) for name in self._img_names]
        self._num_img = len(self._img_names)

    def getImdb(self, lim_num=-1):
        h5_file = h5py.File(osp.join(cfg.COCO_DIR, 'imdb_' + self._split + '_1024.h5'), 'w')
        num_images = self._num_img if lim_num <= 0 else lim_num
        shape = (num_images, 3, 1024, 1024)
        image_dset = h5_file.create_dataset('images', shape, dtype=np.uint8)
        image_names = h5_file.create_dataset('image_names', (num_images,), dtype=h5py.special_dtype(vlen=str))
        original_heights = np.zeros(num_images, dtype=np.int32)
        original_widths = np.zeros(num_images, dtype=np.int32)
        image_heights = np.zeros(num_images, dtype=np.int32)
        image_widths = np.zeros(num_images, dtype=np.int32)
        for i in range(num_images):
            im = imread(self._img_paths[i])
            if i % 1000 == 0:
                print(i, self._img_paths[i])
            # handle grayscale
            if im.ndim == 2:
                im = im[:, :, None][:, :, [0, 0, 0]]
            ih = im.shape[0]
            iw = im.shape[1]
            scale = 1024. / max(ih, iw)
            im = imresize(im, scale)
            H, W = im.shape[0], im.shape[1]
            # swap to bgr
            r = im[:, :, 0].copy()
            im[:, :, 0] = im[:, :, 2]
            im[:, :, 2] = r
            original_heights[i] = ih
            original_widths[i] = iw
            image_heights[i] = H
            image_widths[i] = W
            image_dset[i, :, :H, :W] = im.transpose(2, 0, 1)
            image_names[i] = self._img_names[i]
        h5_file.create_dataset('image_heights', data=image_heights)
        h5_file.create_dataset('image_widths', data=image_widths)
        h5_file.create_dataset('original_heights', data=original_heights)
        h5_file.create_dataset('original_widths', data=original_widths)

if __name__ == '__main__':
    loader = COCOLoader('train')
    loader.getImdb()




