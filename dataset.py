
import cv2
import annt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF 

import utils

class PaperDataset(Dataset):

    def __init__(self, annotation_dir, width=512, height=512, shrink_rate=1, transform=None):
        super(PaperDataset, self).__init__()
        # Data
        self.images = []
        self.heatmaps = []

        # Configure
        self.width = width
        self.height = height
        self.num_class = 4
        self.transform = transform

        to_tensor = ToTensor()

        # Preprocess
        raw_files = list(annt.load(annotation_dir))
        for item in raw_files:
            item = item.resize(self.width, self.height)
            image = item.image
            heatmap = create_heatmap((self.width, self.height, self.num_class), item.boxes)
            # assert image.shape == (3, self.height, self.width)
            # assert heatmap.shape == (self.num_class, self.height, self.width)

            image = to_tensor(image).to(torch.float32)
            heatmap = to_tensor(heatmap).to(torch.float32)

            self.images.append(image)
            self.heatmaps.append(heatmap)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        heatmap = self.heatmaps[idx]
        obj = {
            'image': image,
            'heatmap': heatmap,
        }
        if self.transform:
            obj = self.transform(obj)
        return obj

    def show(self, idx):
        pass

def create_heatmap(shape, polygons, mask="gaussian", mask_radius=25):
    height, width, channels = shape
    heatmap = np.zeros(shape=shape, dtype=np.float16)

    msize = mask_radius*2
    mask = gaussian_mask(msize, msize, 10)

    for polygon in polygons:
        assert len(polygon.points) == channels
        for idx, point in enumerate(polygon.points):
            x, y = [int(p) for p in point]
            # Compute point.
            left, right = min(x, mask_radius), min(width - x, mask_radius)
            top, bottom = min(y, mask_radius), min(height - y, mask_radius)
            cropped_heatmap = heatmap[y-top:y+bottom, x-left:x+right, idx]
            cropped_mask = mask[mask_radius-top:mask_radius+bottom, mask_radius-left:mask_radius+right]
            heatmap[y-top:y+bottom, x-left:x+right, idx] = np.maximum(cropped_heatmap, cropped_mask)
    return heatmap

class RandomRotate():
    def __init__(self, deg_range, seed=43):
        self.seed = seed
        self.deg_range = deg_range

    def __call__(self, sample):
        channels, height, width = sample['image'].size()
        r = np.random.uniform(-self.deg_range, self.deg_range)

        rot_image = TF.rotate(sample['image'], r, expand=True)
        rot_image = TF.resize(rot_image, [height, width])
        rot_heatmap = TF.rotate(sample['heatmap'], r, expand=True)
        rot_heatmap = TF.resize(rot_heatmap, [height, width])
        
        return {
            'image': rot_image,
            'heatmap': rot_heatmap
        }

class RandomScale():
    def __init__(self, min_scale, max_scale, seed=43):
        self.seed = seed
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        channels, height, width = sample['image'].size()
        r = np.random.uniform(self.min_scale, self.max_scale)

        rot_image = TF.affine(sample['image'], 0, [0, 0], r, [0, 0])
        rot_heatmap = TF.affine(sample['heatmap'], 0, [0, 0], r, [0, 0])
        
        return {
            'image': rot_image,
            'heatmap': rot_heatmap
        }

class RandomShear():
    def __init__(self, min_shear, max_shear, seed=43):
        self.seed = seed
        self.min_shear = min_shear
        self.max_shear = max_shear

    def __call__(self, sample):
        channels, height, width = sample['image'].size()
        r = np.random.uniform(self.min_shear, self.max_shear)

        shear_image = TF.affine(sample['image'], 0, [0, 0], 1, r)
        shear_heatmap = TF.affine(sample['heatmap'], 0, [0, 0], 1, r)
        
        return {
            'image': shear_image,
            'heatmap': shear_heatmap
        }


class RandomColor():
    def __init__(self, min_ratio, max_ratio, seed=43):
        self.seed = seed
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, sample):
        r_sat = np.random.uniform(self.min_ratio, self.max_ratio)
        r_hue = np.random.uniform(self.min_ratio-1, self.max_ratio-1)
        r_con = np.random.uniform(self.min_ratio, self.max_ratio)

        adj_image = sample['image']
        adj_image = TF.adjust_saturation(adj_image, r_sat)
        adj_image = TF.adjust_hue(adj_image, r_hue)
        adj_image = TF.adjust_contrast(adj_image, r_con)
        
        return {
            'image': adj_image,
            'heatmap': sample['heatmap']
        }

    
def gaussian_mask(width, height, sigma):
    width = (width-1) / 2
    height = (height-1) / 2
    x = np.arange(-width, width+1)
    y = np.arange(-height, height+1)
    gridx, gridy = np.meshgrid(x, y)
    h = np.exp(-(gridx*gridx+gridy*gridy) / (2.*sigma*sigma))
    # Fit to 1.
    h = h / h.max()
    return h


if __name__ == "__main__":
    path = "data/test"
    df = PaperDataset(path, 256, 256)

    for idx, item in enumerate(df):
        rot = RandomRotate(30)
        scale = RandomScale(0.9, 1.0)
        shear = RandomShear(-5, 5)
        color = RandomColor(0.9, 1.1)
        item = rot(item)
        item = scale(item)
        item = shear(item)
        item = color(item)
        heatmap = utils.visualize_heatmap(item['image'].numpy(), item['heatmap'].numpy())
        #cv2.imshow('heatmap', item['image'].numpy())
        cv2.imshow('heatmap', heatmap)
        cv2.waitKey()
        break

    corners = utils.get_corners(item['heatmap'])
    edges = utils.visiualize_edge(item['image'].numpy(), corners)
    cv2.imshow('edges', edges)
    cv2.waitKey()
    print(corners)

