
import cv2
import annt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF 

import utils

class PaperDataset(Dataset):

    def __init__(self, annotation_dir, width=512, height=512, shrink_rate=1, transform=None, R=8):
        """
        Args:
            annotation_dir: Dataset directory formatted . 
        """
        super(PaperDataset, self).__init__()
        # Data
        self.images = []
        self.heatmaps = []
        self.regmaps = []
        self.vecmaps = []

        # Configure
        self.width = width
        self.height = height
        self.num_class = 4
        self.transform = transform

        to_tensor = ToTensor()

        # Preprocess
        print("[*] Loading data...")
        raw_files = list(annt.load(annotation_dir))
        for item in raw_files:
            item = item.resize(self.width, self.height)
            image = item.image

            x_shrink = 1 / shrink_rate
            y_shrink = 1 / shrink_rate
            heatmap = create_heatmap((self.width, self.height), item.boxes, self.num_class, x_ratio=x_shrink, y_ratio=y_shrink)
            regmap = create_regmap((self.width, self.height), item.boxes, x_shrink, y_shrink, r=R)
            vecmap = create_vector((self.width, self.height), item.boxes, r=R) 

            image = to_tensor(image).to(torch.float32)
            heatmap = to_tensor(heatmap).to(torch.float32)
            regmap = to_tensor(regmap).to(torch.float32)
            vecmap = to_tensor(vecmap).to(torch.float32)

            self.images.append(image)
            self.heatmaps.append(heatmap)
            self.regmaps.append(regmap)
            self.vecmaps.append(vecmap)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        heatmap = self.heatmaps[idx]
        regmap = self.regmaps[idx]
        vecmap = self.vecmaps[idx]
        obj = {
            'image': image,
            'heatmap': heatmap,
            'regmap': regmap,
            'vecmap': vecmap,
        }
        if self.transform:
            obj = self.transform(obj)
        return obj

    def show(self, idx):
        pass

def create_heatmap(shape, polygons, class_num, mask="gaussian", mask_radius=25, x_ratio=1, y_ratio=1):
    """ Create heatmap from specfied polygons.
    Args:
        shape (tuple): Input image shape
        polygons (Polygon): Polygon data.
        class_num (number): Number of classes.
    """
    height, width = shape
    heatmap = np.zeros(shape=(height, width, class_num), dtype=np.float16)
    msize = mask_radius*2
    mask = gaussian_mask(msize, msize, 10)
    for polygon in polygons:
        assert len(polygon.points) == class_num
        for idx, point in enumerate(polygon.points):
            x, y = [int(p) for p in point]
            # Compute point.
            left, right = min(x, mask_radius), min(width - x, mask_radius)
            top, bottom = min(y, mask_radius), min(height - y, mask_radius)
            cropped_heatmap = heatmap[y-top:y+bottom, x-left:x+right, idx]
            cropped_mask = mask[mask_radius-top:mask_radius+bottom, mask_radius-left:mask_radius+right]
            heatmap[y-top:y+bottom, x-left:x+right, idx] = np.maximum(cropped_heatmap, cropped_mask)
    nw = int(width * x_ratio)
    nh = int(height * y_ratio)
    return heatmap

def create_regmap(shape, boxes, x_ratio=1, y_ratio=1, r=1):
    """ Create regmap from specfied polygons.
    Args:
        shape (tuple): Input image shape ordered by (H x W).
        boxes (Polygon): Polygon data.
        x_ratio (float): Shrink ratio along to x-axis.
        y_ratio (float): Shrink ratio along to y-axis.
        r: (int): Mask size to remain. Only pixels in r from center will be remain to output.
    Returns:
        np.ndarray: Regmap which shapes is (H', W', 2)
    """
    px = shape[1]
    py = shape[0]
    nx = int(px * x_ratio)
    ny = int(py * y_ratio)
    regmap = np.zeros(shape=(ny, nx, 2), dtype=np.float16)

    x_array = np.arange(0, nx)
    y_array = np.arange(0, ny)
    gridx, gridy = np.meshgrid(x_array, y_array)
    for polygon in boxes:
        for idx, point in enumerate(polygon.points):
            x, y = point
            x = x * x_ratio
            y = y * y_ratio
            gx = x - gridx
            gy = y - gridy
            rad = (gx * gx + gy * gy) < r * r
            gx = gx * rad
            gy = gy * rad
            regmap[:, :, 0] += gx
            regmap[:, :, 1] += gy
    return regmap


def create_vector(shape, boxes, x_ratio=1, y_ratio=1, r=1):
    """ Create long-span vector features from polygons.

    Warning:
        Shrink function is not implemented!
    """
    px = shape[1]
    py = shape[0]
    nx = int(px * x_ratio)
    ny = int(py * y_ratio)
    vecmap = np.zeros(shape=(ny, nx, 2), dtype=np.float16)

    x_array = np.arange(0, nx)
    y_array = np.arange(0, ny)
    gridx, gridy = np.meshgrid(x_array, y_array)

    for polygon in boxes:
        points = polygon.points
        for i0 in range(len(points)):
            i1 = (i0 + 1) % len(points)
            x0 = points[i0][0]
            y0 = points[i0][1]
            vx = (points[i1][0] - points[i0][0]) / nx
            vy = (points[i1][1] - points[i0][1]) / ny
            
            gx = gridx - x0
            gy = gridy - y0
            mask = (gx * gx + gy * gy) < r * r
            vecmap[:, :, 0] += mask * vx
            vecmap[:, :, 1] += mask * vy
    return vecmap


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
            'heatmap': sample['heatmap'],
            'regmap': sample['regmap'],
            'vecmap': sample['vecmap']
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

