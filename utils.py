
import cv2
import numpy as np
import torch
from torch import nn
from collections import deque

def visualize_heatmap(input_image, pred_result):
    """
    Args:
        input_image (np.ndarray): CHW formatted image.
        pred_result (np.ndarray): CHW formatted heatmap prediction.
    """
    input_image = input_image.copy()
    channel, height, width = pred_result.shape
    hmap = np.zeros((height, width, 3))
    for c in range(channel):
        source = pred_result[c, :, :] * 255
        source = source.astype(np.uint8) 
        cmap = cv2.applyColorMap(source, cv2.COLORMAP_JET)
        hmap = np.maximum(hmap, cmap)

    hmap = hmap.astype(np.uint8)
    input_image *= 255
    input_image = input_image.transpose([1, 2, 0]).astype(np.uint8)
    assert input_image.shape == hmap.shape, "Different size error"
    add = cv2.addWeighted(input_image, 0.3, hmap, 0.7, 0)
    return add

def visiualize_edge(input_image, corners, torch_order=True):
    image = input_image.copy() * 255
    if torch_order:
        image = image.transpose([1, 2, 0]).astype(np.float32)
    image = cv2.UMat(image)

    assert len(corners) == 4
    for i in range(4):
        nxt_idx = (i+1) % 4
        image = cv2.line(image, corners[i], corners[nxt_idx], (0, 255, 0), 2)
    image = image.get().astype(np.uint8)
    return image

def get_corners_by_heatmap(scores, regmaps=None, K=10):
    """
    Args:
        scores (torch.Tensor)
    """
    channel, height, width = scores.size()
    # Preprocess
    # socres = nms(scores, kernel_size=5)

    x_cands = []
    y_cands = []
    # List top-k scored positions.
    for i in range(channel):
        heatmap = scores[i, :, :]
        topk_scores, topk_inds = torch.topk(heatmap.view(-1), k=K)
        topk_y = (topk_inds // width).float()
        topk_x = (topk_inds % width).float()
        y_cands.append(topk_y)
        x_cands.append(topk_x)

    queue = deque()
    initial = (0, 0, 0, 0)
    queue.append(initial)
    visit = set()

    while len(queue) > 0:
        front = queue.popleft()
        if not all([f < K for f in front]) or front in visit:
            continue
        visit.add(front)

        corners = []
        for c in range(channel):
            idx = front[c]
            x = int(x_cands[c][idx].item())
            y = int(y_cands[c][idx].item())
            if regmaps is not None:
                dx = regmaps[0, y, x].item()
                dy = regmaps[1, y, x].item()
                x += dx
                y += dy
            corners.append((x, y))

        # If corners are valid rectangle, return corners 
        if is_valid_rectangle(corners):
            return corners
        for c in range(channel):
            nxt = list(front)
            nxt[c] += 1
            queue.append(tuple(nxt))
    return [(0, 0), (0, 0), (0, 0), (0, 0)]

def get_corners_by_vector(heatmap, regmap, vecmap, K=10):
    """ Detect paper corners by vector based method.
    """
    channel, height, width = heatmap.shape

    x_cands = []
    y_cands = []
    # List top-k scored positions.
    for i in range(channel):
        hi = heatmap[i, :, :]
        topk_scores, topk_inds = torch.topk(hi.view(-1), k=K)
        topk_y = (topk_inds // width).float()
        topk_x = (topk_inds % width).float()
        y_cands.append(topk_y)
        x_cands.append(topk_x)

    for i in range(K):
        x = int(x_cands[0][i].item())
        y = int(y_cands[0][i].item())
        dx = regmap[0][y][x].item()
        dy = regmap[1][y][x].item()

        corners = [(x+dx, y+dy)]
        prob = heatmap[0][y][x].item()
        for i in range(3):
            rx = x + vecmap[0][y][x].item() * width # Raw next x
            ry = y + vecmap[1][y][x].item() * height # Raw next y
            cx = int(np.clip(rx, 0, width-1)) # Clipped rx
            cy = int(np.clip(ry, 0, height-1)) # Clipped ry
            dx = regmap[0][cy][cx].item()
            dy = regmap[1][cy][cx].item()

            # Update coodinate by offset
            rx = rx + dx
            ry = ry + dy
            x = int(np.clip(rx, 0, width-1)) # Clipped rx
            y = int(np.clip(ry, 0, height-1)) # Clipped ry
            corners.append((rx, ry))
            prob += heatmap[i+1, y, x]
        prob /= 4
        return corners
    return None

def is_valid_rectangle(corners):
    for i0 in range(len(corners)):
        i1 = (i0 + 1) % len(corners)
        i2 = (i0 + 2) % len(corners)
        v1 = (corners[i1][0] - corners[i0][0], corners[i1][1] - corners[i0][1])
        v2 = (corners[i2][0] - corners[i1][0], corners[i2][1] - corners[i1][1])
        cp = v1[0] * v2[1] - v1[1] * v2[0]
        if cp > 0:
            return False
    return True

def nms(pred_result, kernel_size=3):
    padding = (kernel_size - 1) // 2
    suppress = nn.functional.max_pool2d(pred_result, (kernel_size, kernel_size), stride=1, padding=padding)
    keep = (pred_result == suppress).float()
    return pred_result * keep

