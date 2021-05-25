
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Lambda, Compose, Resize
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import utils


class PaperNet(nn.Module):
    def __init__(self, dropout_rate):
        super(PaperNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
        )
        self.layer2 = Hourglass(4, 32, 1.5, dropout_rate=dropout_rate)

        self.hmap = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.regs = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, padding=1)
        )
        self.vecs = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x = self.feature_extractor(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out = [self.hmap(x), self.regs(x), self.vecs(x)]
        return out


class Residual(nn.Module):

    """ Bottleneck Residual Module
    """

    def __init__(self, inp_dims, out_dims, dropout_rate=0.3):
        super(Residual, self).__init__()
        bottleneck_size = out_dims//2
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dims)
        self.conv1 = nn.Conv2d(inp_dims, bottleneck_size, 1)
        self.bn2 = nn.BatchNorm2d(bottleneck_size)
        self.conv2 = nn.Conv2d(bottleneck_size, bottleneck_size, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(bottleneck_size)
        self.dr = nn.Dropout2d(dropout_rate)
        self.conv3 = nn.Conv2d(bottleneck_size, out_dims, 1)

        if inp_dims == out_dims:
            self.skip_layer = None
        else:
            self.skip_layer = nn.Conv2d(inp_dims, out_dims, 1)

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dr(x)
        x = self.conv3(x)

        if self.skip_layer is None:
            x += residual
        else:
            x += self.skip_layer(residual)
        return x

class Hourglass(nn.Module):
    """ Hourglass Module
    https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/models/layers.py
    """

    def __init__(self, depth, channel_size, increase_ratio=1, dropout_rate=0.3):
        super(Hourglass, self).__init__()
        self.up1 = Residual(channel_size, channel_size, dropout_rate=dropout_rate)
        self.pool = nn.MaxPool2d(2, 2)

        # Recursive Hourglass Layer
        next_channel = int(channel_size * increase_ratio)
        self.low1 = Residual(channel_size, next_channel, dropout_rate=dropout_rate)
        if depth > 1:
            self.low2 = Hourglass(depth-1, next_channel, dropout_rate=dropout_rate)
        else:
            self.low2 = None
        self.low3 = Residual(next_channel, channel_size, dropout_rate=dropout_rate)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        up1 = self.up1(x)
        x = self.pool(x)
        x = self.low1(x)
        if self.low2 is not None:
            x = self.low2(x)
        x = self.low3(x)
        x = self.upsample(x)
        return x + up1

class Trainer():

    TAG_EDGES = "EDGES"
    TAG_HEATMAP = "HEATMAP"

    def __init__(self, input_resolution, log_dir, lr, log_interval, lambda1, dropout_rate):
        self.input_resolution = input_resolution
        self.lr = lr
        self.log_interval = log_interval
        self.lambda1 = lambda1
        self.dropout_rate = dropout_rate

        # Constants
        self.focal_loss_alpha = 2
        self.focal_loss_beta = 4
        self.loss_weights = [1, 0.4, 150]

        # Setup model.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PaperNet(dropout_rate=dropout_rate).to(self.device)

        # Setup model components.
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.writer = SummaryWriter(f'{log_dir}')
        print(self.model)

    def train(self, epoch, train_dl, val_dl=None):
        """ Train model.
        Args:
            epoch (int): Epoch number.
            train_dl (torch.DataLoader): Train data loader.
            val_dl (torch.DataLoader): Validation data loader. If None, validation evaluation will be skip.
        """
        print('\n Epoch: %d\n ========================' % epoch)
        for batch_idx, batch in enumerate(train_dl):
            outputs = self.model(batch['image'])
            hmap, regs, vecs = outputs

            self.optimizer.zero_grad()
            f_loss = self.loss_weights[0] * focal_loss(hmap, batch["heatmap"], self.focal_loss_alpha, self.focal_loss_beta)
            r_loss = self.loss_weights[1] * reg_loss(regs, batch["regmap"])
            v_loss = self.loss_weights[2] * reg_loss(vecs, batch["vecmap"])

            loss = f_loss + r_loss + v_loss
            loss.backward()
            self.optimizer.step()

            if (batch_idx+1) % self.log_interval == 0:
                # Log
                print(f'{batch_idx+1}/{len(train_dl)} - train loss : {loss}')
        if val_dl:
            loss = self.test(epoch, val_dl)
            return loss
        return 0


    def test(self, global_step, val_dl):
        size = len(val_dl)
        f_loss = 0
        r_loss = 0
        v_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(val_dl):
                hmap_pred, regs_pred, vecs_pred = self.model(batch['image'])
                f_loss += self.loss_weights[0] * focal_loss(hmap_pred, batch["heatmap"], self.focal_loss_alpha, self.focal_loss_beta)
                r_loss += self.loss_weights[1] * reg_loss(regs_pred, batch["regmap"])
                v_loss += self.loss_weights[2] * reg_loss(vecs_pred, batch["vecmap"])

                # Log to tensorboard
                if idx == 0:
                    image_np = batch['image'].numpy()
                    hmap_np = hmap_pred.numpy()

                    superimposed = utils.visualize_heatmap(image_np[0], hmap_np[0])
                    superimposed = superimposed.transpose([2, 0, 1]) / 255.
                    self.writer.add_image(self.TAG_HEATMAP, superimposed, global_step)

                    corners = utils.get_corners(hmap_pred[0], K=10)
                    edges = utils.visiualize_edge(image_np[0], corners)
                    edges = edges.transpose([2, 0, 1]) / 255.
                    self.writer.add_image(self.TAG_EDGES, edges, global_step)

                    corners2 = utils.get_corners2(hmap_pred[0], regs_pred[0], vecs_pred[0])
                    corners2 = [(int(x), int(y)) for x ,y in corners2]
                    edges = utils.visiualize_edge(image_np[0], corners2)
                    edges = edges.transpose([2, 0, 1]) / 255.
                    self.writer.add_image("TEST_EDGES", edges, global_step)


        loss = (f_loss + r_loss + v_loss) / size
        print(f"Validation Error: Avg loss: {loss:>8f} Focal loss: {f_loss:>8f}, Reg loss: {r_loss:>8f}, Vec loss: {v_loss:>8f}\n")
        return loss

    def update_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def checkpoint(self, output_path, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, output_path)

    def restore(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt['epoch']
        return epoch

    def to_onnx(self, output_dir):
        dummy_input = torch.randn(1, 3, self.input_resolution[0], self.input_resolution[1])
        input_names = ["inputs"]
        output_names = ["heatmap", "regmap", "vecmap"]
        torch.onnx.export(self.model, dummy_input, output_dir, input_names=input_names, output_names=output_names, opset_version=11)

class Predictor:
    def __init__(self, input_width, input_height):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PaperNet().to(self.device)
        self.model.eval()
        self.input_width = input_width
        self.input_height = input_height
        self.resizer = Resize([input_height, input_width])

    def predict(self, image, method="vector"):
        """
        Args:
            image (np.ndarray): (H, W, C) orederd image.
        """
        w = image.shape[1]
        h = image.shape[0]
        rw = self.input_width / w
        rh = self.input_height / h

        # Predict
        image = image.copy()
        tensor = ToTensor()(image)
        tensor = tensor.to(torch.float32)
        tensor = self.resizer(tensor)
        tensor = tensor.unsqueeze(0)
        heatmap, regmap, vecmap = self.model(tensor)
        if method == "vector":
            corners = utils.get_corners_by_vector(heatmap[0], regmap[0], vecmap[0], K=10)
        elif method == "heatmap":
            corners = utils.get_corners_by_heatmap(heatmap[0], regmap[0], K=10)
        else:
            raise Exception(f"Unknown method {method}")

        # Fit corners to input image size.
        resized = []
        for c in corners:
            x = int(c[0] / rw)
            y = int(c[1] / rh)
            resized.append((x, y))
        return resized

    def restore(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model_state_dict'])


def loss_fn(y_pred, y_true, reg_pred, reg_true):
    return focal_loss(y_pred, y_true, 2, 4) + reg_loss(reg_pred, reg_true)
    

def focal_loss(y_pred, y_true, alpha, beta, eps=1e-5):
    pos = y_true.gt(0.99).float()
    neg = y_true.lt(1).float()
    pos_num = pos.sum()
    y_pred = torch.clamp(y_pred, eps, 1-eps)

    # Compute pos loss
    pos_coef1 = torch.pow(1 - y_pred, alpha)
    pos_loss = pos * pos_coef1 * torch.log(y_pred)

    # Compute neg loss
    neg_coef1 = torch.pow(1 - y_true, beta)
    neg_coef2 = torch.pow(y_pred, alpha)
    neg_loss = neg * neg_coef1 * neg_coef2 * torch.log(1 - y_pred)

    # Sum and average.
    loss = -1 * (neg_loss + pos_loss)
    assert pos_num.item() > 0, y_true.max().item()
    return loss.sum() / pos_num

def reg_loss(y_pred, y_true):
    pos = y_true.ne(0.).float()
    loss = torch.nn.SmoothL1Loss(reduction='none')(y_pred, y_true)
    loss = pos * loss
    loss = loss.sum() / pos.sum()
    return loss

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PaperNet().to(device)