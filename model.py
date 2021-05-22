
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import utils


class PaperNet(nn.Module):
    def __init__(self):
        super(PaperNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
        )
        self.layer2 = Hourglass(3, 32, 2)

        self.hmap = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.regs = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x = self.feature_extractor(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out = [self.hmap(x), self.regs(x)]
        return out


class Residual(nn.Module):

    """ Bottleneck Residual Module
    """

    def __init__(self, inp_dims, out_dims):
        super(Residual, self).__init__()
        bottleneck_size = out_dims//2
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dims)
        self.conv1 = nn.Conv2d(inp_dims, bottleneck_size, 1)
        self.bn2 = nn.BatchNorm2d(bottleneck_size)
        self.conv2 = nn.Conv2d(bottleneck_size, bottleneck_size, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(bottleneck_size)
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

    def __init__(self, depth, channel_size, increase_ratio=1):
        super(Hourglass, self).__init__()
        self.up1 = Residual(channel_size, channel_size)
        self.pool = nn.MaxPool2d(2, 2)

        # Recursive Hourglass Layer
        next_channel = channel_size * increase_ratio
        self.low1 = Residual(channel_size, next_channel)
        if depth > 1:
            self.low2 = Hourglass(depth-1, next_channel)
        else:
            self.low2 = None
        self.low3 = Residual(next_channel, channel_size)
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

    def __init__(self, log_dir, lr, log_interval, lambda1):
        self.lr = lr
        self.log_interval = log_interval
        self.lambda1 = lambda1

        # Setup model.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PaperNet().to(self.device)

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
            hmap, regs = outputs

            self.optimizer.zero_grad()
            loss = self.loss_fn(hmap, batch["heatmap"])
            loss.backward()
            self.optimizer.step()

            if (batch_idx+1) % self.log_interval == 0:
                # Log
                print(f'{batch_idx+1}/{len(train_dl)} - train loss : {loss}')
        if val_dl:
            self.test(epoch, val_dl)


    def test(self, global_step, val_dl):
        size = len(val_dl)
        loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(val_dl):
                hmap_pred, regs_pred = self.model(batch['image'])
                loss += self.loss_fn(hmap_pred, batch["heatmap"])
                # Log to tensorboard
                if idx == 0:
                    image_np = batch['image'].numpy()
                    hmap_np = hmap_pred.numpy()

                    superimposed = utils.visualize_heatmap(image_np[0], hmap_np[0])
                    superimposed = superimposed.transpose([2, 0, 1]) / 255.
                    self.writer.add_image(self.TAG_HEATMAP, superimposed, global_step)

                    corners = utils.get_corners(hmap_pred[0], 10)
                    edges = utils.visiualize_edge(image_np[0], corners)
                    edges = edges.transpose([2, 0, 1]) / 255.
                    self.writer.add_image(self.TAG_EDGES, edges, global_step)
        loss /= size
        print(f"Validation Error: Avg loss: {loss:>8f} \n")

    def checkpoint(self, output_path, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, output_path)

    def restore(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        # self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        epoch = ckpt['epoch']
        return epoch

    def to_onnx(self, output_dir):
        dummy_input = torch.randn(1, 3, 256, 256)
        input_names = ["inputs"]
        output_names = ["heatmap", "regs"]
        torch.onnx.export(self.model, dummy_input, output_dir, input_names=input_names, output_names=output_names, opset_version=11)

def loss_fn(y_pred, y_true):
    return focal_loss(y_pred, y_true, 2, 4)
    

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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PaperNet().to(device)