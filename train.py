import argparse
import os
from torch.utils.data import DataLoader
import torchvision

import dataset
import model

def main():
    parser = argparse.ArgumentParser(description='Yaoyorozu PaperNet Trainer')

    #parser.add_argument('--root_dir', type=str, default='./')
    #parser.add_argument('--pretrain_name', type=str, default='pretrain')

    parser.add_argument('--model_name', type=str, default='yaoyorozu')
    parser.add_argument('--log_dir', type=str, default='runs/')
    parser.add_argument('--train_data', type=str, default='./data/train')
    parser.add_argument('--test_data', type=str, default='./data/test')
    parser.add_argument('--model_dir', type=str, default='models/')

    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--split_ratio', type=float, default=1.0)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_step', type=str, default='90,120')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--early_stopping', type=int, default=10)

    parser.add_argument('--test_topk', type=int, default=100)

    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--val_interval', type=int, default=5)
    cfg = parser.parse_args()

    transforms = torchvision.transforms.Compose([
        dataset.RandomRotate(20),
        dataset.RandomScale(0.9, 1.),
        dataset.RandomShear(-5, 5),
        dataset.RandomColor(0.9, 1.1),
    ])

    # Load dataset
    train_ds = dataset.PaperDataset(cfg.train_data, width=cfg.img_size, height=cfg.img_size, transform=transforms)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_ds = dataset.PaperDataset(cfg.test_data, width=cfg.img_size, height=cfg.img_size)
    val_dl = DataLoader(val_ds, batch_size=len(val_ds), shuffle=True)

    resolution = [cfg.img_size, cfg.img_size]
    trainer = model.Trainer(resolution, cfg.log_dir, lr=cfg.lr, log_interval=cfg.log_interval, lambda1=0.1)

    onnx_path = os.path.join(cfg.model_dir, f'{cfg.model_name}.onnx')
    ckpt_path = os.path.join(cfg.model_dir, f'{cfg.model_name}.torch')
    initial_epoch = 1

    if os.path.exists(ckpt_path):
        last_epoch = trainer.restore(ckpt_path)
        initial_epoch = last_epoch + 1

    best_loss = 9999999999
    last_update = initial_epoch
    for epoch in range(initial_epoch, cfg.num_epochs+1):
        loss = trainer.train(epoch, train_dl, val_dl)
        if loss < best_loss:
            best_loss = loss
            last_update = epoch
            # Save Model
            trainer.to_onnx(onnx_path)
            trainer.checkpoint(ckpt_path, epoch)

        if epoch - last_update >= cfg.early_stopping:
            print('Early stopping.')
            break


if __name__ == "__main__":
    main()