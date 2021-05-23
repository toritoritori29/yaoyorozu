
import os
import cv2
import argparse

import model
import utils

def main():
    parser = argparse.ArgumentParser(description='Yaoyorozu PaperNet Trainer')

    parser.add_argument('image_path', type=str)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--model_name', type=str, default='yaoyorozu')
    parser.add_argument('--model_dir', type=str, default='models/')
    cfg = parser.parse_args()

    ckpt_path = os.path.join(cfg.model_dir, f'{cfg.model_name}.torch')
    predictor = model.Predictor(cfg.img_size, cfg.img_size)
    predictor.restore(ckpt_path)

    try:
        image = cv2.imread(cfg.image_path)
    except Exception:
        print(f"Failed to read ${cfg.image_path}. Check whether file exists.")

    image = image / 255.
    corners = predictor.predict(image)
    print(corners)
    result = utils.visiualize_edge(image, corners, torch_order=False)
    cv2.imshow('Result', result)
    cv2.waitKey()

if __name__ == "__main__":
    main()

