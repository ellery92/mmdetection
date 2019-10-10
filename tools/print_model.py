from mmdet.apis import init_detector
import argparse
import torch
import torchsummary


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def print_model():
    args = parse_args()
    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))
    print(model)
    torchsummary.summary(model, input_size=(512, 512))


if __name__ == "__main__":
    print_model()
