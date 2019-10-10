import argparse

import cv2
import torch
import os
import time

from mmdet.apis import inference_detector, init_detector, show_result


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--video', type=str, required=True, help='video file')
    parser.add_argument(
        '--dst-dir', type=str, required=True, help='directory to save')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    video = cv2.VideoCapture(args.video)
    frame_count = 0

    if not os.path.exists(args.dst_dir):
        os.mkdir(args.dst_dir)

    total_time = 0
    while True:
        ret_val, img = video.read()
        if not ret_val:
            break
        frame_count += 1
        start = time.time()
        result = inference_detector(model, img)
        end = time.time()
        total_time += end - start

        result_img = show_result(img, result, model.CLASSES,
                                 score_thr=args.score_thr,
                                 wait_time=1, show=False)
        cv2.imwrite(os.path.join(args.dst_dir, str(frame_count) + ".jpg"),
                    result_img)
    avg_time = total_time/frame_count * 1000
    fps = 1 / avg_time * 1000
    print("Avg Time: {} ms, {} fps".format(avg_time, fps))


if __name__ == '__main__':
    main()
