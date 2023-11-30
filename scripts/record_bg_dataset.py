import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from my_pkg.camera import *
from utils import save_image

plt.ion()


def record_bg_dataset(log_path, count=1):

    # if os.path.exists(log_path): shutil.rmtree(log_path)
    if not os.path.exists(log_path): os.makedirs(log_path)

    camera = RsCamera()

    while True:
        rgb_img = camera.get_rgb()

        cv2.imshow("Output1", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        # Wait until user press some key
        key = cv2.waitKey(30)
        if key == ord('q'):
            print('\33[1m\33[33mTerminating...\33[0m')
            break
        elif key == ord('r'):
            print('\33[1m\33[32mRecorded sample %d...\33[0m' % count)
            save_image(image=rgb_img, path=log_path, filename=f"bg_{str(count).zfill(4)}.png")
            count += 1


# ==========================================
# =================== MAIN =================
# ==========================================

# Example usage:
# python3 record_bg_dataset.py --save_to=data/bg_dataset/ --count=1


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--save_to', required=True, type=str,
                        help='The folder where to save the dataset.')
    parser.add_argument('--count', default=1, type=int,
                        help='The number of the first sample. Set > 1 to add more data to an existing dataset folder.')

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':

    args = parse_args()
    record_bg_dataset(args['save_to'], args['count'])

