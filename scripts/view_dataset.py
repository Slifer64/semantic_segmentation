import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import load_image, load_mask, apply_mask_to_image

def view_dataset(path):
    data_samples = [os.path.join(path, i) for i in list(sorted(os.listdir(path)))
                    if os.path.isdir(os.path.join(path, i))]            

    plt.ion()

    for sample in data_samples:
        rgb = load_image(sample)
        mask = load_mask(sample)
        mask = (mask>0) * 255

        # =====  Try different erosions mask  =====
        # mask0 = mask.copy().astype('uint8')
        # k = [0, 2, 4, 6, 10]

        # print('mask0:', mask0.shape, mask0.dtype)

        # masks = [cv2.erode(mask0, np.ones((i, i), "uint8"), iterations=1) for i in k]
        
        # masked_rgb = [np.zeros_like(rgb) for _ in range(len(masks))]
        # for i, mask in enumerate(masks):
        #     ind = mask > 0
        #     masked_rgb[i][ind] = rgb[ind]
        
        # n_masks = len(masked_rgb)
        # fig = plt.figure(figsize=(16, 5))
        # gs = gridspec.GridSpec(1, n_masks)
        # gs.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.
        # ax = [plt.subplot(gs[i]) for i in range(n_masks)]
        # for i, im in enumerate(masked_rgb):
        #     ax[i].axis('off')
        #     ax[i].imshow(im)
        #     ax[i].set_title('%d' % k[i])
        # plt.pause(0.001)
        # print("To continue press [enter]. To stop type 's' and press [enter]")
        # if input().lower() == 's': break
        # plt.close(fig)
        # continue

        masked_rgb = apply_mask_to_image(image=rgb, mask=mask)

        fig = plt.figure(figsize=(16, 5))
        gs = gridspec.GridSpec(1, 3)
        gs.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.
        ax = [plt.subplot(gs[i]) for i in range(3)]
        for ax_ in ax: ax_.axis('off')
        ax[0].imshow(rgb)
        ax[1].imshow(masked_rgb)
        ax[2].imshow(mask, cmap='gray')
        plt.pause(0.001)
        print("To continue press [enter]. To stop type 's' and press [enter]")
        if input().lower() == 's': break
        plt.close(fig)


# ==========================================
# =================== MAIN =================
# ==========================================

# Example usage:
# python3 view_dataset.py --dataset=data/grapebunch_dataset/


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', required=True, type=str,
                        help='The folder with the dataset')

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':

    args = parse_args()
    view_dataset(args['dataset'])
