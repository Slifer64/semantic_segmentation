import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import load_image, load_mask, save_image, save_mask, apply_mask_to_image, SegConfig

def erode_dataset(path: str, kernel: int, iters: int, viz):

    data_samples = [os.path.join(path, i) for i in list(sorted(os.listdir(path)))
                    if os.path.isdir(os.path.join(path, i))]

    id = SegConfig.load(path).get_ids()[0]

    plt.ion()

    for sample in data_samples:
        rgb = load_image(sample)
        mask = load_mask(sample)
        mask = ((mask>0) * 255).astype('uint8')

        # =====  overwrite dataset  ======
        mask = cv2.erode(mask, np.ones((kernel, kernel), "uint8"), iterations=iters)
        # save_image(image=rgb, path=sample)
        
        if viz:
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

        save_mask(mask=np.where(mask == 255, id, 0).astype(int), path=sample)


# ==========================================
# =================== MAIN =================
# ==========================================

# Example usage:
# python3 view_dataset.py --dataset=data/grapebunch_dataset/


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', required=True, type=str,
                        help='The folder with the dataset.')
    parser.add_argument('--kernel', required=True, type=int,
                        help='The size of the erosion kernel.')
    parser.add_argument('--iters', default=1, type=int,
                        help='The number of times to apply the erosion kernel.')
    parser.add_argument('--viz', default=0, type=int,
                        help='Whether to vizualize the process.')

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':

    args = parse_args()
    erode_dataset(path=args['dataset'], kernel=args['kernel'], iters=args['iters'], viz=args['viz'])
