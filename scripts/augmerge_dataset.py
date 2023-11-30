import cv2
import numpy as np
import os
import shutil
import torch
import torchvision.transforms.functional as torchvision_F
import torchvision.transforms as torchvision_T
import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import load_image, load_mask, save_image, save_mask, SegConfig


def draw_uniform(range, dtype='float'):
    value = float(torch.empty(1).uniform_(range[0], range[1]).item())
    if dtype == 'int':
        value = int(round(value))
    return value


def merge_and_augment(datasets, items_range, bg_path, augmented_data_path, iters, seed, viz=False):

    torch.manual_seed(seed)
    np.random.seed(seed)

    plt.ion()

    # =======  create augmented data directory  =======
    if os.path.exists(augmented_data_path):
        shutil.rmtree(augmented_data_path)
    os.makedirs(augmented_data_path)

    count = 1

    # ========  random transforms, value ranges  ========

    ALLOW_OVERLAP = False # whether to allow overlapping between object instances of the same class
    MAX_N_OVERLAP = 3 # if ALLOW_OVERLAP==False, maximum attempts to add an object without overlapping, before skipping it

    # object transforms: applied to each object instance separately
    angle_range = [-80, 80]
    tx_range = [-180, 180]
    ty_range = [-250, 250]
    scale_range = [0.8, 1.5]
    p_hflip = 0.5

    ## Color jitter applies in the final mixed image, 
    ## so it may be too much to also apply separately color jitter to the bg.
    # bg_color_jit = {'brightness': 0.2, 'contrast': 0.1, 'saturation': 0.1, 'hue': 0.05}

    # global transforms: applied on the final mixed image (with different objects and background)
    color_jit = {'brightness': [0.8, 1.6], 'contrast': [0.85, 1.2], 'saturation': [0.75, 1.4], 'hue': 0.04}
    perspective_tf = {'p': 0.0, 'distortion': 0.5} # probablity of applying distortion and distortion scale
    p_noise = 0.25 # probability of adding Gaussian noise
    gaussian_noise = {'mean': 0.0, 'std': 0.1}

    # data samples path
    object_samples = [[os.path.join(obj_path, fname) for fname in list(sorted(os.listdir(obj_path))) 
                       if os.path.isdir(os.path.join(obj_path, fname))] 
                       for obj_path in datasets]
    bg_samples = [os.path.join(bg_path, fname) for fname in list(sorted(os.listdir(bg_path)))]

    cfg_dict = [SegConfig.load(obj_path) for obj_path in datasets]
    obj_ids = [cfg.get_ids()[0] for cfg in cfg_dict]
    obj_labels = [cfg.get_labels()[0] for cfg in cfg_dict]

    SegConfig({label:id for label, id in zip(obj_labels, obj_ids)}).save(augmented_data_path)

    # ======  Generate new random scenes 'iters' times  ======
    for _ in tqdm.tqdm(range(iters)):
        
        # load randomly one background image
        i_bg = np.random.randint(low=0, high=len(bg_samples))
        bg_im = torchvision_F.to_tensor(cv2.cvtColor(cv2.imread(bg_samples[i_bg]), cv2.COLOR_BGR2RGB))

        # apply random transform to background
        # im = torchvision_T.ColorJitter(**bg_color_jit)(bg_im)
        im = bg_im
        if (torch.rand(1) < p_noise):
            im = im + torch.randn(im.size()) * gaussian_noise['std'] + gaussian_noise['mean']

        # generate random number 'n_i' for object-i within items_range[i]
        n_items = [np.random.randint(low=items_range[i][0], high=items_range[i][1]+1) for i in range(len(items_range))]
        # pick randomly 'n_i' indices from each object dataset
        obj_ind = [np.random.choice(len(object_samples[i]), size=n_items[i], replace=False) for i in range(len(n_items))]
        # collect all selected object paths in a list and shuffle them
        obj_samples_k = [(object_samples[i][j], obj_ids[i]) for i, ind in enumerate(obj_ind) for j in ind]
        obj_samples_k = [obj_samples_k[i] for i in np.random.permutation(len(obj_samples_k))]

        # print('n_items:', n_items)
        # print('obj_ind:', obj_ind)

        final_mask = torch.zeros(im.shape[-2:], dtype=int)
        ones_mask = torch.ones_like(final_mask)

        # ======  loop through the objects and put them on the image  ======
        for obj_path, obj_id in obj_samples_k:

            # load the object image and its mask
            rgb0 = torchvision_F.to_tensor(load_image(obj_path))
            mask0 = torch.as_tensor(load_mask(obj_path))[None, ...]  # add 1 channel

            n_overlap = 0
            add_obj = False
            while n_overlap <= MAX_N_OVERLAP:

                # ======  apply random transforms  ======
                angle = draw_uniform(angle_range)
                tx = draw_uniform(tx_range, dtype='int')
                ty = draw_uniform(ty_range, dtype='int')
                scale = draw_uniform(scale_range)
                hflip_flag = torch.rand(1) < p_hflip

                rgb = torchvision_F.affine(img=rgb0, angle=angle, translate=[tx, ty], scale=scale, shear=0, fill=0,
                                            interpolation=torchvision_T.InterpolationMode.NEAREST)
                mask = torchvision_F.affine(img=mask0, angle=angle, translate=[tx, ty], scale=scale, shear=0, fill=0,
                                            interpolation=torchvision_T.InterpolationMode.NEAREST)
                if hflip_flag:
                    rgb = torchvision_F.hflip(rgb)
                    mask = torchvision_F.hflip(mask)

                # ======  calc the image indices where the object is  ======
                mask_idx = (mask > 0)[0] # remove batch dimension

                # check if it overlaps with previous mask instance of the same object
                if ALLOW_OVERLAP or not torch.any(final_mask[mask_idx] == obj_id).item():
                    add_obj = True
                    break

                n_overlap += 1
                # print('overlap count:', n_overlap)


            if not add_obj: continue

            final_mask[mask_idx] = obj_id*ones_mask[mask_idx]

            # put transformed object in the background image
            im[:, mask_idx] = rgb[:, mask_idx]

        # apply transforms to the final image
        im = torchvision_T.ColorJitter(**color_jit)(im)
        im = torch.clip(im, 0, 1)  # clip RGB values in [0, 1]

        if torch.rand(1) < perspective_tf['p']:
            startpoints, endpoints = torchvision_T.RandomPerspective.get_params(height=im.shape[-2], width=im.shape[-1], distortion_scale=perspective_tf['distortion'])
            im = torchvision_F.perspective(im, startpoints, endpoints, torchvision_T.InterpolationMode.NEAREST, fill=1.0)
            final_mask = torchvision_F.perspective(final_mask[None, ...], startpoints, endpoints, torchvision_T.InterpolationMode.NEAREST, fill=0)[0]
        # im = torchvision_T.RandomPerspective(distortion_scale=perspective_tf['distortion'], p=perspective_tf['p'])(im)

        # convert to numpy
        im = im.permute(1, 2, 0).numpy()
        im_mask = final_mask.numpy()  # remove channel dimension

        # visualize
        if viz:
            # im_viz = np.hstack([rgb0.permute(1, 2, 0).numpy(), im])
            im_viz = im

            fig = plt.figure(figsize=(13, 5))
            gs = gridspec.GridSpec(1, 2)
            gs.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.
            ax = [plt.subplot(gs[i]) for i in range(2)]
            for ax_ in ax: ax_.axis('off')
            ax[0].imshow(im_viz)
            ax[1].imshow(im_mask) # , cmap='gray')
            plt.pause(0.001)
            print("To continue press [enter]. To stop type 'q' and press [enter]")
            if input().lower() == 'q' : return
            plt.close(fig)

        # save image and mask
        file_path = f"{augmented_data_path}/{str(count).zfill(4)}/"
        if os.path.exists(file_path): shutil.rmtree(file_path)
        os.makedirs(file_path)
        
        save_image(image=im*255, path=file_path)
        save_mask(mask=im_mask, path=file_path)

        count += 1


# ==========================================
# =================== MAIN =================
# ==========================================

# Example usage:
# python3 augmerge_dataset.py --datasets data/grapebunch_dataset/ data/leaves_dataset/ --items_range 1:3 2:10 --bg_dataset=data/background_dataset/ --save_to=data/grape_leaves_dataset/ --iters=20 --seed=0 --viz=0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--datasets', required=True, type=str, nargs='+',
                        help='The datasets paths separated by white spaces')
    parser.add_argument('--items_range', required=True, type=str, nargs='+',
                        help='The min/max number of items to generate in each scene')
    parser.add_argument('--bg_dataset', required=True, type=str,
                        help='The folder with the background dataset')
    parser.add_argument('--save_to', required=True, type=str,
                        help='The folder where to save a dataset. Applies to ("record", "augment")')
    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for all random operations')
    parser.add_argument('--iters', default=2, type=int,
                        help='Iters to loop through all data for data augmentation')
    parser.add_argument('--viz', default=0, type=int,
                        help='The name of the segmentation model')

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':

    args = parse_args()
    items_range = [list(map(int, i_range.split(':'))) for i_range in args['items_range']]
    merge_and_augment(datasets=args['datasets'], items_range=items_range,
                      bg_path=args['bg_dataset'], augmented_data_path=args['save_to'], iters=args['iters'],
                      seed=args['seed'], viz=args['viz'])
