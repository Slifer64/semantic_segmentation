import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from my_pkg.models import ResnetSegmentation
from my_pkg.camera import RsCamera, DatasetCamera
from utils import SegConfig, SegGui, draw_labeled_contours, filter_mask


def test_model_in_RT(model_filename, seg_cfg_path, camera_type):

    model = ResnetSegmentation.load(model_filename)
    model.eval()

    if camera_type == 'rs':
        camera = RsCamera()
    else:
        dataset = camera_type.split(':')[-1]
        camera = DatasetCamera(dataset)

    cfg = SegConfig.load(seg_cfg_path)
    class_ids = cfg.get_ids()

    # BGR triplets
    COLORS = ((255, 0, 255), # magenta
              (255, 255, 0), # cyan
              (0, 255, 255), # yellow
              (32, 177, 237), # mustard
            )

    plt.ion()

    gui = SegGui()
    gui.addParam('area_thres', 50, 1500)

    while True:
        rgb_img = camera.get_rgb()

        # add and then extract bacth dim
        with torch.no_grad():
            pred_class, prob = model.output(torchvision.transforms.ToTensor()(rgb_img)[None, ...], return_prob=True)

        pred_class = pred_class[0]
        prob = prob[0]

        # pred_class[prob < 0.9] = 0

        prob = (prob*255).numpy().astype('uint8')
        prob_map = cv2.applyColorMap(prob, cv2.COLORMAP_JET)

        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        img_viz = bgr_img.copy()
        masked_img = np.ones_like(bgr_img) * 255
        
        class_mask = []
        class_filt_mask = []
        class_masked_img = [np.ones_like(bgr_img) * 255 for _ in range(len(class_ids))]

        for k in range(len(class_ids)):
            id = class_ids[k]
            class_label = cfg.get_label(id)
            class_color = COLORS[k % len(COLORS)]

            mask = ((pred_class.numpy() == id) * 255).astype(np.uint8)


            filt_mask, contours = filter_mask(mask, area_thres=gui.get('area_thres'))

            draw_labeled_contours(img_viz, contours, color=class_color, label=class_label)

            idx = filt_mask > 0
            # idx = mask > 0
            masked_img[idx] = bgr_img[idx]
            class_masked_img[k][idx] = bgr_img[idx]
            class_mask.append(mask)
            class_filt_mask.append(filt_mask)


        mask_viz = np.vstack([np.hstack([mask, filt_mask]) for mask, filt_mask in zip(class_mask, class_filt_mask)])
        img_viz = np.hstack([img_viz, masked_img, prob_map])

        cv2.imshow("Output1", img_viz)
        cv2.imshow("Output2", np.hstack(class_masked_img))
        cv2.imshow("Mask", mask_viz)

        key = cv2.waitKey(30)
        if key == ord('q'):
            print('\33[1m\33[33mTerminating...\33[0m')
            break


# ==========================================
# =================== MAIN =================
# ==========================================

# Example usage:
# python3 test_model_in_RT.py --model=models/resnet18_seg.bin --seg_cfg_path=data/grapes_leaves_dataset/

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', required=True, type=str,
                        help='The name of the segmentation model')
    parser.add_argument('--seg_cfg_path', required=True, type=str,
                        help='The path of the segmentation config file.')
    parser.add_argument('--camera', default="rs", type=str,
                        help='The camera type. For realsense "rs", for dataset folder "bag:<dataset_folder>"')

    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':

    args = parse_args()
    test_model_in_RT(args['model'], args['seg_cfg_path'], args['camera'])
