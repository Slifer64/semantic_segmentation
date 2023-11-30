import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

from my_pkg.camera import RsCamera, DatasetCamera

from utils import save_image, save_mask, SegConfig, ColorSliders, get_contour_mask


def record_object_dataset(save_path: str, label: str, id: int, i_start=1, camera_type='rs'):

    plt.ion()

    # if os.path.exists(save_path): shutil.rmtree(save_path)
    if not os.path.exists(save_path): os.makedirs(save_path)

    SegConfig({label: id}).save(save_path)

    if camera_type == 'rs':
        camera = RsCamera()
    else:
        dataset = camera_type.split(':')[-1]
        camera = DatasetCamera(dataset)

    # sliders = ColorSliders(format='RGB')
    sliders = ColorSliders(format='HSV')

    count = i_start

    while True:
        rgb_img = camera.get_rgb()
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        img = np.copy(bgr_img)

        mask, bbox, contour = sliders.trackbar_callback(img)

        # use contour mask to fill in the gaps inside the contour
        # mask = get_contour_mask(mask_size=img.shape[0:2], contour=contour)

        bb_mask = np.zeros(mask.shape, np.uint8)

        if bbox is not None:
            x1, x2 = bbox['x'], bbox['x'] + bbox['width']
            y1, y2 = bbox['y'], bbox['y'] + bbox['height']

            bb_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

            # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            img = cv2.drawContours(img, [contour], 0, (255, 0, 255), 2)
            cv2.putText(img, "Detection", (bbox['x'], bbox['y']), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

        idx = bb_mask > 0
        masked_img = np.ones_like(bgr_img) * 255
        masked_img[idx] = bgr_img[idx]

        mask_viz = np.hstack([mask, bb_mask])
        img_viz = np.hstack([img, masked_img])

        cv2.imshow("Output1", img_viz)
        # cv2.imshow("Output2", imgHSV)
        cv2.imshow("Mask", mask_viz)

        # Wait until user press some key
        key = cv2.waitKey(30)
        if key == ord('q'):
            print('\33[1m\33[33mTerminating...\33[0m')
            break
        elif key == ord('r'):
            print('\33[1m\33[32mRecorded sample %d...\33[0m' % count)

            file_path = f"{save_path}/{str(count).zfill(4)}/"
            if os.path.exists(file_path): shutil.rmtree(file_path)
            os.makedirs(file_path)
            save_image(image=rgb_img, path=file_path)
            save_mask(mask=np.where(bb_mask == 255, id, 0).astype(int), path=file_path)
            count += 1



# ==========================================
# =================== MAIN =================
# ==========================================

# Example usage:
# python3 record_object_dataset.py --save_to=data/grapebunch_dataset/ --label_id=grapes:1 --i_start=1


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--save_to', required=True, type=str,
                        help='The folder where to save the dataset.')
    parser.add_argument('--label_id', required=True, type=str,
                        help='The object label and the integer id as a string: <label>:<id>')
    parser.add_argument('--i_start', default=1, type=int,
                        help='The number of the first sample. Set > 1 to add more data to an existing dataset folder.')
    parser.add_argument('--camera', default="rs", type=str,
                        help='The camera type. For realsense "rs", for dataset folder "bag:<dataset_folder>"')

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':

    args = parse_args()
    label, id = args['label_id'].split(':')
    record_object_dataset(save_path=args['save_to'], label=label, id=id, i_start=args['i_start'], camera_type=args['camera'])
