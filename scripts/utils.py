import numpy as np
import cv2
import os
import json
from typing import Dict, Tuple, Iterable, List


# ============ Color sliders ===========


class ColorSliders:

    grapes_color = {
    'min': [19, 43, 85],
    'max': [94, 186, 177]
    }

    leaves_color = {
        'min': [34, 89, 33],
        'max': [94, 209, 126]
    }

    default_color_range = grapes_color

    def __init__(self, format='RGB'):
        self.trackbars_win = "TrackedBars"
        cv2.namedWindow(self.trackbars_win)
        cv2.resizeWindow(self.trackbars_win, 700, 300)

        self.format = format
        if self.format == 'RGB':
            self.labels = ("Red", "Green", "Blue")
            self.limits = (255, 255, 255)
            init_min = [0, 182, 142]
            init_max = [209, 229, 196]
        elif self.format == 'HSV':
            self.labels = ("Hue", "Sat", "Val")
            self.limits = (179, 255, 255)
            init_min = ColorSliders.default_color_range['min']
            init_max = ColorSliders.default_color_range['max']
        else:
            raise RuntimeError('Unsupported format "%s"' % self.format)

        for label, limit, min0, max0 in zip(self.labels, self.limits, init_min, init_max):
            cv2.createTrackbar(label + " Min", self.trackbars_win, min0, limit, self.trackbar_callback)
            cv2.createTrackbar(label + " Max", self.trackbars_win, max0, limit, self.trackbar_callback)

    def trackbar_callback(self, rgb_img):

        img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV) if self.format == 'HSV' else rgb_img

        lower = []
        upper = []
        for label in self.labels:
            lower.append(cv2.getTrackbarPos(label + " Min", self.trackbars_win))
            upper.append(cv2.getTrackbarPos(label + " Max", self.trackbars_win))

        mask = cv2.inRange(img, np.array(lower), np.array(upper))

        # dilate mask
        mask = cv2.dilate(mask, np.ones((7, 7), "uint8"), iterations=1)
        mask = cv2.dilate(mask, np.ones((5, 5), "uint8"), iterations=1)
        # mask = cv2.dilate(mask, np.ones((4, 4), "uint8"), iterations=1)
        mask = cv2.erode(mask, np.ones((4, 4), "uint8"), iterations=1)
        mask = cv2.erode(mask, np.ones((3, 3), "uint8"), iterations=1)

        # find largest contour
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        i_max = -1
        max_area = -1

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max(36, max_area):
                i_max = i
                max_area = area

        if i_max > -1:
            contour = contours[i_max]
            x, y, w, h = cv2.boundingRect(contour)  # get bounding box
            bbox = {'x': x, 'y': y, 'width': w, 'height': h}
        else:
            bbox = None
            contour = None

        return mask, bbox, contour


# ============ SegConfig ===========


class SegConfig:
    def __init__(self, label_id: Dict[str, int]):
        self.lid = {k:int(v) for k, v in label_id.items()}
        self._inv_lid = {v:k for k, v in self.lid.items()}
        self.n_classes = len(self.lid)

    def items(self) -> Iterable[Tuple[str, int]]:
        return self.lid.items()
    
    def get_ids(self) -> List[int]:
        return list(self.lid.values())

    def get_labels(self) -> List[str]:
        return list(self.lid.keys())

    def __getitem__(self, label: str) -> int:
        return self.lid[label]

    def get_id(self, label: str) -> int:
        return self.lid[label]

    def get_label(self, id: int) -> str:
        return self._inv_lid[id]

    @classmethod
    def load(cls, path: str) -> 'SegConfig':
        with open(os.path.join(path, 'config.txt')) as fp:
            label_id = json.loads(fp.read())
        return SegConfig(label_id=label_id)

    def save(self, path: str):
        with open(os.path.join(path, 'config.txt'), 'w') as fp:
            json.dump(self.lid, fp)



# ============ Misc ===========


def load_image(path: str, filename='rgb.png') -> np.array:
    return cv2.cvtColor(cv2.imread(os.path.join(path, filename)), cv2.COLOR_BGR2RGB)


def save_image(image: np.array, path: str, filename='rgb.png'):
    cv2.imwrite(os.path.join(path, filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def load_mask(path: str, filename='mask.png') -> np.array:
    return cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)


def save_mask(mask: np.array, path: str, filename='mask.png'):
    """
    Arguments
    mask: np.array, type int
    """
    cv2.imwrite(os.path.join(path, filename), mask)


def apply_mask_to_image(image: np.array, mask: np.array) -> np.array:
    ind = mask > 0
    masked_image = np.zeros_like(image)
    masked_image[ind] = image[ind]
    return masked_image


def get_contour_mask(mask_size: Tuple[int, int], contour) -> np.array:
    img = cv2.drawContours(np.zeros(mask_size), [contour], 0, color=255, thickness=-1)
    return cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1]


def filter_mask(mask: np.array, area_thres: float, filt_shape='contour', fill_shape_area=False) -> Tuple[np.array, List[List[int]]]:
    """
    Filters the input mask, removing contours with area less than a given threshold.

    Arguments:
        mask -- np.array(H, W, uint8), the input mask
        area_thres -- float, the area threshold
        filt_shape -- str, 'contour': keeps only what's inside the contour
                           'box': keeps only what's inside the bounding box
        fill_shape_area -- bool, applies for filt_shape='contour', used to fill or not the area inside the contour

    Returns:
        np.array, filtered mask
        List[List[int]], the filtered contours

    """

    if filt_shape not in ('contour', 'box'):
        raise AttributeError(f'Unsupported filt_shape "{filt_shape}"...')

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    filt_contours = [c for c in contours if cv2.contourArea(c) > area_thres]
    filt_mask = np.zeros(mask.shape, np.uint8)

    for contour in filt_contours:
        if filt_shape == 'contour':
            c_mask = get_contour_mask(mask.shape, contour)
            if fill_shape_area:
                filt_mask[c_mask > 0] = 255  # takes all the area inside the contour
            else:
                filt_mask[(c_mask > 0) & (mask > 0)] = 255  # takes the pixesl inside the contour that belong to the mask
        elif filt_shape =='box':    
            x, y, w, h = cv2.boundingRect(contour)  # get bounding box
            x1, x2 = x, x+w
            y1, y2 = y, y+h
            filt_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        
    return filt_mask, filt_contours


def draw_labeled_contours(img_viz: np.array, contours: List[List[int]], color, label: str) -> None:

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # get bounding box
        img_viz = cv2.drawContours(img_viz, [contour], 0, color, 2)
        cv2.putText(img_viz, text=label, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=color, thickness=2)


class SegGui:
    def __init__(self) -> None:
        
        self._win_name = "Control options"
        cv2.namedWindow(self._win_name)
        cv2.resizeWindow(self._win_name, 1000, 300)
        self._params = {}

    def addParam(self, name, min_value, max_value):

        is_float = isinstance(min_value, float) or isinstance(max_value, float)

        if is_float:
            int_max = int(1000*max_value + 0.5)
            int_min = int(1000*min_value + 0.5)
        else:
            int_max = max_value
            int_min = min_value

        self._params[name] = is_float # (min_value, max_value)

        cv2.createTrackbar(name, self._win_name, int_min, int_max, self._callback)
        cv2.setTrackbarMin(name, self._win_name, int_min)

    def _callback(self, img):
        pass

    def get(self, name):
        is_float = self._params[name]
        value = cv2.getTrackbarPos(name, self._win_name)
        if is_float:
            value = value / 1000.0
        return value
    
    def set(self, name, value):
        is_float = self._params[name]
        if is_float:
            value = int(value*1000 + 0.5)
        cv2.setTrackbarPos(name, self._win_name, value)
