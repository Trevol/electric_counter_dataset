from cv2 import cv2
from trvo_utils.cv2gui_utils import imshowWait
import numpy as np

from dataset_generation.augmentations import Augmentations
from dataset_generation.box_drawer import box_drawer
from dataset_generation.dataset_gen import AugmentedGenerator
from utils.box_utils import box_utils
from utils.dataset_directory import DatasetDirectory


def gen_screen_views(img, screen_box, view_wh_ratio, area_limit, distance_limit):
    screen_box_area = box_utils.area(screen_box)
    area_limit_min, area_limit_max = area_limit
    pass


def main():
    mobile_roi_w, mobile_roi_h = 400, 180

    dataset_dir = "../training_datasets/v1/Musson_counters"
    augmenter = AugmentedGenerator(Augmentations(p=1.0), padding=.2)
    img, boxes, labels, img_path, ann_path, ann_exist = next(DatasetDirectory(dataset_dir).load_and_parse())
    screen_box = next(b for (b, l) in zip(boxes, labels) if l == "screen")
    while True:
        gen_screen_views(
            img,
            screen_box=screen_box,
            view_wh_ratio=mobile_roi_w / mobile_roi_h,
        )
        if imshowWait(box_drawer.xyxy(img, boxes)) == 27: break


if __name__ == '__main__':
    main()
