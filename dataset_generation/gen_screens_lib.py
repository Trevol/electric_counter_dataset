from math import sqrt

from cv2 import cv2
from trvo_utils import TODO
from trvo_utils.cv2gui_utils import imshowWait
import numpy as np

from dataset_generation.augmentations import Augmentations
from dataset_generation.box_drawer import box_drawer
from dataset_generation.dataset_gen import AugmentedGenerator
from utils.box_utils import box_utils
from utils.dataset_directory import DatasetDirectory
import random


class screen_view_sampler:
    class rnd:
        @staticmethod
        def value(a, b):
            return random.uniform(a, b)

        @classmethod
        def from_limit(cls, limit):
            a, b = limit
            return cls.value(a, b)

    @staticmethod
    def box_wh(box_area, wh_ratio):
        h = sqrt(box_area / wh_ratio)
        w = box_area / h
        return w, h

    @classmethod
    def views(cls, n_screen_views, img, screen_box, digit_boxes, view_wh_ratio, area_limit, min_distance,
              fill_value):
        for _ in range(n_screen_views):
            v = cls.view(img, screen_box, digit_boxes, view_wh_ratio, area_limit, min_distance, fill_value)
            yield v

    @classmethod
    def _rnd_view_box(cls, screen_box, view_wh_ratio, area_limit, min_distance):
        view_box_w, view_box_h = cls.box_wh(
            box_area=box_utils.area(screen_box) * cls.rnd.from_limit(area_limit),
            wh_ratio=view_wh_ratio)
        assert view_box_w > (box_utils.width(screen_box) + min_distance * 2)
        assert view_box_h > (box_utils.height(screen_box) + min_distance * 2)
        # 1) random box origin - x1, y1. Calculate using min_distance and screen_box_wh
        x_min = box_utils.x2(screen_box) + min_distance - view_box_w
        x_max = box_utils.x1(screen_box) - min_distance
        y_min = box_utils.y2(screen_box) + min_distance - view_box_h
        y_max = box_utils.y1(screen_box) - min_distance
        view_box_x = cls.rnd.value(x_min, x_max)
        view_box_y = cls.rnd.value(y_min, y_max)
        view_box = [view_box_x, view_box_y, view_box_x + view_box_w, view_box_y + view_box_h]
        return view_box

    @classmethod
    def view(cls, img, screen_box, digit_boxes, view_wh_ratio, area_limit, min_distance, fill_value):
        """
        area_limit = area(view_box)/area(screen_box)
        """

        # random box size
        view_box = cls._rnd_view_box(screen_box, view_wh_ratio, area_limit, min_distance)
        # 2) remap screen_box/digit_boxes from image to view_box coordinates
        # 3) take image by view_box. Fill by fill_value if necessary
        raise NotImplementedError()
        return view_img, boxes_in_view, view_box


def main():
    mobile_roi_w, mobile_roi_h = 400, 180

    dataset_dir = "../training_datasets/v1/Musson_counters"
    augmenter = AugmentedGenerator(Augmentations(p=1.0), padding=.2)
    img, boxes, labels, img_path, ann_path, ann_exist = next(DatasetDirectory(dataset_dir).load_and_parse())
    screen_box = next(b for (b, l) in zip(boxes, labels) if l == "screen")
    n_screen_views = 1000

    view_samples = screen_view_sampler.views(
        n_screen_views=n_screen_views,
        img=img,
        screen_box=screen_box,
        digit_boxes=[],
        view_wh_ratio=mobile_roi_w / mobile_roi_h,
        area_limit=(2, 8),
        min_distance=10,
        fill_value=0)
    for view_img, view_boxes, view_box_in_img in view_samples:
        disp_img = img.copy()
        box_drawer.xyxy(disp_img, [view_box_in_img], color=(0, 255, 0)),
        box_drawer.xyxy(disp_img, boxes)

        key = imshowWait(disp_img)
        if key == 27: break


if __name__ == '__main__':
    main()
