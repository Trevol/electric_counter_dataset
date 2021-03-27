from math import sqrt

import trvo_utils
from cv2 import cv2
from trvo_utils import TODO, toInt_array
from trvo_utils.core import Rect
from trvo_utils.cv2gui_utils import imshowWait
import numpy as np
from trvo_utils.imutils import imWH, img_by_xyxy_box_unsafe

from utils.box_drawer import box_drawer
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
    def views(cls, n_screen_views, img, screen_box, boxes, view_wh_ratio, area_limit, min_distance,
              fill_value):
        for num_of_view in range(n_screen_views):
            v = cls.view(img, screen_box, boxes, view_wh_ratio, area_limit, min_distance, fill_value)
            yield num_of_view, v

    @classmethod
    def _rnd_view_box(cls, screen_box, view_wh_ratio, area_limit, min_distance):
        rnd_area_limit = cls.rnd.from_limit(area_limit)
        view_box_w, view_box_h = cls.box_wh(
            box_area=box_utils.area(screen_box) * rnd_area_limit,
            wh_ratio=view_wh_ratio)
        try:
            assert view_box_w > (box_utils.width(screen_box) + min_distance * 2)
            assert view_box_h > (box_utils.height(screen_box) + min_distance * 2)
        except Exception:
            print("AssertionError!!", rnd_area_limit)
            raise

        # 1) random box origin - x1, y1. Calculate using min_distance and screen_box_wh
        x_min = box_utils.x2(screen_box) + min_distance - view_box_w
        x_max = box_utils.x1(screen_box) - min_distance
        y_min = box_utils.y2(screen_box) + min_distance - view_box_h
        y_max = box_utils.y1(screen_box) - min_distance
        view_box_x = cls.rnd.value(x_min, x_max)
        view_box_y = cls.rnd.value(y_min, y_max)
        view_box = [view_box_x, view_box_y, view_box_x + view_box_w, view_box_y + view_box_h]
        return view_box

    @staticmethod
    def _remap(src_boxes, dst_box):
        dst_x1, dst_y1 = dst_box[0], dst_box[1]
        return [
            [x1 - dst_x1, y1 - dst_y1, x2 - dst_x1, y2 - dst_y1]
            for x1, y1, x2, y2 in src_boxes
        ]

    @classmethod
    def view(cls, img, screen_box, boxes, view_wh_ratio, area_limit, min_distance, fill_value):
        """
        area_limit = area(view_box)/area(screen_box)
        """

        # random box size
        view_box = cls._rnd_view_box(screen_box, view_wh_ratio, area_limit, min_distance)
        # 2) remap screen_box/digit_boxes from image to view_box coordinates
        boxes_in_view = cls._remap(boxes, view_box)
        # 3) take image by view_box. Fill by fill_value if necessary
        view_img = cls._view_img(img, view_box, fill_value)
        # raise NotImplementedError()
        return view_img, boxes_in_view, view_box

    @classmethod
    def _view_img(cls, src_img, view_box, fill_value):
        view_box = toInt_array(view_box)
        im_width, im_height = imWH(src_img)

        view_height, view_width = box_utils.size_hw(view_box)
        view_img = np.full([view_height, view_width, 3],
                           fill_value, np.uint8)
        # in image coords
        src_img_rect = Rect([0, 0, im_width, im_height])
        view_rect = Rect.fromXyxy(view_box)
        src_img_part = img_by_xyxy_box_unsafe(
            src_img,
            src_img_rect.intersection(view_rect).xyxy
        )

        # in view box coords
        view_rect = Rect([0, 0, view_width, view_height])
        src_img_rect = Rect(
            [
                -box_utils.x1(view_box),
                -box_utils.y1(view_box),
                im_width,
                im_height
            ]
        )
        dst_box = view_rect.intersection(src_img_rect).xyxy
        x1, y1, x2, y2 = dst_box
        view_img[y1:y2, x1:x2] = src_img_part
        return view_img


if __name__ == '__main__':
    def main():
        mobile_roi_w, mobile_roi_h = 400, 180

        dataset_dir = "../training_datasets/v1/Musson_counters"
        img, boxes, labels, img_path, ann_path, ann_exist = next(DatasetDirectory(dataset_dir).load_and_parse())
        screen_box = next(b for (b, l) in zip(boxes, labels) if l == "screen")
        n_screen_views = 1000

        view_samples = screen_view_sampler.views(
            n_screen_views=n_screen_views,
            img=img,
            screen_box=screen_box,
            boxes=boxes,
            view_wh_ratio=mobile_roi_w / mobile_roi_h,
            area_limit=(2, 8),
            min_distance=10,
            fill_value=0)
        for num_of_view, (view_img, view_boxes, view_box_in_img) in view_samples:

            disp_img = img.copy()
            box_drawer.xyxy(disp_img, [view_box_in_img], color=(0, 255, 0)),
            box_drawer.xyxy(disp_img, boxes)

            key = imshowWait(disp_img, view_img)
            if key == 27: break


    main()
