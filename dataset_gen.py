import os
from itertools import repeat

import numpy as np
from cv2 import cv2
from trvo_utils.cv2gui_utils import imshowWait
from trvo_utils.imutils import imHW, fit_image_boxes_to_shape

from utils.augmentations import Augmentations
from utils.box_drawer import box_drawer
from utils.gen_screens_lib import screen_view_sampler
from utils.dataset_directory import DatasetDirectory
from utils.pascal_voc_writer import PascalVocWriter


class padder:
    @staticmethod
    def get_absolute_padding(imgHW, src_padding):
        assert src_padding >= 0
        if isinstance(src_padding, float):
            h, w = imgHW
            return int(h * src_padding), int(w * src_padding)
        return src_padding, src_padding

    @staticmethod
    def pad_boxes(xyxy_boxes, x_padding, y_padding):
        padded_boxes = []
        for x1, y1, x2, y2 in xyxy_boxes:
            padded_box = [
                x1 + x_padding,
                y1 + y_padding,
                x2 + x_padding,
                y2 + y_padding
            ]
            padded_boxes.append(np.float32(padded_box))
        return padded_boxes

    @staticmethod
    def pad_image(img, x_padding, y_padding, pad_value):
        assert len(img.shape) in [2, 3]
        depth = img.shape[2] if len(img.shape) == 3 else 0
        h, w = imHW(img)
        padded_img_shape = [h + y_padding * 2, w + x_padding * 2, 3] if depth > 0 else [h + y_padding * 2,
                                                                                        w + x_padding * 2]
        padded_img = np.full(padded_img_shape, pad_value, np.uint8)
        padded_img[y_padding:y_padding + h, x_padding: x_padding + w] = img
        return padded_img

    @classmethod
    def pad_annotated_image(cls, img, boxes, padding, pad_value=0):
        x_padding, y_padding = cls.get_absolute_padding(imHW(img), padding)
        padded_img = cls.pad_image(img, x_padding, y_padding, pad_value)
        padded_boxes = cls.pad_boxes(boxes, x_padding, y_padding)
        return padded_img, padded_boxes


def prepared_images(augmenter, src_dataset_dir, padding=.5, pad_value=0):
    for img, boxes, labels, img_path, ann_path, ann_exist in DatasetDirectory(src_dataset_dir).load_and_parse():
        # Pad image/boxes to avoid cropping during transformations
        padded_img, padded_boxes = padder.pad_annotated_image(img, boxes, padding=padding, pad_value=pad_value)
        augm_img, augm_boxes, augm_labels = augmenter(padded_img, padded_boxes, labels)
        yield (augm_img, augm_boxes, augm_labels), (img, boxes, labels), (img_path, ann_path, ann_exist)


class visualizer:
    @staticmethod
    def fit_to_screen(img, boxes):
        screen_shape = (1850, 950)
        img, boxes, _ = fit_image_boxes_to_shape(img, boxes, screen_shape)
        return img, boxes

    @classmethod
    def show(cls, augm_img, augm_boxes, img=None, boxes=None):
        if img is not None:
            img = box_drawer.xyxy(*cls.fit_to_screen(img, boxes))
        augm_img = box_drawer.xyxy(*cls.fit_to_screen(augm_img, augm_boxes))
        key = imshowWait(
            img,
            augm_img
        )
        if key == 27: return "esc"


class AugmentedGenerator:
    def __init__(self, augmentations: Augmentations, padding=.5, pad_value=0):
        self.augmentations = augmentations
        self.padding = padding
        self.pad_value = pad_value

    def generate(self, src_img, src_boxes, src_labels, n):
        # Pad image/boxes to avoid cropping during transformations
        padded_img, padded_boxes = padder.pad_annotated_image(src_img, src_boxes,
                                                              padding=self.padding, pad_value=self.pad_value)
        for num in range(n):
            augm_img, augm_boxes, augm_labels = self.augmentations(padded_img, padded_boxes, src_labels)
            yield num, (augm_img, augm_boxes, augm_labels)

    def __call__(self, src_img, src_boxes, src_labels, n):
        return self.generate(src_img, src_boxes, src_labels, n)


def main():
    mobile_roi_w, mobile_roi_h = 400, 180
    src_dst_dataset_dirs = [
        # ("training_datasets/v1/Musson_counters", "training_datasets/v1_generated/Musson_counters"),
        # ("training_datasets/v1/Musson_counters_3_1280x960", "training_datasets/v1_generated/Musson_counters_3"),
        ("training_datasets/v2/1_trvo", "training_datasets/v2_generated/1_trvo"),
    ]

    # TODO("If only screen is annotated (there is no annotated digits) - make this sample negative")
    # TODO("Collect negative samples. Hard negative mining???")
    augmenter = AugmentedGenerator(Augmentations(p=1.0), padding=.2)

    num_of_augmentations = 15
    n_screen_views = 6

    i = 0
    for src_dataset_dir, dst_dataset_dir in src_dst_dataset_dirs:
        src_dataset_dir = file_utils.ensure_path_sep(src_dataset_dir)
        dst_dataset_dir = file_utils.ensure_path_sep(dst_dataset_dir)

        os.makedirs(dst_dataset_dir, exist_ok=True)

        for img, boxes, labels, img_path, ann_path, ann_exist in \
                DatasetDirectory(src_dataset_dir, recursive=True).load_and_parse():
            if not ann_exist:
                continue
            screen_boxes = [b for (b, l) in zip(boxes, labels) if l == "screen"]
            if len(screen_boxes) != 1:
                continue
            if len(boxes) == 1:  # only screen is annotated
                continue

            img_parent_dir, img_base_name, img_ext = file_utils.base_name_and_ext(img_path)
            dst_parent_dir = file_utils.dst_parent_dir(dst_dataset_dir, src_dataset_dir, img_parent_dir,
                                                       ensure_existence=True)

            for version_of_img, (augm_img, augm_boxes, augm_labels) in augmenter(img, boxes, labels,
                                                                                 num_of_augmentations):
                screen_box = next(b for (b, l) in zip(augm_boxes, augm_labels) if l == "screen")
                view_samples = screen_view_sampler.views(
                    n_screen_views=n_screen_views,
                    img=augm_img,
                    screen_box=screen_box,
                    boxes=augm_boxes,
                    view_wh_ratio=mobile_roi_w / mobile_roi_h,
                    area_limit=(2.3, 12),
                    min_distance=10,
                    fill_value=0)
                for num_of_view, (view_img, view_boxes, view_box_in_img) in view_samples:
                    dst_img_base_name = f"{img_base_name}_{version_of_img:04d}_{num_of_view:04d}"
                    dst_img_file_name = dst_img_base_name + img_ext
                    dst_ann_file_name = dst_img_base_name + '.xml'
                    dst_img_path = os.path.join(dst_parent_dir, dst_img_file_name)
                    dst_ann_path = os.path.join(dst_parent_dir, dst_ann_file_name)

                    cv2.imwrite(dst_img_path, view_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    PascalVocWriter.write(dst_ann_path, dst_img_path, view_img.shape, view_boxes, augm_labels)
                    i += 1
                    if i % 50 == 0:
                        print(i)


class file_utils:
    @staticmethod
    def base_name_and_ext(path):
        parent, file_name = os.path.split(path)
        base_name, ext = os.path.splitext(file_name)
        return parent, base_name, ext

    @staticmethod
    def ensure_path_sep(path: str):
        # at the end of path
        return os.path.join(path, "")

    @staticmethod
    def __assert_path_sep(path: str):
        # at the end of path
        assert (path[-1] == os.path.sep)

    @classmethod
    def dst_parent_dir(cls, dst_dir: str, src_dir: str, src_parent_dir: str, ensure_existence=True):
        cls.__assert_path_sep(dst_dir)
        cls.__assert_path_sep(src_dir)
        assert (src_parent_dir.startswith(src_dir))

        dst_parent_dir = src_parent_dir.replace(src_dir, dst_dir, 1)
        if (ensure_existence):
            os.makedirs(dst_parent_dir, exist_ok=True)
        return dst_parent_dir


if __name__ == '__main__':
    main()
