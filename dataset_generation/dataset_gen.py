import os
from itertools import repeat

import numpy as np
from trvo_utils.cv2gui_utils import imshowWait
from trvo_utils.imutils import imHW, fit_image_boxes_to_shape

from dataset_generation.augmentations import Augmentations
from dataset_generation.box_drawer import box_drawer
from utils.dataset_directory import DatasetDirectory


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


def fit_to_screen(img, boxes):
    screen_shape = (1850, 950)
    img, boxes, _ = fit_image_boxes_to_shape(img, boxes, screen_shape)
    return img, boxes


def prepared_images(augmenter, src_dataset_dir, padding=.5, pad_value=0):
    for img, boxes, labels, img_path, ann_path, ann_exist in DatasetDirectory(src_dataset_dir).load_and_parse():
        # Pad image/boxes to avoid cropping during transformations
        padded_img, padded_boxes = padder.pad_annotated_image(img, boxes, padding=padding, pad_value=pad_value)
        augm_img, augm_boxes, augm_labels = augmenter(padded_img, padded_boxes, labels)
        yield (augm_img, augm_boxes, augm_labels), (img, boxes, labels), (img_path, ann_path, ann_exist)


def show(augm_img, augm_boxes, img=None, boxes=None):
    if img is not None:
        img = box_drawer.xyxy(*fit_to_screen(img, boxes))
    augm_img = box_drawer.xyxy(*fit_to_screen(augm_img, augm_boxes))
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
        for _ in repeat(None, n):
            augm_img, augm_boxes, augm_labels = self.augmentations(padded_img, padded_boxes, src_labels)
            yield augm_img, augm_boxes, augm_labels

    def __call__(self, src_img, src_boxes, src_labels, n):
        return self.generate(src_img, src_boxes, src_labels, n)


def main():
    mobile_roi_wh = 400, 180
    src_dst_dataset_dirs = [
        ("../training_datasets/v1/Musson_counters", "../training_datasets/v1_generated/Musson_counters"),
        ("../training_datasets/v1/Musson_counters_3_1280x960", "../training_datasets/v1_generated/Musson_counters_3")
    ]
    # TODO("skip two (or more) annotated screens")
    # TODO("If only screen is annotated (there is no annotated digits) - make this sample negative")
    # TODO("Collect negative samples. Hard negative mining???")
    augmenter = AugmentedGenerator(Augmentations(p=1.0), padding=.2)
    num_of_augmentations = 100

    for src_dataset_dir, dst_dataset_dir in src_dst_dataset_dirs:
        os.makedirs(dst_dataset_dir, exist_ok=True)
        for img, boxes, labels, img_path, ann_path, ann_exist in DatasetDirectory(src_dataset_dir).load_and_parse():
            for (augm_img, augm_boxes, augm_labels) in augmenter(img, boxes, labels, num_of_augmentations):
                if show(augm_img, augm_boxes) == 'esc': return


if __name__ == '__main__':
    main()
