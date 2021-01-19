import os
import cv2
from albumentations import (
    IAAPerspective, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip,
    Compose, BboxParams,
    ISONoise
)
from trvo_utils.annotation import PascalVocXmlParser
from trvo_utils.cv2gui_utils import imshowWait

from dataset_generation import augmentations
from dataset_generation.augmentations import Augmentations
from dataset_generation.box_drawer import box_drawer


def augmentation():
    def _all():
        RandomRotate90(),
        Flip(),
        Transpose(),
        IAAAdditiveGaussianNoise(),
        GaussNoise(),

        MotionBlur(p=0.2),
        MedianBlur(blur_limit=3, p=0.1),
        Blur(blur_limit=3, p=0.1),

        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=7, p=0.2),
        OpticalDistortion(p=0.3),

        CLAHE(clip_limit=2),
        IAASharpen(),
        IAAEmboss(),
        RandomBrightnessContrast(),

        HueSaturationValue(p=0.3)

    return ISONoise(always_apply=True)


def use_it():
    return [
        HueSaturationValue(always_apply=True),
        IAAAdditiveGaussianNoise() or GaussNoise(),
        MotionBlur(blur_limit=3, always_apply=True) or MedianBlur(blur_limit=3, always_apply=True),
        ShiftScaleRotate(shift_limit=0.0925, scale_limit=0.4, rotate_limit=7, border_mode=cv2.BORDER_CONSTANT,
                         value=0, always_apply=True),
        # OpticalDistortion(always_apply=True),
        CLAHE(clip_limit=2, always_apply=True),
        # IAASharpen(always_apply=True),
        IAAEmboss(always_apply=True),
        RandomBrightnessContrast(always_apply=True)
    ]


def main():
    imgFile = "/hdd/Datasets/counters/training_datasets/v0/small/000011.jpg"
    ann_file = os.path.splitext(imgFile)[0] + ".xml"
    parser = PascalVocXmlParser(ann_file)
    image = cv2.imread(imgFile)
    # image = cv2.resize(image, None, None, .5, .5)
    boxes, labels = parser.annotation()

    original_boxes_img = box_drawer.xyxy(image.copy(), boxes)

    # augm = use_it()

    # augm = [
    #     ShiftScaleRotate(
    #         shift_limit=0.01,  # 0.0925
    #         scale_limit=0.4 - 0.4,
    #         rotate_limit=9-9,
    #         border_mode=cv2.BORDER_CONSTANT,
    #         value=0,
    #         always_apply=True)
    # ]

    augm = Augmentations(1.0)
    while True:
        augm_image, augm_boxes, augm_labels = augm(image, boxes, labels)
        k = imshowWait(
            image=original_boxes_img,
            augmented=box_drawer.xyxy(augm_image, augm_boxes)
        )
        if k == 27:
            break


def main_():
    imgFile = "/hdd/Datasets/counters/training_datasets/v0/small/000011.jpg"
    ann_file = os.path.splitext(imgFile)[0] + ".xml"
    parser = PascalVocXmlParser(ann_file)
    image = cv2.imread(imgFile)
    # image = cv2.resize(image, None, None, .5, .5)
    boxes, labels = parser.annotation()

    original_boxes_img = box_drawer.xyxy(image.copy(), boxes)

    # augm = use_it()

    # augm = [
    #     ShiftScaleRotate(
    #         shift_limit=0.01,  # 0.0925
    #         scale_limit=0.4 - 0.4,
    #         rotate_limit=9-9,
    #         border_mode=cv2.BORDER_CONSTANT,
    #         value=0,
    #         always_apply=True)
    # ]

    augm = Augmentations.make(p=.7)
    augm = Compose(augm, bbox_params=BboxParams(format='pascal_voc', label_fields=['labels']))
    while True:
        augmented = augm(image=image, bboxes=boxes, labels=labels)
        k = imshowWait(
            image=original_boxes_img,
            augmented=box_drawer.xyxy(augmented['image'], augmented['bboxes'])
        )
        if k == 27:
            break


main()
