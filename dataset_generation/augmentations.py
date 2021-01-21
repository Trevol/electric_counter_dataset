from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, ISONoise, RGBShift, ImageOnlyTransform,
    BboxParams, Perspective, Sharpen, Emboss
)
import cv2


class Augmentations:
    @classmethod
    def __make1(cls, p=0.5):
        return Compose([
            # OneOf([
            #     IAAAdditiveGaussianNoise(),
            #     GaussNoise(),
            #     ISONoise()
            # ], p=0.9),
            # MotionBlur(p=0.3),
            # ShiftScaleRotate(shift_limit=0.0925, scale_limit=0.4, rotate_limit=7, border_mode=cv2.BORDER_CONSTANT,
            #                  value=0, p=0.6),
            # IAAPerspective(scale=(.055, .060), keep_size=False, p=.2),
            # OpticalDistortion(p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p=0.3),
            HueSaturationValue(p=0.3),
            RGBShift(40, 40, 40, p=.5),
            Invert(p=.5)
        ], p=p)

    @classmethod
    def make(cls, bbox_params, p):
        return Compose([
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
                ISONoise()
            ], p=0.9),
            MotionBlur(p=0.6),
            ShiftScaleRotate(shift_limit=0.0, scale_limit=(0, 0), rotate_limit=9,
                             border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
            # Perspective(scale=(.055, .060), keep_size=True, p=1.6),
            OneOf([
                CLAHE(clip_limit=2),
                Sharpen(),
                Emboss(),
                RandomBrightnessContrast(),
            ], p=0.6),
            HueSaturationValue(p=0.6),
            RGBShift(40, 40, 40, p=.6),
            # Invert(p=.5)
        ], bbox_params=bbox_params, p=p)

    def __init__(self, p):
        bbox_params = BboxParams(format='pascal_voc', label_fields=['labels'])
        self._augmentation_pipeline = self.make(bbox_params, p)

    def apply(self, image, xyxy_boxes, labels):
        result = self._augmentation_pipeline(image=image, bboxes=xyxy_boxes, labels=labels)
        return result["image"], result["bboxes"], result["labels"]

    def __call__(self, image, xyxy_boxes, labels):
        return self.apply(image, xyxy_boxes, labels)


class Invert(ImageOnlyTransform):
    def apply(self, image, **params):
        return 255 - image
