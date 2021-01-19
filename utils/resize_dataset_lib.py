from trvo_utils import TODO
import cv2
import numpy as np
import os

from trvo_utils.imutils import imHW

from utils.dataset_directory import DatasetDirectory
from utils.pascal_voc_writer import PascalVocWriter


class resizer:
    scale_tolerance = 0.0001

    @classmethod
    def calc_scale(cls, src_size, desired_size, force_desired_size):
        src_dim1, src_dim2 = cls.min_max(*src_size)
        dst_dim1, dst_dim2 = cls.min_max(*desired_size)
        # что выбрать - больше или меньше?
        scale1 = dst_dim1 / src_dim1
        scale2 = dst_dim2 / src_dim2
        if force_desired_size:
            assert scale1 == scale2
            return scale1
        scale = max(scale1, scale2)
        return scale

    @staticmethod
    def min_max(a1, a2):
        return (a1, a2) if a1 < a2 else (a2, a1)

    @classmethod
    def resize_boxes(cls, xyxy_boxes, scale):
        resized_boxes = [cls.resize_box(xyxy_box, scale) for xyxy_box in xyxy_boxes]
        return resized_boxes

    @staticmethod
    def _is_np_array(arg):
        return isinstance(arg, np.ndarray)

    @staticmethod
    def list_mult(src_array, k):
        return [i * k for i in src_array]

    @classmethod
    def resize_box(cls, xyxy_box, scale):
        if cls._is_np_array(xyxy_box):
            return xyxy_box * scale
        return cls.list_mult(xyxy_box, scale)

    @classmethod
    def resize_annotated_image(cls, img, boxes, labels, desired_size, force_desired_size):
        resized_img, scale = cls.resize_image(img, desired_size, force_desired_size)
        resized_boxes = cls.resize_boxes(boxes, scale)
        return resized_img, resized_boxes, labels, scale

    @classmethod
    def resize_image(cls, img, desired_size, force_desired_size):
        scale = cls.calc_scale(imHW(img), desired_size, force_desired_size)
        if scale == 1:
            return img.copy(), 1
        interpolation = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
        resized_img = cv2.resize(img, None, None, scale, scale, interpolation)
        return resized_img, scale


def imwrite(img_path, img):
    cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])


def save_annotation(ann_path, img_path, img_shape, boxes, labels):
    PascalVocWriter.write(ann_path, img_path, img_shape, boxes, labels)


def dst_path(src_path, dst_dir):
    file_name = os.path.split(src_path)[1]
    return os.path.join(dst_dir, file_name)


def resize_images_in_dir(src_dir, dst_dir, desired_size, force_desired_size):
    assert src_dir != dst_dir
    for img, boxes, labels, src_img_path, src_ann_path, ann_exist in DatasetDirectory(src_dir).load_and_parse():
        dst_img_path = dst_path(src_img_path, dst_dir)
        if not ann_exist:
            resized_img, _ = resizer.resize_image(img, desired_size, force_desired_size)
            imwrite(
                dst_img_path,
                resized_img)
            continue
        resized_img, resized_boxes, resized_labels, _ = resizer.resize_annotated_image(img, boxes, labels, desired_size,
                                                                                       force_desired_size)

        imwrite(
            dst_img_path,
            resized_img)
        save_annotation(
            dst_path(src_ann_path, dst_dir),
            dst_img_path,
            resized_img.shape,
            resized_boxes,
            resized_labels
        )
