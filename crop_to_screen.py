from glob import glob
import os
import cv2
import numpy as np
from trvo_utils import TODO, toInt_array
from trvo_utils.annotation import PascalVocXmlParser
from trvo_utils.cv2gui_utils import imshowWait
from trvo_utils.imutils import imWH

from utils.pascal_voc_writer import PascalVocWriter


class box_util:
    @staticmethod
    def width(box):
        # 0123
        # xyxy
        return box[2] - box[0]  # x2-x1

    @staticmethod
    def height(box):
        # 0123
        # xyxy
        return box[3] - box[1]  # y2-y1

    @staticmethod
    def size_wh(box):
        return box_util.width(box), box_util.height(box)

    @staticmethod
    def ensure_containment(outer_size_wh, inner_box):
        TODO()

    @staticmethod
    def draw(img, boxes):
        for box in boxes:
            x1, y1, x2, y2 = toInt_array(box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
        return img


def cropped_image_box(src_box, relative_pad_w, relative_pad_h, image_size_wh):
    box_w, box_h = box_util.size_wh(src_box)
    pad_w = box_w * relative_pad_w
    pad_h = box_h * relative_pad_h
    image_w, image_h = image_size_wh
    box_x1, box_y1, box_x2, box_y2 = src_box
    cropped_x1, cropped_y1, cropped_x2, cropped_y2 = (
        max(box_x1 - pad_w, 0),
        max(box_y1 - pad_h, 0),
        min(box_x2 + pad_w, image_w),
        min(box_y2 + pad_h, image_h)
    )
    return cropped_x1, cropped_y1, cropped_x2, cropped_y2


def load_annotated_image(ann_file):
    base_name = os.path.splitext(ann_file)[0]
    image_path = base_name + ".jpg"
    assert os.path.isfile(image_path)
    image_file_name = os.path.split(image_path)[1]
    return cv2.imread(image_path), image_file_name


def image_by_box(src_img, box):
    x1, y1, x2, y2 = toInt_array(box)
    return src_img[y1:y2, x1:x2]


def crop_boxes(src_boxes, cropped_box):
    cropped_box_x1, cropped_box_y1, _, _ = cropped_box
    dst_boxes = [np.float32([src_x1 - cropped_box_x1,
                             src_y1 - cropped_box_y1,
                             src_x2 - cropped_box_x1,
                             src_y2 - cropped_box_y1])
                 for src_x1, src_y1, src_x2, src_y2 in src_boxes]
    return dst_boxes


def main():
    screen_label = "screen"
    src_dataset_dir = "small"
    dst_dataset_dir = "small_cropped"

    relative_pad_w = .5
    relative_pad_h = 1

    for annotation_path in sorted(glob(os.path.join(src_dataset_dir, "*.xml"))):
        src_boxes, labels = PascalVocXmlParser(annotation_path).annotation()
        src_image, image_file_name = load_annotated_image(annotation_path)
        src_screen_box = next((b for b, l in zip(src_boxes, labels) if l == screen_label))

        region_box = cropped_image_box(src_screen_box, relative_pad_w, relative_pad_h, imWH(src_image))
        cropped_image = image_by_box(src_image, region_box)
        cropped_boxes = crop_boxes(src_boxes, region_box)

        dst_image_path = os.path.join(dst_dataset_dir, image_file_name)
        cv2.imwrite(dst_image_path, cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

        annotation_file_name = os.path.split(annotation_path)[1]
        PascalVocWriter.write(
            annotation_path=os.path.join(dst_dataset_dir, annotation_file_name),
            image_path=dst_image_path,
            image_shape=cropped_image.shape,
            boxes=cropped_boxes,
            labels=labels
        )


main()
