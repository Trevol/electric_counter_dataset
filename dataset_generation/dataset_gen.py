import os
from glob import glob

from cv2 import cv2
from trvo_utils import TODO
from trvo_utils.annotation import PascalVocXmlParser
from trvo_utils.cv2gui_utils import imshowWait

from dataset_generation.augmentations import Augmentations
from dataset_generation.box_drawer import box_drawer


class Dataset:
    def __init__(self, directory):
        self.directory = directory

    def item_paths(self):
        img_path_pattern = os.path.join(self.directory, "*.jpg")
        for img_path in sorted(glob(img_path_pattern)):
            base_name = os.path.splitext(img_path)[0]
            ann_path = base_name + ".xml"
            yield img_path, ann_path


def main():
    mobile_roi_wh = 400, 180
    src_dst_dataset_dirs = [
        ("../training_datasets/v0/Musson_counters", "../training_datasets/v0_generated/Musson_counters"),
        ("../training_datasets/v0/Musson_counters_3", "../training_datasets/v0_generated/Musson_counters_3")
    ]
    augmenter = Augmentations(p=1.0)
    for src_dataset_dir, dst_dataset_dir in src_dst_dataset_dirs:
        os.makedirs(dst_dataset_dir, exist_ok=True)
        for img_path, ann_path in Dataset(src_dataset_dir).item_paths():
            img = cv2.imread(img_path)
            if os.path.isfile(ann_path):
                boxes, labels = PascalVocXmlParser(ann_path).annotation()
            else:
                boxes, labels = [], []
            augm_img, augm_boxes, augm_labels = augmenter(img, boxes, labels)
            key = imshowWait(
                box_drawer.xyxy(img, boxes),
                box_drawer.xyxy(augm_img, augm_boxes)
            )
            if key == 27: break
            # TODO()
        # TODO()


if __name__ == '__main__':
    main()
