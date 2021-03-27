import os
from glob import glob
import cv2
from trvo_utils.annotation import PascalVocXmlParser


class DatasetDirectory:
    def __init__(self, directory, recursive):
        self.directory = directory
        self.recursive = recursive

    def __item_paths(self):
        if self.recursive:
            img_path_pattern = os.path.join(self.directory, "**", "*.jpg")
        else:
            img_path_pattern = os.path.join(self.directory, "*.jpg")
        for img_path in sorted(glob(img_path_pattern, recursive=self.recursive)):
            base_name = os.path.splitext(img_path)[0]
            ann_path = base_name + ".xml"
            ann_exist = os.path.isfile(ann_path)
            yield img_path, ann_path, ann_exist

    def load_and_parse(self):
        for img_path, ann_path, ann_exist in self.__item_paths():
            img = cv2.imread(img_path)
            if ann_exist:
                boxes, labels = PascalVocXmlParser(ann_path).annotation()
            else:
                boxes, labels = [], []
            yield img, boxes, labels, img_path, ann_path, ann_exist
