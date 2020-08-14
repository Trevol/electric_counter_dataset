from collections import namedtuple
from typing import List

DatasetDescriptor = namedtuple("DatasetDescriptor", "image_path, digits_annotations_dir0, digits_dir")
dd = DatasetDescriptor

datasetDescriptions: List[DatasetDescriptor] = [
    dd("/hdd/Datasets/counters/data/0_from_internet/train", None, None),
    dd("/hdd/Datasets/counters/data/0_from_internet/val", "digits_annotations", "digits"),
    dd("/hdd/Datasets/counters/data/1_from_phone/train", "digits_annotations", "digits"),
    dd("/hdd/Datasets/counters/data/1_from_phone/val", "digits_annotations", "digits"),
    dd("/hdd/Datasets/counters/data/2_from_phone/train", "digits_annotations", "digits"),
    dd("/hdd/Datasets/counters/data/2_from_phone/val", "digits_annotations", "digits"),
    dd("/hdd/Datasets/counters/data/3_from_phone", "digits_annotations", "digits"),
    dd("/hdd/Datasets/counters/data/4_from_phone", "digits_annotations", "digits"),
    dd("/hdd/Datasets/counters/data/5_from_phone", "digits_annotations", "digits"),
    dd("/hdd/Datasets/counters/data/6_from_phone", "digits_annotations", "digits"),
    dd("/hdd/Datasets/counters/data/7_from_app", "digits_annotations", "digits"),
    dd("/hdd/Datasets/counters/data/8_from_phone", "digits_annotations", "digits"),
    dd("/hdd/Datasets/counters/data/Musson_counters", "digits_annotations", "digits"),
    dd("/hdd/Datasets/counters/data/Musson_counters_2", "", "digits"),
    dd("/hdd/Datasets/counters/data/Musson_counters_3", "", "digits")
]

image_paths: List[str] = [d.image_path for d in datasetDescriptions]
