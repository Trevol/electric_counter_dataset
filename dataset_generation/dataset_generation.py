import os
from glob import glob

from trvo_utils import TODO


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
    for src_dataset_dir, dst_dataset_dir in src_dst_dataset_dirs:
        assert os.path.isdir(src_dataset_dir)
        os.makedirs(dst_dataset_dir, exist_ok=True)
        for img_path, ann_path in Dataset(src_dataset_dir).item_paths():
            print(img_path, ann_path)
            TODO()
        TODO()


if __name__ == '__main__':
    main()
