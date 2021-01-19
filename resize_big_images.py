from trvo_utils import TODO
import os

from utils.resize_dataset_lib import resize_images_in_dir


def main():
    desired_size = (1280, 960)
    src_dst_dataset_dirs = [
        ("data/Musson_counters_2", "data/Musson_counters_2_1280x960"),
        ("data/Musson_counters_3", "data/Musson_counters_3_1280x960")
    ]
    for src_dir, dst_dir in src_dst_dataset_dirs:
        os.makedirs(dst_dir, exist_ok=True)
        resize_images_in_dir(src_dir, dst_dir, desired_size, force_desired_size=False)
        print(f"{src_dir} -> {dst_dir} done.")


if __name__ == '__main__':
    main()
