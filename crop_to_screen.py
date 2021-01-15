from utils.crop_to_screen_lib import crop_to_screen_directory


def main():
    dataset_dirs = [
        ("training_datasets/v0/Musson_counters", "training_datasets/v0/Musson_counters_cropped"),
        ("training_datasets/v0/Musson_counters_3", "training_datasets/v0/Musson_counters_3_cropped")
    ]
    for src_dataset_dir, dst_dataset_dir in dataset_dirs:
        crop_to_screen_directory(src_dataset_dir, dst_dataset_dir)


if __name__ == '__main__':
    main()
