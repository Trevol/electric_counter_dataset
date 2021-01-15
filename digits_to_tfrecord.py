from utils.dataset_to_tfrecord_lib import dataset_directory_to_tfrecord, dataset_directories_to_tfrecord


def main_all():
    all_classes = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "screen"
    ]
    class2index_map = {clazz: indx + 1 for indx, clazz in enumerate(all_classes)}
    tfrecord_path = 'training_datasets/v0/digits_cropped.record'
    dataset_dirs = [
        "training_datasets/v0/Musson_counters_3_cropped",
        "training_datasets/v0/Musson_counters_cropped"
    ]
    totalNumOfObjects, numOfImages = dataset_directories_to_tfrecord(
        dataset_dirs=dataset_dirs,
        class2index_map=class2index_map,
        tfrecord_path=tfrecord_path
    )
    print(f"{numOfImages} images/{totalNumOfObjects} objects was written to {tfrecord_path}")


def main_only_screens():
    screen = "screen"
    all_classes = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        screen
    ]
    class2index_map = {clazz: indx + 1 for indx, clazz in enumerate(all_classes)}
    tfrecord_path = 'training_datasets/v0/screens_without_digits.record'
    dataset_dirs = [
        "training_datasets/v0/Musson_counters_3",
        "training_datasets/v0/Musson_counters"
    ]
    totalNumOfObjects, numOfImages = dataset_directories_to_tfrecord(
        dataset_dirs=dataset_dirs,
        class2index_map=class2index_map,
        tfrecord_path=tfrecord_path,
        desired_classes=[screen]
    )
    print(f"{numOfImages} images/{totalNumOfObjects} objects was written to {tfrecord_path}")


if __name__ == '__main__':
    main_all()
