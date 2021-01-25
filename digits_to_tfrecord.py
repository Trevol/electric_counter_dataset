from utils.dataset_to_tfrecord_lib import dataset_directory_to_tfrecord, dataset_directories_to_tfrecord


def main_all():
    all_classes = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "screen"
    ]
    class2index_map = {clazz: indx + 1 for indx, clazz in enumerate(all_classes)}
    tfrecord_path = 'training_datasets/v1_generated/digits_generated.record'
    dataset_dirs = [
        "training_datasets/v1_generated/Musson_counters",
        "training_datasets/v1_generated/Musson_counters_3"
    ]
    totalNumOfObjects, numOfImages = dataset_directories_to_tfrecord(
        dataset_dirs=dataset_dirs,
        class2index_map=class2index_map,
        tfrecord_path=tfrecord_path
    )
    print(f"{numOfImages} images/{totalNumOfObjects} objects was written to {tfrecord_path}")


if __name__ == '__main__':
    main_all()
