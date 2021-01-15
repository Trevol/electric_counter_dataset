from utils.dataset_to_tfrecord_lib import dataset_to_tfrecord


def main_all():
    all_classes = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "screen"
    ]
    tfrecord_path = 'digits_cropped.record'
    dataset_dir = "small_cropped"
    totalNumOfObjects, numOfImages = dataset_to_tfrecord(
        dataset_dir=dataset_dir,
        all_classes=all_classes,
        tfrecord_path=tfrecord_path
    )
    print(f"{numOfImages} images/{totalNumOfObjects} objects was written to {tfrecord_path}")


def main_only_screens():
    screen = "screen"
    all_classes = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        screen
    ]
    tfrecord_path = 'screens_without_digits.record'
    totalNumOfObjects, numOfImages = dataset_to_tfrecord(
        dataset_dir="small",
        all_classes=all_classes,
        tfrecord_path=tfrecord_path,
        desired_classes=[screen]
    )
    print(f"{numOfImages} images/{totalNumOfObjects} objects was written to {tfrecord_path}")


if __name__ == '__main__':
    main_all()
