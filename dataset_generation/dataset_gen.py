import os

from trvo_utils import TODO
from trvo_utils.cv2gui_utils import imshowWait

from dataset_generation.augmentations import Augmentations
from dataset_generation.box_drawer import box_drawer
from utils.dataset_directory import DatasetDirectory


def main():
    mobile_roi_wh = 400, 180
    src_dst_dataset_dirs = [
        ("../training_datasets/v1/Musson_counters", "../training_datasets/v1_generated/Musson_counters"),
        ("../training_datasets/v1/Musson_counters_3_1280x960", "../training_datasets/v1_generated/Musson_counters_3")
    ]
    TODO("skip two (or more) annotated screens")
    TODO("If only screen is annotated (there is no annotated digits) - make this sample negative")
    TODO("Collect negative samples. Hard negative mining???")
    augmenter = Augmentations(p=1.0)
    for src_dataset_dir, dst_dataset_dir in src_dst_dataset_dirs:
        os.makedirs(dst_dataset_dir, exist_ok=True)
        for img, boxes, labels, img_path, ann_path, ann_exist in DatasetDirectory(src_dataset_dir).load_and_parse():
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
