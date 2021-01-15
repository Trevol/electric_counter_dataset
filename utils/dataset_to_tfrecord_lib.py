import os
import io
import glob
import hashlib
from typing import Dict, List

import tensorflow as tf
import random

from PIL import Image
from object_detection.utils import dataset_util
from trvo_utils.annotation import PascalVocXmlParser


def create_example(xml_file, class_text_to_index_dict: Dict, desired_classes: List[str] = None):
    if desired_classes is None:
        desired_classes = []
    parser = PascalVocXmlParser(xml_file)

    image_name = parser.filename()
    file_name = image_name.encode('utf8')
    width, height = parser.size()

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    numOfObjects = 0
    boxes, labels = parser.annotation()
    for box, class_text in zip(boxes, labels):
        if len(desired_classes) > 0 and class_text not in desired_classes:
            continue
        classes_text.append(class_text.encode('utf8'))
        x1, y1, x2, y2 = box
        xmin.append(float(x1) / width)
        ymin.append(float(y1) / height)
        xmax.append(float(x2) / width)
        ymax.append(float(y2) / height)
        difficult_obj.append(0)

        classes.append(class_text_to_index_dict[class_text])
        truncated.append(0)
        poses.append('Unspecified'.encode('utf8'))
        numOfObjects += 1

    # read corresponding image
    parent_dir = os.path.split(xml_file)[0]
    full_path = os.path.join(parent_dir, image_name)  # provide the path of images directory
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    w, h = image.size
    assert w == width and h == height  # check real size with size specified in annotation
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    # create TFRecord Example
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(file_name),
        'image/source_id': dataset_util.bytes_feature(file_name),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example, numOfObjects


def __dataset_directory_to_tfrecord(writer: tf.io.TFRecordWriter,
                                    dataset_dir: str,
                                    class2index_map: Dict[str, int],
                                    desired_classes: List[str] = None,
                                    shuffle=True):
    annotation_path_list = glob.glob(os.path.join(dataset_dir, "*.xml"))
    if shuffle:
        random.shuffle(annotation_path_list)
    else:
        annotation_path_list = sorted(annotation_path_list)
    totalNumOfObjects = 0
    numOfImages = 0
    for xml_file in annotation_path_list:
        example, numOfExampleItems = create_example(xml_file, class2index_map, desired_classes)
        writer.write(example.SerializeToString())
        totalNumOfObjects += numOfExampleItems
        numOfImages += 1
    return totalNumOfObjects, numOfImages


def dataset_directory_to_tfrecord(dataset_dir: str,
                                  class2index_map: Dict[str, int],
                                  tfrecord_path: str,
                                  desired_classes: List[str] = None,
                                  shuffle=True):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        totalNumOfObjects, numOfImages = __dataset_directory_to_tfrecord(
            writer, dataset_dir, class2index_map, desired_classes, shuffle)
        return totalNumOfObjects, numOfImages


def dataset_directories_to_tfrecord(dataset_dirs: List[str],
                                    class2index_map: Dict[str, int],
                                    tfrecord_path: str,
                                    desired_classes: List[str] = None,
                                    shuffle=True):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        totalNumOfObjects, totalNumOfImages = 0, 0
        for dataset_dir in dataset_dirs:
            numOfObjects, numOfImages = __dataset_directory_to_tfrecord(
                writer, dataset_dir, class2index_map, desired_classes, shuffle)
            totalNumOfObjects += numOfObjects
            totalNumOfImages += numOfImages
        return totalNumOfObjects, totalNumOfImages
