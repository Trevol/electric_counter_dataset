import os

import cv2
from trvo_utils import toInt_array
from trvo_utils.annotation import PascalVocXmlParser, voc_to_yolo

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree

from trvo_utils.imutils import imgByBox, IMAGES_EXTENSIONS

from trvo_utils.path_utils import list_files

from dataset_paths import datasetDescriptions


def digitsAnnotationFile(imgFile, annotations_dir):
    if annotations_dir is None:
        return None
    parent, file_ = os.path.split(imgFile)
    nameWithoutExt = os.path.splitext(file_)[0]
    ann_file = os.path.join(parent, annotations_dir, nameWithoutExt + '.xml')
    if os.path.isfile(ann_file):
        return ann_file
    return None


def shiftBoxes(boxes, x, y):
    return [(x1 + x, y1 + y, x2 + x, y2 + y) for x1, y1, x2, y2 in boxes]


def extract_screenImg_digitsAnnotations(img_file, ann_file):
    assert ann_file is not None
    p = PascalVocXmlParser(ann_file)

    screenBoxes = []
    digitBoxes = []
    digitLabels = []
    for b, l in zip(p.boxes(), p.labels()):
        if l == 'screen':
            screenBoxes.append(b)
        elif l == 'x':
            continue
        else:
            digitBoxes.append(b)
            digitLabels.append(l)
    if len(screenBoxes) == 0:
        raise Exception(f'screenBoxes is empty. {img_file} {ann_file}')

    extraSpace = 5
    for screenBox in screenBoxes:
        # find digit boxes inside screen
        digitsInScreenBoxes, digitsInScreenLabels = findInnerBoxesAndLabels(screenBox, digitBoxes, digitLabels)
        screenImg = imgByBox(cv2.imread(img_file), screenBox, extraSpace=extraSpace)
        screenLeftX, screenTopY, *_ = screenBox
        digitsInScreenBoxes = shiftBoxes(digitsInScreenBoxes, -(screenLeftX - extraSpace), -(screenTopY - extraSpace))
        yield screenImg, digitsInScreenBoxes, digitsInScreenLabels


def findInnerBoxesAndLabels(outerBox, testBoxes, testLabels):
    innerBoxes = []
    innerLabels = []
    for testBox, testLabel in zip(testBoxes, testLabels):
        if isInnerBox(outerBox, testBox):
            innerBoxes.append(testBox)
            innerLabels.append(testLabel)
    return innerBoxes, innerLabels


def isInnerBox(outerBox, testBox):
    ox1, oy1, ox2, oy2 = outerBox
    tx1, ty1, tx2, ty2 = testBox
    return ox1 <= tx1 and oy1 <= ty1 and ox2 >= tx2 and oy2 >= ty2


def SubElement(parent, tag, text="", attrib={}):
    subEl = ET.SubElement(parent, tag, attrib)
    subEl.text = str(text or "")
    return subEl


def writeAnnotation(annFile, imgFile, imgShape, boxes, labels):
    assert len(boxes)
    assert len(labels)

    root = ET.Element("annotation")
    SubElement(root, 'filename', os.path.basename(imgFile))
    SubElement(root, 'path', imgFile)
    sizeEl = SubElement(root, 'size')
    h, w, d = imgShape
    SubElement(sizeEl, 'width', w)
    SubElement(sizeEl, 'height', h)
    SubElement(sizeEl, 'depth', d)

    for (x1, y1, x2, y2), l in zip(boxes, labels):
        objEl = SubElement(root, 'object')
        SubElement(objEl, 'name', l)
        bndboxEl = SubElement(objEl, 'bndbox')
        SubElement(bndboxEl, 'xmin', x1)
        SubElement(bndboxEl, 'ymin', y1)
        SubElement(bndboxEl, 'xmax', x2)
        SubElement(bndboxEl, 'ymax', y2)

    tree = ElementTree(element=root)
    tree.write(annFile)


def extract_dataset(datasetDescriptions):
    digitsDirs = [os.path.join(d.image_path, d.digits_dir) for d in datasetDescriptions if d.digits_dir is not None]
    for d in digitsDirs:
        os.makedirs(d, exist_ok=True)

    results = []
    for d in datasetDescriptions:
        if d.digits_annotations_dir0 is None:
            continue
        for img_file in list_files([d.image_path], IMAGES_EXTENSIONS):
            ann_file = digitsAnnotationFile(img_file, d.digits_annotations_dir0)
            if ann_file is None:
                continue
            for screenImg, digitBoxes, digitLabels in extract_screenImg_digitsAnnotations(img_file, ann_file):
                if len(digitBoxes):
                    results.append((img_file, d.digits_dir, screenImg, digitBoxes, digitLabels))

    for img_file, digits_dir, screenImg, digitBoxes, digitLabels in results:
        parentDir, imgBaseName = os.path.split(img_file)
        nameWithoutExt = os.path.splitext(imgBaseName)[0]
        screenImgFile = os.path.join(parentDir, digits_dir, imgBaseName)
        annFile = os.path.join(parentDir, digits_dir, nameWithoutExt + '.xml')

        cv2.imwrite(screenImgFile, screenImg, [cv2.IMWRITE_JPEG_QUALITY, 100])
        writeAnnotation(annFile, screenImgFile, screenImg.shape, digitBoxes, digitLabels)

    labels = [str(i) for i in range(10)]
    voc_to_yolo.convert(labels, digitsDirs)


def __main():
    extract_dataset(datasetDescriptions[-2:])


if __name__ == '__main__':
    __main()
