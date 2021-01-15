import xml.etree.ElementTree as et
import os

from trvo_utils import TODO


class PascalVocWriter:
    @classmethod
    def make_header_elements(cls, ann_root, image_path, image_shape):
        et.SubElement(ann_root, "folder").text = "Unspecified"
        et.SubElement(ann_root, "filename").text = os.path.split(image_path)[1]
        et.SubElement(ann_root, "path").text = image_path

        source_el = et.SubElement(ann_root, "source")
        et.SubElement(source_el, "database").text = "Unknown"

        size_el = et.SubElement(ann_root, "size")
        h, w, d = image_shape
        et.SubElement(size_el, "width").text = str(w)
        et.SubElement(size_el, "height").text = str(h)
        et.SubElement(size_el, "depth").text = str(d)

        et.SubElement(ann_root, "segmented").text = "0"

    @classmethod
    def make_object_element(cls, ann_root, box, label):
        obj_el = et.SubElement(ann_root, "object")
        et.SubElement(obj_el, "name").text = label
        et.SubElement(obj_el, "pose").text = "Unspecified"
        et.SubElement(obj_el, "truncated").text = "0"
        et.SubElement(obj_el, "difficult").text = "0"

        bndbox_el = et.SubElement(obj_el, "bndbox")
        x1, y1, x2, y2 = box
        et.SubElement(bndbox_el, "xmin").text = str(x1)
        et.SubElement(bndbox_el, "ymin").text = str(y1)
        et.SubElement(bndbox_el, "xmax").text = str(x2)
        et.SubElement(bndbox_el, "ymax").text = str(y2)

    @classmethod
    def write(cls, annotation_path, image_path, image_shape, boxes, labels):
        annotation_root = et.Element("annotation")
        cls.make_header_elements(annotation_root, image_path, image_shape)

        for box, label in zip(boxes, labels):
            cls.make_object_element(annotation_root, box, label)

        et.ElementTree(annotation_root).write(annotation_path, encoding="utf-8", xml_declaration=True)
