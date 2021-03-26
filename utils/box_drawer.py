import cv2
from trvo_utils import toInt_array


class box_drawer:
    red = 0, 0, 200

    @staticmethod
    def __rect(img, xyxy_rect, color, thickness=1):
        x1, y1, x2, y2 = toInt_array(xyxy_rect)
        return cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    @classmethod
    def xywh(cls, img, cocoBoxes, color=red):
        for x, y, w, h in cocoBoxes:
            cls.__rect(img, [x, y, x + w, y + h], color, thickness=1)
        return img

    @classmethod
    def xyxy(cls, img, xyxy_boxes, color=red):
        for xyxy_box in xyxy_boxes:
            cls.__rect(img, xyxy_box, color, thickness=1)
        return img
