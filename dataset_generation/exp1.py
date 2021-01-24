import numpy as np
from trvo_utils import toInt_array
from trvo_utils.core import Rect
from trvo_utils.cv2gui_utils import imshowWait
from trvo_utils.imutils import imWH, img_by_xyxy_box_unsafe

from utils.box_utils import box_utils


def _view_img(src_img, view_box, fill_value):
    view_box = toInt_array(view_box)

    view_img = np.full([*box_utils.size_hw(view_box), 3],
                       fill_value, np.uint8)
    # in image coords
    src_img_rect = Rect([0, 0, *imWH(src_img)])
    view_rect = Rect.fromXyxy(view_box)
    src_img_part = img_by_xyxy_box_unsafe(
        src_img,
        src_img_rect.intersection(view_rect).xyxy
    )

    # in view box coords
    view_rect = Rect([0, 0,
                      box_utils.width(view_box),
                      box_utils.height(view_box)])
    src_img_rect = Rect(
        [
            -box_utils.x1(view_box),
            -box_utils.y1(view_box),
            *imWH(src_img)
        ]
    )
    dst_box = view_rect.intersection(src_img_rect).xyxy
    x1, y1, x2, y2 = dst_box
    view_img[y1:y2, x1:x2] = src_img_part
    return view_img


def main():
    green = 0, 255, 0
    img = np.full([200, 400, 3], green, np.uint8)
    view_box = [-50, 50, 100, 100]
    fill_value = 127

    view_img = _view_img(img, view_box, fill_value)
    imshowWait(img, view_img)


def main_():
    # In image coords
    img_box = Rect([0, 0, 200, 400])
    # view_box = Rect.fromXyxy([-50, 50, 100, 100])
    view_box = Rect.fromXyxy([50, 50, 100, 100])
    assert img_box.intersection(view_box).xyxy == [50, 50, 100, 100]
    assert view_box.intersection(img_box).xyxy == [50, 50, 100, 100]

    # In view_box coords
    view_box = Rect.fromXyxy([0, 0, 50, 50])
    img_box = Rect([-50, -50, 200, 400])
    assert img_box.intersection(view_box).xyxy == [0, 0, 50, 50]
    assert view_box.intersection(img_box).xyxy == [0, 0, 50, 50]


if __name__ == '__main__':
    main()
