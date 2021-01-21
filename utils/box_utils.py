class box_utils:
    @staticmethod
    def width(box):
        # 0123
        # xyxy
        return box[2] - box[0]  # x2-x1

    @staticmethod
    def height(box):
        # 0123
        # xyxy
        return box[3] - box[1]  # y2-y1

    @staticmethod
    def size_wh(box):
        return box_utils.width(box), box_utils.height(box)

    @staticmethod
    def area(xyxy_box):
        return box_utils.width(xyxy_box) * box_utils.height(xyxy_box)