class box_utils:
    @staticmethod
    def width(xyxy_box):
        # 0123
        # xyxy
        return xyxy_box[2] - xyxy_box[0]  # x2-x1

    @staticmethod
    def height(xyxy_box):
        # 0123
        # xyxy
        return xyxy_box[3] - xyxy_box[1]  # y2-y1

    @staticmethod
    def x1(xyxy_box):
        # 0123
        # xyxy
        return xyxy_box[0]

    @staticmethod
    def x2(xyxy_box):
        # 0123
        # xyxy
        return xyxy_box[2]

    @staticmethod
    def y1(xyxy_box):
        # 0123
        # xyxy
        return xyxy_box[1]

    @staticmethod
    def y2(xyxy_box):
        # 0123
        # xyxy
        return xyxy_box[3]

    @staticmethod
    def size_wh(xyxy_box):
        return box_utils.width(xyxy_box), box_utils.height(xyxy_box)

    @staticmethod
    def area(xyxy_box):
        return box_utils.width(xyxy_box) * box_utils.height(xyxy_box)
