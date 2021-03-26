import os
from dataclasses import dataclass
from glob import glob
from shutil import copy2
from typing import List


@dataclass
class SrcDstMap:
    src_dir: str
    dst_dir: str


@dataclass
class AnnotatedImage:
    img_file: str
    ann_file: str


class Search:
    def __init__(self, src_dir):
        self.src_dir = src_dir

    @staticmethod
    def __annotations(src_dir):
        pattern = os.path.join(src_dir, "**", "*.xml")
        return sorted(glob(pattern, recursive=True))

    @staticmethod
    def __img_file_by_annotation(ann_file):
        (base, ext) = os.path.splitext(ann_file)
        return base + '.jpg'

    def annotated_images(self) -> List[AnnotatedImage]:
        result = list()
        for ann_file in self.__annotations(self.src_dir):
            img_file = self.__img_file_by_annotation(ann_file)
            assert (os.path.isfile(img_file))
            result.append(AnnotatedImage(img_file, ann_file))
        return result


class Copy:
    def __init__(self, src_dst_map: SrcDstMap):
        self.src_dir = self.__ensure_path_sep(src_dst_map.src_dir)
        self.dst_dir = self.__ensure_path_sep(src_dst_map.dst_dir)

    @staticmethod
    def __ensure_path_sep(path: str):
        # at the end of path
        return os.path.join(path, "")

    @classmethod
    def __copy(cls, src_path, dst_path):
        assert (os.path.isfile(src_path))
        cls.__make_parent(dst_path)
        copy2(src_path, dst_path)

    @staticmethod
    def __make_parent(file_path):
        dst_dir, _ = os.path.split(file_path)
        os.makedirs(dst_dir, exist_ok=True)

    def annotated_image(self, ann_img: AnnotatedImage):
        # copy each image/annotation to dst (with relative directory)
        assert (ann_img.img_file.startswith(self.src_dir))
        assert (ann_img.ann_file.startswith(self.src_dir))
        self.__copy(
            src_path=ann_img.img_file,
            dst_path=ann_img.img_file.replace(self.src_dir, self.dst_dir, 1)
        )
        self.__copy(
            src_path=ann_img.ann_file,
            dst_path=ann_img.ann_file.replace(self.src_dir, self.dst_dir, 1)
        )


def main():
    src_dst_maps = [
        SrcDstMap("data/recording_from_musson/1_trvo", "training_datasets/v2/1_trvo")
    ]

    for src_dst_map in src_dst_maps:
        search = Search(src_dst_map.src_dir)
        copy = Copy(src_dst_map)
        for ann_img in search.annotated_images():
            copy.annotated_image(ann_img)


if __name__ == '__main__':
    main()
