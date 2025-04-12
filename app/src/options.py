from typing import Any


class Options:
    def __init__(self):
        self.cls_model: str = ""
        self.cls_augmentations: dict[str, dict[str, Any]] = {}
        self.seg_model: str = ""
        self.download_from: str = "https://torrens-files.s3.ap-southeast-2.amazonaws.com/ITA602"
