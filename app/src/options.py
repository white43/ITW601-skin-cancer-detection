from typing import Any

class ClsModel:
    def __init__(self, name: str, augmentations: dict[str, dict[str, Any]] = None):
        self.name = name
        self.augmentations: dict[str, dict[str, Any]] = augmentations

class Options:
    def __init__(self):
        self.cls_models: list[ClsModel] = []
        self.cls_model: str = ""
        self.cls_augmentations: dict[str, dict[str, Any]] = {}
        self.seg_model: str = ""
        self.download_from: str = "https://torrens-files.s3.ap-southeast-2.amazonaws.com/ITA602"
        self.debug: bool = False
        self.debug_diameter: bool = False
        self.debug_asymmetry: bool = False
