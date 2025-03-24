from typing import Any

import numpy as np
import numpy.typing as npt
from albumentations import ImageOnlyTransform


class ShadesOfGrey(ImageOnlyTransform):
    norm_p: int = 0
    to_linear_lut: npt.ArrayLike | None = None

    def __init__(self,
        norm_p: int = 6,
        gamma: float | None = None,
        always_apply: bool | None = None,
        p: float = 0.5):
        super().__init__(p=p, always_apply=always_apply)

        self.norm_p = norm_p

        if gamma is not None:
            self.to_linear_lut = np.array([
                i / 12.92 if i <= 0.03928 else (i + 0.055) / 1.055 ** gamma for i in (np.arange(256) / 255)
            ])

    # User: Roberto Montagna
    # Published: 30 Mar 2012
    # Link: https://github.com/thecolourgroup/shadesofgrey/blob/8d2fb9c5e56455731ae66abfb1c3c6a3e7836409/shadesofgrey.c
    def apply(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        if isinstance(img.dtype, np.uint8):
            raise Exception("ColorConstancy expects images to be of type uint8, %s given" % str(img.dtype))

        # L512-L531
        if self.to_linear_lut is not None:
            img = self.to_linear_lut[img]
        else:
            img = img / 255

        # L338-L351
        ill_est = np.power(np.sum(np.power(img, self.norm_p), axis=(0, 1)), 1/self.norm_p)

        # L359-L361
        ill_est /= img.size

        # L362-L364
        ill_norm = np.sqrt(np.sum(np.power(ill_est, 2)))

        # L366-L368
        ill_est /= ill_norm

        # L382-L389
        img /= ill_est

        # L392-L400
        if img.max() > 1:
            img /= img.max()

        # L415-L431
        if self.to_linear_lut is not None:
            img = np.where(img <= 0.00304, img * 12.92, ((img*1.055)**1/2.4)-0.055)

        img *= 255

        return img.astype(np.uint8)
