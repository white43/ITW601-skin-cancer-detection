from typing import Any

import numpy as np
import numpy.typing as npt
from albumentations import ImageOnlyTransform

from common.shades_of_grey import shades_of_grey


class ShadesOfGrey(ImageOnlyTransform):
    norm_p: int = 0
    to_linear_lut: npt.ArrayLike | None = None

    def __init__(self,
        norm_p: int = 6,
        gamma: float | None = None,
        always_apply: bool = False,
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
        return shades_of_grey(
            img=img,
            norm_p=self.norm_p,
            lut=self.to_linear_lut,
        )
