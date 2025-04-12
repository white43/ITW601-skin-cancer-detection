import numpy as np

# User: Roberto Montagna
# Published: 30 Mar 2012
# Link: https://github.com/thecolourgroup/shadesofgrey/blob/8d2fb9c5e56455731ae66abfb1c3c6a3e7836409/shadesofgrey.c
def shades_of_grey(img: np.ndarray, norm_p: int = 6, lut: np.ndarray = None):
    if isinstance(img.dtype, np.uint8):
        raise Exception("shades_of_grey expects images to be of type uint8, %s given" % str(img.dtype))

    # L512-L531
    if lut is not None:
        img = lut[img]
    else:
        img = img / 255

    # L338-L351
    ill_est = np.power(np.sum(np.power(img, norm_p), axis=(0, 1)), 1 / norm_p)

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
    if lut is not None:
        img = np.where(img <= 0.00304, img * 12.92, ((img * 1.055) ** 1 / 2.4) - 0.055)

    img *= 255

    return img.astype(np.uint8)
