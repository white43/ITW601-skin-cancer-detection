import cv2

# Available labels in ISIC2018 dataset
LABELS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# Initial crop size that is carried out before downscaling images to 224 pixels
CROP_SIZE = 448

INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "nearest-exact": cv2.INTER_NEAREST_EXACT,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}

# https://keras.io/api/applications/#available-models
INPUT_SIZES = {
    "efficientnetv2-b0": 224,
    # "efficientnetv2-b1": 240,
    # "efficientnetv2-b2": 260,
    # "efficientnetv2-b3": 300,
    "efficientnetv2-s": 384,
    "resnet50v2": 224,
    "densenet121": 224,
    "densenet169": 224,
    "convnexttiny": 224,
}
