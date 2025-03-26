import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import zipfile
from pathlib import Path

import cv2
import keras
import numpy as np
from PIL import Image
from tqdm import tqdm

URL_PREFIX = "https://isic-challenge-data.s3.amazonaws.com/2018/"

SEGMENTATION_TRAINING_INPUT = "ISIC2018_Task1-2_Training_Input"
SEGMENTATION_TRAINING_GROUND_TRUTH = "ISIC2018_Task1_Training_GroundTruth"

SEGMENTATION_TEST_INPUT = "ISIC2018_Task1-2_Test_Input"
SEGMENTATION_TEST_GROUND_TRUTH = "ISIC2018_Task1_Test_GroundTruth"

cli_opts = argparse.ArgumentParser()
cli_opts.add_argument("--cache", type=str, default="isic2018-datasets")
cli_opts.add_argument("--target", type=str, default="isic2018-segmentation")
options = cli_opts.parse_args()

if options.cache[0] != "/":
    options.cache = os.path.join(os.getcwd(), options.cache)

if options.target[0] != "/":
    options.target = os.path.join(os.getcwd(), options.target)


def extract_images(archive: str, dest: str):
    print("Extracting %s to %s" % (archive, dest))
    counter = 0

    with zipfile.ZipFile(os.path.join(archive)) as zp:
        for file in tqdm(zp.filelist):
            basename = os.path.basename(file.filename)

            if (basename[-3:] == "png" or basename[-3:] == "jpg") and not os.path.exists(os.path.join(dest, basename)):
                with zp.open(file.filename) as source, open(os.path.join(dest, basename), "wb") as target:
                    shutil.copyfileobj(source, target)
                    counter += 1

    print("Extracted %d files to %s from %s" % (counter, dest, archive))


def extract_optimized_images(archive: str, dest: str):
    with zipfile.ZipFile(os.path.join(archive)) as zp:
        for file in zp.filelist:
            basename = os.path.basename(file.filename)

            if (basename[-3:] == "png" or basename[-3:] == "jpg") and not os.path.exists(os.path.join(dest, basename)):
                with Image.open(zp.open(file.filename)) as img:
                    img.save(os.path.join(dest, basename), "jpeg", quality=95)


# Author: Ultralytics
# Link: https://github.com/ultralytics/ultralytics/blob/16fc32530809b74961aabf26a39d8a6461948d17/ultralytics/data/converter.py#L337
#
# This is a slightly modified version that takes a dict with pixel intensities as keys and classes as values. This was
# done to avoid an additional step during downloading as the original does not support binary black/white masks.
def convert_segment_masks_to_yolo_seg(masks_dir, output_dir, pixel_to_class_mapping):
    """
    Converts a dataset of segmentation mask images to the YOLO segmentation format.

    This function takes the directory containing the binary format mask images and converts them into YOLO segmentation format.
    The converted masks are saved in the specified output directory.

    Args:
        masks_dir (str): The path to the directory where all mask images (png, jpg) are stored.
        output_dir (str): The path to the directory where the converted YOLO segmentation masks will be stored.
        pixel_to_class_mapping (dict): The dictionary with pixel values in the 0-255 range and classes

    Example:
        ```python
        from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

        # The classes here is the total classes in the dataset, for COCO dataset we have 80 classes
        convert_segment_masks_to_yolo_seg("path/to/masks_directory", "path/to/output/directory", classes=80)
        ```

    Notes:
        The expected directory structure for the masks is:

            - masks
                ├─ mask_image_01.png or mask_image_01.jpg
                ├─ mask_image_02.png or mask_image_02.jpg
                ├─ mask_image_03.png or mask_image_03.jpg
                └─ mask_image_04.png or mask_image_04.jpg

        After execution, the labels will be organized in the following structure:

            - output_dir
                ├─ mask_yolo_01.txt
                ├─ mask_yolo_02.txt
                ├─ mask_yolo_03.txt
                └─ mask_yolo_04.txt
    """
    for mask_path in tqdm(Path(masks_dir).iterdir()):
        if mask_path.suffix == ".png":
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Read the mask image in grayscale
            img_height, img_width = mask.shape  # Get image dimensions

            unique_values = np.unique(mask)  # Get unique pixel values representing different classes
            yolo_format_data = []

            for value in unique_values:
                if value == 0:
                    continue  # Skip background
                class_index = pixel_to_class_mapping.get(value, -1)
                if class_index == -1:
                    print(f"Unknown class for pixel value {value} in file {mask_path}, skipping.")
                    continue

                # Create a binary mask for the current class and find contours
                contours, _ = cv2.findContours(
                    (mask == value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )  # Find contours

                for contour in contours:
                    if len(contour) >= 3:  # YOLO requires at least 3 points for a valid segmentation
                        contour = contour.squeeze()  # Remove single-dimensional entries
                        yolo_format = [class_index]
                        for point in contour:
                            # Normalize the coordinates
                            yolo_format.append(round(point[0] / img_width, 6))  # Rounding to 6 decimal places
                            yolo_format.append(round(point[1] / img_height, 6))
                        yolo_format_data.append(yolo_format)
            # Save Ultralytics YOLO format data to file
            # ISIC_0012169_segmentation
            output_path = Path(output_dir) / f"{mask_path.stem[:-13]}.txt"
            with open(output_path, "w") as file:
                for item in yolo_format_data:
                    line = " ".join(map(str, item))
                    file.write(line + "\n")

os.makedirs(options.cache, mode=0o755, exist_ok=True)
os.makedirs(options.target, mode=0o755, exist_ok=True)

# Download and extract a training dataset
if not os.path.exists(os.path.join(options.target, "train", "images")):
    keras.utils.get_file(
        origin=URL_PREFIX + SEGMENTATION_TRAINING_INPUT + ".zip",
        cache_subdir=options.cache,
        extract=False,
    )

    os.makedirs(os.path.join(options.target, "train", "images"), mode=0o755, exist_ok=True)

    extract_images(
        os.path.join(options.cache, SEGMENTATION_TRAINING_INPUT + ".zip"),
        os.path.join(options.target, "train", "images"),
    )

# Download and extract labels for a training dataset
if not os.path.exists(os.path.join(options.target, "train", "labels")):
    keras.utils.get_file(
        origin=URL_PREFIX + SEGMENTATION_TRAINING_GROUND_TRUTH + ".zip",
        cache_subdir=options.cache,
        extract=False,
    )

    os.makedirs(os.path.join(options.target, "train", "labels"), mode=0o755, exist_ok=True)
    os.makedirs(os.path.join(options.target, "train", "masks"), mode=0o755, exist_ok=True)

    extract_images(
        os.path.join(options.cache, SEGMENTATION_TRAINING_GROUND_TRUTH + ".zip"),
        os.path.join(options.target, "train", "masks"),
    )

    convert_segment_masks_to_yolo_seg(
        os.path.join(options.cache, "train", "masks"),
        os.path.join(options.target, "train", "labels"),
        {255: 0}
    )

# Download and extract a validation dataset
if not os.path.exists(os.path.join(options.target, "val", "images")):
    keras.utils.get_file(
        origin=URL_PREFIX + SEGMENTATION_TEST_INPUT + ".zip",
        cache_subdir=options.cache,
        extract=False,
    )

    os.makedirs(os.path.join(options.target, "val", "images"), mode=0o755, exist_ok=True)

    extract_images(
        os.path.join(options.cache, SEGMENTATION_TEST_INPUT + ".zip"),
        os.path.join(options.target, "val", "images"),
    )

# Download and extract labels for a validation dataset
if not os.path.exists(os.path.join(options.target, "val", "labels")):
    keras.utils.get_file(
        origin=URL_PREFIX + SEGMENTATION_TEST_GROUND_TRUTH + ".zip",
        cache_subdir=options.cache,
        extract=False,
    )

    os.makedirs(os.path.join(options.target, "val", "labels"), mode=0o755, exist_ok=True)
    os.makedirs(os.path.join(options.target, "val", "masks"), mode=0o755, exist_ok=True)

    extract_images(
        os.path.join(options.cache, SEGMENTATION_TEST_GROUND_TRUTH + ".zip"),
        os.path.join(options.target, "val", "masks"),
    )

    convert_segment_masks_to_yolo_seg(
        os.path.join(options.cache, "val", "masks"),
        os.path.join(options.target, "val", "labels"),
        {255: 0}
    )
