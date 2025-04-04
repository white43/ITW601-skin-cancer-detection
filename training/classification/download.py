import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import zipfile

import keras
import pandas as pd
from tqdm import tqdm

URL_PREFIX = "https://isic-challenge-data.s3.amazonaws.com/2018/"

CLASSIFICATION_TRAINING_INPUT = "ISIC2018_Task3_Training_Input"
CLASSIFICATION_TRAINING_GROUND_TRUTH = "ISIC2018_Task3_Training_GroundTruth"

CLASSIFICATION_TEST_INPUT = "ISIC2018_Task3_Test_Input"
CLASSIFICATION_TEST_GROUND_TRUTH = "ISIC2018_Task3_Test_GroundTruth"

LABELS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

cli_opts = argparse.ArgumentParser()
cli_opts.add_argument("--cache", type=str, default="isic2018-datasets")
cli_opts.add_argument("--target", type=str, default="isic2018-classification")
cli_opts.add_argument("--clean", action='store_true', default=False, help="Remove downloaded ZIP archives at the end")
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


def extraxt_csv(archive: str, dest: str):
    print("Extracting %s to %s" % (archive, dest))

    with zipfile.ZipFile(os.path.join(archive)) as zp:
        for file in zp.filelist:
            basename = os.path.basename(file.filename)

            if basename[-3:] == "csv" and not os.path.exists(os.path.join(dest, basename)):
                with zp.open(file.filename) as source, open(os.path.join(dest, basename), "wb") as target:
                    shutil.copyfileobj(source, target)

    print("Extracted a file to %s from %s" % (dest, archive))


os.makedirs(options.cache, mode=0o755, exist_ok=True)
os.makedirs(options.target, mode=0o755, exist_ok=True)

# Download and extract a training dataset
if not os.path.exists(os.path.join(options.target, "train")):
    keras.utils.get_file(
        origin=URL_PREFIX + CLASSIFICATION_TRAINING_INPUT + ".zip",
        cache_subdir=options.cache,
        extract=False,
    )

    os.makedirs(os.path.join(options.target, "train"), mode=0o755, exist_ok=True)

    extract_images(
        os.path.join(options.cache, CLASSIFICATION_TRAINING_INPUT + ".zip"),
        os.path.join(options.target, "train"),
    )

    if options.clean:
        os.unlink(os.path.join(options.cache, CLASSIFICATION_TRAINING_INPUT + ".zip"))

# Download and extract a validation dataset
if not os.path.exists(os.path.join(options.target, "val")):
    keras.utils.get_file(
        origin=URL_PREFIX + CLASSIFICATION_TEST_INPUT + ".zip",
        cache_subdir=options.cache,
        extract=False,
    )

    os.makedirs(os.path.join(options.target, "val"), mode=0o755, exist_ok=True)

    extract_images(
        os.path.join(options.cache, CLASSIFICATION_TEST_INPUT + ".zip"),
        os.path.join(options.target, "val"),
    )

    if options.clean:
        os.unlink(os.path.join(options.cache, CLASSIFICATION_TEST_INPUT + ".zip"))

# Extract CSV files from ground truth zip files
if not os.path.exists(os.path.join(options.cache, CLASSIFICATION_TRAINING_GROUND_TRUTH + ".csv")):
    # Download ground truth for training and validation datasets
    keras.utils.get_file(
        origin=URL_PREFIX + CLASSIFICATION_TRAINING_GROUND_TRUTH + ".zip",
        cache_subdir=options.cache,
        extract=False,
    )

    extraxt_csv(
        os.path.join(options.cache, CLASSIFICATION_TRAINING_GROUND_TRUTH + ".zip"),
        os.path.join(options.cache),
    )

    if options.clean:
        os.unlink(os.path.join(options.cache, CLASSIFICATION_TRAINING_GROUND_TRUTH + ".zip"))

if not os.path.exists(os.path.join(options.cache, CLASSIFICATION_TEST_GROUND_TRUTH + ".csv")):
    keras.utils.get_file(
        origin=URL_PREFIX + CLASSIFICATION_TEST_GROUND_TRUTH + ".zip",
        cache_subdir=options.cache,
        extract=False,
    )

    extraxt_csv(
        os.path.join(options.cache, CLASSIFICATION_TEST_GROUND_TRUTH + ".zip"),
        os.path.join(options.cache),
    )

    if options.clean:
        os.unlink(os.path.join(options.cache, CLASSIFICATION_TEST_GROUND_TRUTH + ".zip"))

df_train = pd.read_csv(os.path.join(options.cache, CLASSIFICATION_TRAINING_GROUND_TRUTH + ".csv"))
df_val = pd.read_csv(os.path.join(options.cache, CLASSIFICATION_TEST_GROUND_TRUTH + ".csv"))

print(df_train)
print(df_val)

# Create subdirectories for training and validation datasets
for label in LABELS:
    if not os.path.exists(os.path.join(options.target, "train", label)):
        os.makedirs(os.path.join(options.target, "train", label), mode=0o755)

    if not os.path.exists(os.path.join(options.target, "val", label)):
        os.makedirs(os.path.join(options.target, "val", label), mode=0o755)

    for _, row in tqdm(df_train[df_train[label] == 1].iterrows(), desc='Moving train/%s' % label):
        source_path = os.path.join(options.target, "train", row["image"] + ".jpg")

        if os.path.exists(source_path):
            target_path = os.path.join(options.target, "train", label, row["image"] + ".jpg")
            os.replace(source_path, target_path)

    for _, row in tqdm(df_val[df_val[label] == 1].iterrows(), desc='Moving val/%s' % label):
        source_path = os.path.join(options.target, "val", row["image"] + ".jpg")

        if os.path.exists(source_path):
            target_path = os.path.join(options.target, "val", label, row["image"] + ".jpg")
            os.replace(source_path, target_path)
