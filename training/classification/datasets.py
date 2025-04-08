import os
from typing import Union

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from albumentations import BasicTransform, BaseCompose

from training.classification.constants import INTERPOLATIONS, DATASET_SIZE, CROP_SIZE
from training.classification.options import Options
from training.classification.shades_of_grey import ShadesOfGrey


def get_class_weights(dataset: tf.data.Dataset, labels: list[str]) -> dict[int, float]:
    """
    This function calculates class weights for a given dataset by iterating its contents once and counting class
    incidence.
    """
    class_incidence: dict[int, int] = {}
    class_weights: dict[int, float] = {}

    for image in dataset.as_numpy_iterator():
        image = image.decode('utf-8')
        directory = image.split(os.sep)[-2]
        index = labels.index(directory)

        if index not in class_incidence:
            class_incidence[index] = 0

        class_incidence[index] += 1

    total_labels = len(labels)
    total_samples = len(dataset)

    for k, v in sorted(class_incidence.items()):
        class_weights[k] = (1 / v) * (total_samples / total_labels)

    print("Class weights: %s" % str(class_weights))

    return class_weights


def load_image_and_labels(filename: str, labels: list[str]) -> (tf.Tensor, tf.Tensor):
    """
    This function reads images from a disk and creates one-hot-encoded labels for every file
    """
    img: tf.Tensor = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(img, channels=3)
    tf.ensure_shape(img, (DATASET_SIZE[0], DATASET_SIZE[1], 3))

    parts = tf.strings.split(filename, os.sep)
    one_hot = tf.cast(parts[-2] == labels, dtype=tf.uint8)

    return img, one_hot


def save_dataset_examples(ds: tf.data.Dataset, work_dir: str, labels: list[str], filename: str):
    """
    This function saves an example of how images will be seen by models while training. It helps identify errors in
    augmentation pipelines.
    """
    fig, axs = plt.subplots(4, 4, figsize=(22, 22))

    for x, y in ds.take(1):
        for ax, image, label in zip(axs.flat, x, y):
            ax.imshow(image.numpy())
            ax.set_title(labels[np.argmax(label.numpy())])
            ax.set_yticks([])

    fig.tight_layout()
    plt.savefig(os.path.join(work_dir, filename))
    plt.clf()


# User: ShairozS
# Published: 24 April 2021
# Title: Applying transforms to a batch of images
# Link: https://github.com/albumentations-team/albumentations/issues/881#issuecomment-825768702
class Transform:
    """
    This class applies augmentations defined in the transform parameter to batches of images, reducing the number of
    calls of tf.numpy_function
    """

    def __init__(self, pipeline: A.BaseCompose, shape: tuple[int, int, int], labels: list[str]):
        self.pipeline = pipeline
        self.shape = shape
        self.labels = labels

    def run_batched(self, images: tf.Tensor) -> np.ndarray:
        return np.array([self.pipeline(image=image)['image'] for image in images], dtype=np.uint8)

    def run_non_batched(self, images: tf.Tensor) -> np.ndarray:
        return self.pipeline(image=images)['image'].astype(np.uint8)

    def __call__(self, images: tf.Tensor, labels: tf.Tensor, *args, **kwargs):
        if images.shape.ndims == 4:
            images = tf.numpy_function(func=self.run_batched, inp=[images], Tout=tf.uint8, stateful=False)
            images = tf.ensure_shape(images, (None, self.shape[0], self.shape[1], self.shape[2]))
            labels = tf.ensure_shape(labels, (None, len(self.labels)))
        else:
            images = tf.numpy_function(func=self.run_non_batched, inp=[images], Tout=tf.uint8, stateful=False)
            images = tf.ensure_shape(images, (self.shape[0], self.shape[1], self.shape[2]))
            labels = tf.ensure_shape(labels, (len(self.labels)), )

        return images, labels


def get_train_dataset(
        options: Options,
        work_dir: str,
        shape: tuple[int, int, int],
        labels: list[str],
) -> (tf.data.Dataset, dict[int, float], int):
    """
    This function creates an instance of tf.data.Dataset with applied augmentations. This instance is ready to be passed
    to model.fit() for training. The function uses parallel processing. Additionally, it saves an example of augmented
    images for debugging/reporting.
    """
    ds_train = tf.data.Dataset.list_files(os.path.join(options.dataset, "train", '*', '*'), shuffle=True)
    ds_len = len(ds_train)  # The number of images can be obtained before calling batch(N)

    # Take a slice of data
    if 0 < options.portion < 1:
        ds_train = ds_train.take(round(ds_len * options.portion))

    class_weights = get_class_weights(ds_train, labels)

    # Load images and their labels from directories
    ds_train = ds_train.map(lambda x: load_image_and_labels(x, labels))

    # The list of augmentations that will be applied to images before caching
    static: list[Union[BasicTransform, BaseCompose]] = []

    if options.shades_of_grey:
        static.append(ShadesOfGrey(norm_p=6, always_apply=True))

    if static:
        ds_train = ds_train.map(
            Transform(A.Compose(static), (DATASET_SIZE[0], DATASET_SIZE[1], 3), labels),
            num_parallel_calls=options.workers,
        )

    # The list of augmentations that will be applied to images after caching
    dynamic: list[Union[BasicTransform, BaseCompose]] = [
        A.RandomCrop(CROP_SIZE, CROP_SIZE),
        A.Resize(shape[0], shape[1], interpolation=INTERPOLATIONS[options.resize_interpolation]),
    ]

    if options.d4 is True:
        dynamic.append(A.D4())

    if options.brightness > 0:
        dynamic.append(A.RandomBrightnessContrast(
            brightness_limit=(-options.brightness, options.brightness),
            contrast_limit=0,
        ))

    if options.contrast > 0:
        dynamic.append(A.RandomBrightnessContrast(
            brightness_limit=0,
            contrast_limit=(-options.contrast, options.contrast),
        ))

    if options.saturation > 0:
        dynamic.append(A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=(-int(255 * options.saturation), int(255 * options.saturation)),
        ))

    if options.thin_plate_spline:
        dynamic.append(A.ThinPlateSpline(interpolation=INTERPOLATIONS[options.resize_interpolation]))

    if options.rotate > 0:
        dynamic.append(A.Affine(rotate=(-options.rotate, options.rotate),
                                interpolation=INTERPOLATIONS[options.resize_interpolation]))

    if options.grid_dropout:
        dynamic.append(A.GridDropout(random_offset=True))

    if options.normalization is True:
        dynamic.append(A.Normalize(always_apply=True))

    # User: Julius Simonelli
    # Published: 26 Jan 2021
    # Title: TensorFlow and Albumentations
    # Link: https://jss367.github.io/tensorflow-and-albumentations.html
    transform = Transform(A.Compose(dynamic), shape, labels)

    # Save examples for debug/history
    save_dataset_examples(ds_train.batch(16).map(transform), work_dir, labels, "train_augmentation")

    # Create the final training dataset
    # In this approach we keep unprocessed images in memory and then apply transformations
    ds_train = (ds_train
                .cache()
                .shuffle(ds_len, reshuffle_each_iteration=True)
                .batch(options.batch, num_parallel_calls=options.workers)
                .map(transform, num_parallel_calls=options.workers)
                .prefetch(options.workers))

    return ds_train, class_weights, ds_len


def get_val_dataset(
        options: Options,
        run_dir: str,
        shape: tuple[int, int, int],
        labels: list[str],
) -> [tf.data.Dataset, int]:
    """
    This function creates an instance of tf.data.Dataset. This instance is ready to be passed to model.fit() for
    validation. The function uses parallel processing. Additionally, it saves an example of images for
    debugging/reporting.
    """
    ds_val = tf.data.Dataset.list_files(os.path.join(options.dataset, "val", '*', '*'), shuffle=True)
    ds_len = len(ds_val)  # The number of images can be known before calling batch(N)

    # Take a slice of data
    if 0 < options.portion < 1:
        ds_val = ds_val.take(round(ds_len * options.portion))

    # Load images and their labels from directories
    ds_val = ds_val.map(lambda x: load_image_and_labels(x, labels))

    augmentations: list[Union[BasicTransform, BaseCompose]] = [
        A.CenterCrop(CROP_SIZE, CROP_SIZE),
        A.Resize(shape[0], shape[1], interpolation=INTERPOLATIONS[options.resize_interpolation]),  # cv2.INTER_CUBIC
    ]

    if options.shades_of_grey:
        augmentations.append(ShadesOfGrey(norm_p=6, always_apply=True))

    # Apply simple dimensional transformations
    transform = Transform(A.Compose(augmentations), shape, labels)

    # Save examples for debug/history
    save_dataset_examples(ds_val.batch(16).map(transform), run_dir, labels, "val_no_augmentation")

    # Create the final validation dataset
    # In this approach we apply simple transformations first and the keep processed images in memory
    ds_val = (ds_val
              .batch(options.batch, num_parallel_calls=options.workers)
              .map(transform, num_parallel_calls=options.workers)
              .cache()
              .prefetch(options.workers))

    return ds_val, ds_len


def get_test_dataset(
        options: Options,
        shape: tuple[int, int, int],
        labels: list[str],
) -> [tf.data.Dataset, tf.data.Dataset]:
    """
    This function creates an instance of tf.data.Dataset. This instance is ready to be passed to model.evaluate() for
    testing. Unlike get_train_dataset and get_val_dataset it does not save examples for debugging/reporting, and does
    not use parallel processing/shuffling.
    """
    ds_test = tf.data.Dataset.list_files(os.path.join(options.dataset, "val", '*', '*'), shuffle=False)
    ds_files = ds_test.batch(options.batch)

    # Load images and their labels from directories
    ds_test = ds_test.map(lambda x: load_image_and_labels(x, labels))

    augmentations: list[Union[BasicTransform, BaseCompose]] = [
        A.CenterCrop(CROP_SIZE, CROP_SIZE),
        A.Resize(shape[0], shape[1], interpolation=INTERPOLATIONS[options.resize_interpolation]),
    ]

    if options.shades_of_grey:
        augmentations.append(ShadesOfGrey(norm_p=6, always_apply=True))

    if options.thin_plate_spline:
        augmentations.append(A.ThinPlateSpline(interpolation=INTERPOLATIONS[options.resize_interpolation]))

    # Apply simple transformations
    transform = Transform(A.Compose(augmentations), shape, labels)

    # Create the final test dataset
    ds_test = (ds_test
               .batch(options.batch)
               .map(transform)
               .prefetch(1))

    return ds_files, ds_test
