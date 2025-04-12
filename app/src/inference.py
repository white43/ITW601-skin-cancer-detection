import math
import time
from argparse import Namespace
from queue import Queue
from threading import Thread

import numpy as np
from PIL import Image

from .events import Events


class ClassificationWorker(Thread):
    def __init__(self,
                 options: Namespace,
                 events: Events,
                 tasks: Queue[Image.Image],
                 results: Queue[tuple[int, float]],
                 ):
        Thread.__init__(self)
        self.events = events
        self.options: Namespace = options
        self.tasks: Queue[Image.Image] = tasks
        self.results: Queue[tuple[int, float]] = results

    def run(self):
        # Moving inference off the main thread
        import onnxruntime as ort

        # Wait for models to be downloaded from S3
        self.events.models_downloaded.wait(1800.0)

        model = ort.InferenceSession(self.options.cls_model)
        self.events.cls_runtime_loaded.set()

        while True:
            if self.tasks.qsize() > 0:
                img = self.tasks.get()

                if not isinstance(img, Image.Image):
                    continue

                img = img.resize((224, 224), resample=Image.NEAREST)
                prediction = model.run(None, {"input_layer": np.asarray(img).astype(np.float32)[np.newaxis]})[0][0]
                label = int(np.argmax(prediction))
                probability = float(prediction[label] * 100)

                self.tasks.task_done()
                self.results.put((label, probability))
            else:
                time.sleep(0.05)

            if self.events.stop_everything.is_set():
                self.events.cls_runtime_stopped.set()
                break


class SegmentationWorker(Thread):
    def __init__(self,
                 options: Namespace,
                 events: Events,
                 tasks: Queue[Image.Image],
                 results: Queue[tuple[Image.Image, tuple[int, int, int, int]]],
                 ):
        Thread.__init__(self)
        self.events = events
        self.options: Namespace = options
        self.tasks: Queue[Image.Image] = tasks
        self.results: Queue[tuple[Image.Image, tuple[int, int, int, int]]] = results

    def run(self):
        # Moving inference off the main thread
        import onnxruntime as ort

        # Wait for models to be downloaded from S3
        self.events.models_downloaded.wait(1800.0)

        model: ort.InferenceSession = ort.InferenceSession(self.options.seg_model)

        self.events.yolo_loaded.set()

        while True:
            if self.tasks.qsize() > 0:
                img = self.tasks.get()

                if not isinstance(img, Image.Image):
                    continue

                rel_lesion_float = self._run_inference(model, img, 0.25)

                # No lesion has been found
                if rel_lesion_float[2] == 0 or rel_lesion_float[3] == 0:
                    self.tasks.task_done()
                    self.results.put((img, (0, 0, 0, 0)))
                    continue

                rel_crop_float = self._find_crop_around(rel_lesion_float, 224 / img.size[0])

                # A lesion has been found but no crop needed
                if rel_crop_float[2] == 0 or rel_crop_float[3] == 0:
                    self.tasks.task_done()
                    self.results.put((img, self._to_absolute_int_values(img.size[0], rel_lesion_float)))
                    continue

                rel_lesion_float = self._find_lesion_coordinates_within_crop(rel_crop_float, rel_lesion_float)
                rel_crop_int = self._to_relative_int_values(img.size[0], rel_crop_float)
                abs_crop_int = self._to_absolute_int_values(img.size[0], rel_crop_float)
                abs_lesion_int = self._to_absolute_int_values(rel_crop_int[2], rel_lesion_float)

                self.tasks.task_done()
                self.results.put((img.crop(abs_crop_int), abs_lesion_int))
            else:
                time.sleep(0.05)

            if self.events.stop_everything.is_set():
                self.events.yolo_stopped.set()
                break

    @staticmethod
    def _run_inference(model, image: Image.Image, threshold: float) -> tuple[float, float, float, float]:
        """
        Runs inference on the image argument against the model argument. Threshold represents the level of confidence.
        The result will be in a relative form [0, 1] with respect to the original image.
        """
        original_size = image.size[0]

        input_size = model.get_inputs()[0].shape[2]
        input_name = model.get_inputs()[0].name
        input_ratio = original_size / input_size

        if input_size != original_size:
            working_copy = image.resize(
                size=(input_size, input_size),
                resample=Image.NEAREST if input_ratio > 1 else Image.BILINEAR,
            )
        else:
            working_copy = image.copy()

        working_copy = np.asarray(working_copy).astype(np.float32)
        working_copy = np.transpose(working_copy, (2, 0, 1))  # RGB -> BRG
        working_copy /= 255

        prediction = model.run(None, {input_name: working_copy[np.newaxis]})[0][0]

        # User: Roman Velichkin
        # Published: Jan 20, 2025
        # Title: Guide - How to interpet onnx predictions from detection model
        # Link: https://github.com/orgs/ultralytics/discussions/18776
        prediction = prediction.T

        boxes = prediction[:, :4]
        class_probs = prediction[:, 4]

        # TODO: Add non-maximum suppression (NMS)
        max_class_prob = np.max(class_probs)

        if max_class_prob < threshold:
            return 0, 0, 0, 0

        [x, y, w, h] = boxes[class_probs == max_class_prob][0] / input_size

        x -= (w / 2)
        y -= (h / 2)

        x = max(x, 0)
        y = max(y, 0)
        w = w if x + w <= 1 else 1 - x
        h = h if y + h <= 1 else 1 - y

        return x, y, w, h

    @staticmethod
    def _find_crop_around(
            lesion: tuple[float, float, float, float],
            factor: float
    ) -> tuple[float, float, float, float]:
        """
        Finds a relative crop [0, 1] around a detected lesion. The lesion will be centered within the crop. The final
        crop size will be a multiple of the factor argument.
        """
        x, y, w, h = lesion

        crop = math.ceil(max(w, h) / factor) * factor

        if crop >= 1:
            return 0, 0, 0, 0

        cx = max(x - (crop - w) / 2, 0)
        cy = max(y - (crop - h) / 2, 0)
        cw = crop if cx + crop <= 1 else 1 - crop
        ch = crop if cy + crop <= 1 else 1 - crop

        return cx, cy, cw, ch

    @staticmethod
    def _find_lesion_coordinates_within_crop(
            crop: tuple[float, float, float, float],
            lesion: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """
        Finds relative coordinates [0, 1] of a lesion within its surrounding crop.
        """
        cx, cy, cw, ch = crop
        x, y, w, h = lesion

        nx = (x - cx) / cw
        ny = (y - cy) / ch
        nw = (x - cx + w) / cw - nx
        nh = (y - cy + h) / ch - ny

        return nx, ny, nw, nh

    @staticmethod
    def _to_relative_int_values(size: int, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        """
        Converts relative float coordinates [0, 1] to relative int coordinates with respect to the size given
        """
        cx, cy, cw, ch = box
        cx *= size
        cy *= size
        cw *= size
        ch *= size

        return round(cx), round(cy), round(cw), round(ch)

    @staticmethod
    def _to_absolute_int_values(size: int, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        """
        Converts relative float coordinates [0, 1] to absolute int coordinates with respect to the size given
        """
        cx, cy, cw, ch = box
        cx *= size
        cy *= size
        cw *= size
        ch *= size

        return round(cx), round(cy), round(cx + cw), round(cy + ch)



