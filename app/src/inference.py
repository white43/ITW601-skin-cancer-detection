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
        import numpy as np
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
                 results: Queue[tuple[Image.Image, int, int, int, int]],
                 ):
        Thread.__init__(self)
        self.events = events
        self.options: Namespace = options
        self.tasks: Queue[Image.Image] = tasks
        self.results: Queue[tuple[Image.Image, int, int, int, int]] = results

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

                img, x, y, w, h = self._process_image(model, img)

                self.tasks.task_done()
                self.results.put((img, x, y, x + w, y + h))
            else:
                time.sleep(0.05)

            if self.events.stop_everything.is_set():
                self.events.yolo_stopped.set()
                break

    @staticmethod
    def _process_image(model, origin_img: Image.Image) -> tuple[Image.Image, int, int, int, int]:
        input_size = model.get_inputs()[0].shape[2]
        input_name = model.get_inputs()[0].name

        may_be_crop: bool = origin_img.size[0] > input_size

        while True:
            origin_size: int = origin_img.size[0]
            origin_ratio: float = origin_size / input_size

            img = origin_img.resize((input_size, input_size), resample=Image.NEAREST)

            img_for_inf = np.asarray(img).astype(np.float32)
            img_for_inf = np.transpose(img_for_inf, (2, 0, 1))  # RGB -> BRG
            img_for_inf /= 255

            prediction = model.run(None, {input_name: img_for_inf[np.newaxis]})[0][0]

            # User: Roman Velichkin
            # Published: Jan 20, 2025
            # Title: Guide - How to interpet onnx predictions from detection model
            # Link: https://github.com/orgs/ultralytics/discussions/18776
            prediction = prediction.T

            boxes = prediction[:, :4]
            class_probs = prediction[:, 4]

            # TODO: Add non-maximum suppression (NMS)
            max_class_prob = np.max(class_probs)

            x: float = 0
            y: float = 0
            w: float = 0
            h: float = 0

            if max_class_prob > 0.25:
                [x, y, w, h] = boxes[class_probs == max_class_prob][0]

                if may_be_crop:
                    cx = x * origin_ratio
                    cy = y * origin_ratio
                    cw = w * origin_ratio
                    ch = h * origin_ratio

                    max_lesion_side = max(cw, ch)
                    crop_size = math.ceil(max_lesion_side / input_size) * input_size

                    if crop_size < origin_size:
                        # https://github.com/microsoft/onnxruntime-extensions/blob/5c53aaad627d7cf4a8f25efcfde849da586cfe45/tutorials/yolov8_pose_e2e.py#L276
                        cx -= (crop_size / 2)
                        cy -= (crop_size / 2)
                        cw = crop_size
                        ch = crop_size

                        origin_img = origin_img.crop((cx, cy, cx + cw, cy + ch))
                    else:
                        may_be_crop = False

                if not may_be_crop:
                    x -= (w / 2)
                    y -= (h / 2)

            if not may_be_crop:
                break

            if may_be_crop:
                may_be_crop = False

        if w != 0:
            x = max(x, 1)
            y = max(y, 1)
            w = min(w, origin_size - 1)
            h = min(h, origin_size - 1)

        return img, round(x), round(y), round(w), round(h)
