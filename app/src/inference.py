import time
from argparse import Namespace
from queue import Queue
from threading import Thread

from PIL import Image, ImageDraw

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
                 tasks: Queue[tuple[Image.Image, bool]],
                 results: Queue[Image.Image],
                 ):
        Thread.__init__(self)
        self.events = events
        self.options: Namespace = options
        self.tasks: Queue[tuple[Image.Image, bool]] = tasks
        self.results: Queue[Image.Image] = results

    def run(self):
        # Moving inference off the main thread
        import numpy as np
        import onnxruntime as ort

        # Wait for models to be downloaded from S3
        self.events.models_downloaded.wait(1800.0)

        model = ort.InferenceSession(self.options.seg_model)

        self.events.yolo_loaded.set()

        while True:
            if self.tasks.qsize() > 0:
                img, malignant = self.tasks.get()

                if not isinstance(img, Image.Image):
                    continue

                if malignant is True:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)

                # TODO: Train another segmentation model to work with 448x448 images
                img = img.resize((640, 640), resample=Image.NEAREST)

                img_for_inf = np.asarray(img).astype(np.float32)
                img_for_inf = np.transpose(img_for_inf, (2, 0, 1))  # RGB -> BRG
                img_for_inf /= 255

                input_name = model.get_inputs()[0].name
                prediction = model.run(None, {input_name: img_for_inf[np.newaxis]})[0][0]

                # User: Roman Velichkin
                # Published: Jan 20, 2025
                # Title: Guide - How to interpet onnx predictions from detection model
                # Link: https://github.com/orgs/ultralytics/discussions/18776
                prediction = prediction.T

                boxes = prediction[:, :4]
                class_probs = prediction[:, 4]

                draw = ImageDraw.Draw(img)

                # TODO: Add non-maximum suppression (NMS)
                for x, y, w, h in boxes[class_probs > 0.5]:
                    # https://github.com/microsoft/onnxruntime-extensions/blob/5c53aaad627d7cf4a8f25efcfde849da586cfe45/tutorials/yolov8_pose_e2e.py#L276
                    x -= (w / 2)
                    y -= (h / 2)

                    draw.rectangle([x, y, x+w, y+w], outline=color, width=2)

                self.tasks.task_done()
                self.results.put(img)
            else:
                time.sleep(0.05)

            if self.events.stop_everything.is_set():
                self.events.yolo_stopped.set()
                break
