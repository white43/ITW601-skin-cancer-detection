import time
from queue import Queue
from threading import Thread

from PIL import Image, ImageDraw

from .events import Events


class ClassificationWorker(Thread):
    def __init__(self,
                 model: str,
                 events: Events,
                 tasks: Queue[Image.Image],
                 results: Queue[tuple[int, float]],
                 ):
        Thread.__init__(self)
        self.events = events
        self.model: str = model
        self.tasks: Queue[Image.Image] = tasks
        self.results: Queue[tuple[int, float]] = results

    def run(self):
        # Moving inference off the main thread
        import numpy as np
        import onnxruntime as ort

        model = ort.InferenceSession(self.model)
        self.events.cls_runtime_loaded.set()

        while True:
            if self.tasks.qsize() > 0:
                img = self.tasks.get()

                if not isinstance(img, Image.Image):
                    continue

                img = img.resize((224, 224), resample=Image.BICUBIC)
                prediction = model.run(None, {"input_layer": np.asarray(img).astype(np.float32)[np.newaxis]})[0][0]
                label = int(np.argmax(prediction))
                probability = float(prediction[label] * 100)

                self.tasks.task_done()
                self.results.put((label, probability))
            else:
                time.sleep(0.05)


class SegmentationWorker(Thread):
    def __init__(self,
                 model: str,
                 events: Events,
                 tasks: Queue[tuple[Image.Image, bool]],
                 results: Queue[Image.Image],
                 ):
        Thread.__init__(self)
        self.events = events
        self.model: str = model
        self.tasks: Queue[tuple[Image.Image, bool]] = tasks
        self.results: Queue[Image.Image] = results

    def run(self):
        # Moving loading YOLO and other libraries off the main thread
        from ultralytics import YOLO

        model = YOLO(self.model)
        print(model.info())

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

                img = img.resize((224, 224), resample=Image.BICUBIC)
                results = model.predict(
                    img,
                    imgsz=(224, 224),
                    conf=0.5,
                    iou=0.2,
                    save=False,
                    show_labels=False,
                    show_conf=False,
                    show_boxes=False,
                )

                draw = ImageDraw.Draw(img)

                for result in results:
                    for xyxy in result.boxes.xyxy.numpy().tolist():
                        draw.rectangle(xyxy, outline=color, width=2)

                self.tasks.task_done()
                self.results.put(img)
            else:
                time.sleep(0.05)
