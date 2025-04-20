import math
import os.path
import time
from queue import Queue
from threading import Thread
from typing import Optional

import numpy as np
from PIL import Image

from .events import Events
from .options import Options, ClsModel


class ClassificationWorker(Thread):
    def __init__(self,
                 model: ClsModel,
                 events: Events,
                 tasks: Queue[Image.Image],
                 results: Queue[list[float, ...]],
                 ):
        Thread.__init__(self, name=os.path.basename(model.name))
        self.model: ClsModel = model
        self.events = events
        self.tasks: Queue[Image.Image] = tasks
        self.results: Queue[list[float, ...]] = results

    def run(self):
        # Moving inference off the main thread
        import onnxruntime as ort

        # Wait for models to be downloaded from S3
        self.events.models_downloaded.wait(1800.0)

        model = ort.InferenceSession(self.model.name)
        self.events.cls_runtime_loaded.set()

        while True:
            if self.tasks.qsize() > 0:
                img = self.tasks.get()

                if not isinstance(img, Image.Image):
                    continue

                img = img.resize((224, 224), resample=Image.NEAREST)

                if "shades-of-grey" in self.model.augmentations:
                    from common.shades_of_grey import shades_of_grey

                    sog = self.model.augmentations["shades-of-grey"]

                    img = shades_of_grey(
                        img=np.array(img),
                        norm_p=int(sog["norm_p"]) if "norm_p" in sog else 6,
                    )

                if isinstance(img, Image.Image):
                    img = np.array(img)

                prediction = model.run(None, {"input_layer": img.astype(np.float32)[np.newaxis]})[0][0]

                self.tasks.task_done()
                self.results.put(prediction.tolist())
            else:
                time.sleep(0.05)

            if self.events.stop_everything.is_set():
                self.events.cls_runtime_stopped.set()
                break


class SegmentationWorker(Thread):
    def __init__(self,
                 options: Options,
                 events: Events,
                 tasks: Queue[Image.Image],
                 results: Queue[tuple[Image.Image, tuple[int, int, int, int], Optional[np.ndarray]]],
                 ):
        Thread.__init__(self)
        self.events = events
        self.options: Options = options
        self.tasks: Queue[Image.Image] = tasks
        self.results: Queue[tuple[Image.Image, tuple[int, int, int, int], Optional[np.ndarray]]] = results

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

                rel_lesion_float, mask = self._run_inference(model, img, box_threshold=0.25, mask_threshold=0.5)

                # No lesion has been found
                if rel_lesion_float[2] == 0 or rel_lesion_float[3] == 0:
                    self.tasks.task_done()
                    self.results.put((img, (0, 0, 0, 0), mask))
                    continue

                rel_crop_float = self._find_crop_around(rel_lesion_float, 224 / img.size[0])

                # A lesion has been found but no crop needed
                if rel_crop_float[2] == 0 or rel_crop_float[3] == 0:
                    self.tasks.task_done()
                    self.results.put((img, self._to_absolute_int_values(img.size[0], rel_lesion_float), mask))
                    continue

                rel_lesion_float = self._find_lesion_coordinates_within_crop(rel_crop_float, rel_lesion_float)
                rel_crop_int = self._to_relative_int_values(img.size[0], rel_crop_float)
                abs_crop_int = self._to_absolute_int_values(img.size[0], rel_crop_float)
                abs_lesion_int = self._to_absolute_int_values(rel_crop_int[2], rel_lesion_float)

                if mask is not None and rel_crop_int[2] < mask.shape[0]:
                    mask = mask[abs_crop_int[1]:abs_crop_int[3], abs_crop_int[0]:abs_crop_int[2]]

                self.tasks.task_done()
                self.results.put((img.crop(abs_crop_int), abs_lesion_int, mask))
            else:
                time.sleep(0.05)

            if self.events.stop_everything.is_set():
                self.events.yolo_stopped.set()
                break

    @staticmethod
    def _run_inference(model, image: Image.Image, box_threshold: float, mask_threshold: float) -> tuple[
        tuple[float, float, float, float], Optional[np.ndarray]]:
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

        prediction = model.run(None, {input_name: working_copy[np.newaxis]})

        # Here is the full list of steps need to be taken to process prediction for exported to ONNX models
        # 1. https://github.com/ultralytics/ultralytics/blob/bdfac623ddb1683f2298069f2efab998871dcc58/ultralytics/engine/predictor.py#L325
        #    At this step we have a list with two(?) elements: (1, 300, 38) and (1, 32, 160, 160). Since YOLO models
        #    are being exported with built-in NMS (via nms=True), we need to take only the first row from both matrices.
        #    Columns 0-3 contain XYXY coordinates in a range [0, input_size]. Column 4 contains probability in a range
        #    [0, 1].
        [x0, y0, x1, y1] = prediction[0][0, :, :4][0]
        probability = prediction[0][0, :, 4][0]

        if probability < box_threshold:
            return (0, 0, 0, 0), None

        mask_ratio: float = input_size / 160

        # Convert XYXY to normalized XYHW
        x = max(x0, 0) / input_size
        y = max(y0, 0) / input_size
        w = min(x1 - x0, input_size) / input_size
        h = min(y1 - y0, input_size) / input_size

        if probability < mask_threshold:
            return (x, y, w, h), None

        # 2. https://github.com/ultralytics/ultralytics/blob/bdfac623ddb1683f2298069f2efab998871dcc58/ultralytics/engine/predictor.py#L332
        #    Here the postprocessing starts.
        # 3. https://github.com/ultralytics/ultralytics/blob/bdfac623ddb1683f2298069f2efab998871dcc58/ultralytics/models/yolo/segment/predict.py#L48-L67
        #    At this step we separate the list into to distinct elements: preds and protos (0th and 1st elements)
        # 4. https://github.com/ultralytics/ultralytics/blob/bdfac623ddb1683f2298069f2efab998871dcc58/ultralytics/models/yolo/detect/predict.py#L33-L69
        #    Apply NMS if needed to preds
        # 5. https://github.com/ultralytics/ultralytics/blob/bdfac623ddb1683f2298069f2efab998871dcc58/ultralytics/models/yolo/segment/predict.py#L69-L85
        #    Iterate over predictions
        # 6. https://github.com/ultralytics/ultralytics/blob/bdfac623ddb1683f2298069f2efab998871dcc58/ultralytics/models/yolo/segment/predict.py#L88-L113
        #    Each prediction is a (1, 32) matrix and each proto is (32, 160, 160) matrix. Next, we need to carry out
        #    matrix multiplication and get a new matrix of shape (160, 160).
        pred = prediction[0][0, 0, 6:].reshape(1, -1) # (1, 32)
        proto = prediction[1][0].reshape(32, -1) # (32, 160, 160)
        matrix: np.ndarray = np.matmul(pred, proto).reshape(160, 160) # (32, 25600) -> (160, 160)

        # 7. https://github.com/ultralytics/ultralytics/blob/bdfac623ddb1683f2298069f2efab998871dcc58/ultralytics/utils/ops.py#L661-L677
        #    Here we need to get rid of noise around bounding boxes (hmm, there a lot of noise, actually). This
        #    operation filters out everything outside bounding boxes.
        r = np.arange(matrix.shape[1], dtype=x1.dtype)[None, :]
        c = np.arange(matrix.shape[0], dtype=x1.dtype)[:, None]
        matrix *= ((r >= x0 / mask_ratio) * (r < x1 / mask_ratio) * (c >= y0 / mask_ratio) * (c < y1 / mask_ratio))

        # Resize binary mask to the original image's size.
        i = Image.fromarray(matrix > 0)
        i = i.resize((original_size, original_size), resample=Image.BILINEAR)
        mask: np.ndarray = np.array(i).astype(np.uint8) * 255

        return (x, y, w, h), mask

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
