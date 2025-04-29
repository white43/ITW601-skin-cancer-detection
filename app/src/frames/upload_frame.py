import math
import queue
import time
import tkinter as tk
from copy import deepcopy
from io import BytesIO
from queue import Queue, Empty
from threading import Thread
from typing import Optional

import cairosvg
import customtkinter as ctk
import cv2
import numpy as np
from CTkMessagebox import CTkMessagebox
from PIL import Image, ImageDraw
from tkinterdnd2 import TkinterDnD, DND_FILES

from training.classification.constants import CROP_SIZE
from .barber_frame import BarberFrame
from ..events import Events
from ..inference import ClassificationWorker, SegmentationWorker
from ..options import Options
from ..polygon import centroid, circularity
from ..utils import resource_path

LESION_TYPE_UNKNOWN = -1
LESION_TYPE_BENIGN = 0
LESION_TYPE_MALIGNANT = 1

LESION_CLASSES = {
    -1: "Unknown",
    0: "Melanoma (MEL)",
    1: "Melanocytic Nevus (NV)",
    2: "Basal Cell Carcinoma (BCC)",
    3: "Actinic Keratosis (AKIEC)",
    4: "Benign Keratosis (BKL)",
    5: "Dermatofibroma (DF)",
    6: "Vascular Lesions (VASC)",
}

LESION_TYPE_DICT = {
    -1: LESION_TYPE_UNKNOWN,
    0: LESION_TYPE_MALIGNANT,
    1: LESION_TYPE_BENIGN,
    2: LESION_TYPE_MALIGNANT,
    3: LESION_TYPE_MALIGNANT,
    4: LESION_TYPE_BENIGN,
    5: LESION_TYPE_BENIGN,
    6: LESION_TYPE_BENIGN,
}

LESION_TYPE_TEXT_DICT = {
    -1: "Unknown",
    0: "Benign",
    1: "Malignant",
}


class UploadFrame(ctk.CTkFrame):
    def __init__(self,
                 master: ctk.CTk,
                 options: Options,
                 events: Events,
                 seg_tasks: Queue[Image.Image],
                 seg_results: Queue[tuple[Image.Image, tuple[int, int, int, int], Optional[np.ndarray]]],
                 download_meter: Queue[tuple[int, int]],
                 **kwargs,
                 ):
        super().__init__(master, **kwargs)

        self.master: ctk.CTk = master
        self.options: Options = options
        self.events: Events = events
        self.cls_task_queues: dict[str, Queue[Image.Image]] = {}
        self.cls_result_queues: dict[str, Queue[list[float, ...]]] = {}
        self.seg_tasks: Queue[Image.Image] = seg_tasks
        self.seg_results: Queue[tuple[Image.Image, tuple[int, int, int, int], Optional[np.ndarray]]] = seg_results

        self.original_image: Optional[Image.Image] = None
        self.segmented_image: Optional[Image.Image] = None
        self.lesion_bbox: Optional[tuple[int, int, int, int]] = None
        self.lesion_mask: Optional[np.ndarray] = None
        self.polygon_vertices: list[tuple[int, int]] = []

        self.dnd_light_img = Image.open(resource_path("dnd-light.png"))
        self.dnd_dark_img = Image.open(resource_path("dnd-dark.png"))

        # User: Alberto Vassena
        # Published: 23 Jul 2017
        # Title: PIL and vectorbased graphics
        # Link: https://stackoverflow.com/a/45262575
        out = BytesIO()
        cairosvg.svg2png(url=resource_path("razor-normal.svg"), write_to=out)
        self.razor_normal = Image.open(out)
        out = BytesIO()
        cairosvg.svg2png(url=resource_path("razor-disabled.svg"), write_to=out)
        self.razor_disabled = Image.open(out)

        out = BytesIO()
        cairosvg.svg2png(url=resource_path("fit-polygon-normal.svg"), write_to=out)
        self.fit_img_normal = Image.open(out)
        out = BytesIO()
        cairosvg.svg2png(url=resource_path("fit-polygon-disabled.svg"), write_to=out)
        self.fit_img_disabled = Image.open(out)

        out = BytesIO()
        cairosvg.svg2png(url=resource_path("clear-polygon-normal.svg"), write_to=out)
        self.polygon_img_normal = Image.open(out)
        out = BytesIO()
        cairosvg.svg2png(url=resource_path("clear-polygon-disabled.svg"), write_to=out)
        self.polygon_img_disabled = Image.open(out)

        self.razor_button: Optional[ctk.CTkButton] = None
        self.image_label: Optional[ctk.CTkButton] = None
        self.find_lesion_button: Optional[ctk.CTkButton] = None
        self.predict_class_button: Optional[ctk.CTkButton] = None
        self.hint_label: Optional[ctk.CTkLabel] = None
        self.fit_polygon_button: Optional[ctk.CTkButton] = None
        self.clear_polygon_label: Optional[ctk.CTkLabel] = None
        self.circularity_label: Optional[ctk.CTkLabel] = None
        self.circularity_label_value: Optional[ctk.CTkLabel] = None

        # Diameter-related images, controls, and variables
        self.diameter_measure_mode: bool = False
        self.clicks: list[tuple[int, int]] = []
        self.clicks_needed: int = 3

        out = BytesIO()
        cairosvg.svg2png(url=resource_path("scale-normal.svg"), write_to=out)
        self.diameter_img_normal = Image.open(out)
        out = BytesIO()
        cairosvg.svg2png(url=resource_path("scale-disabled.svg"), write_to=out)
        self.diameter_img_disabled = Image.open(out)

        self.diameter_button: Optional[ctk.CTkLabel] = None
        self.diameter_label: Optional[ctk.CTkLabel] = None
        self.diameter_label_value: Optional[ctk.CTkLabel] = None

        self.threads: list[Thread] = []

        thread = SegmentationWorker(options, events, seg_tasks, seg_results)
        thread.start()
        self.threads.append(thread)

        thread = Thread(target=lambda: self._wait_for_models_to_be_downloaded(download_meter))
        thread.start()
        self.threads.append(thread)

        thread = Thread(target=self._wait_for_libraries_to_load)
        thread.start()
        self.threads.append(thread)

        thread = Thread(target=self._spawn_cls_workers)
        thread.start()
        self.threads.append(thread)

    def reset_page(self) -> None:
        for child in self.grid_slaves():
            child.grid_forget()
            child.destroy()

    def draw_page(self) -> None:
        self.razor_button = ctk.CTkButton(
            master=self.master,
            text="",
            height=30,
            width=30,
            border_width=1,
            image=ctk.CTkImage(
                light_image=self.razor_disabled,
                dark_image=self.razor_disabled,
                size=(30, 30),
            ),
            hover=False,
            state=ctk.DISABLED,
            fg_color="transparent",
            command=self._raise_razor_frame,
        )
        self.razor_button.place(x=31, y=100)

        self.image_label = ctk.CTkButton(
            master=self.master,
            text="",
            text_color="#333333",
            height=448,
            width=448,
            border_width=1,
            image=ctk.CTkImage(
                light_image=self.dnd_light_img,
                dark_image=self.dnd_dark_img,
                size=(448, 448),
            ),
            hover=False,
            fg_color="transparent",
        )
        self.image_label.place(x=91, y=50)
        self.image_label.drop_target_register(DND_FILES)
        self.image_label.dnd_bind('<<Drop>>', lambda e: self._update_frame_state_on_dnd(e))
        self.image_label.bind('<Button 1>', self._handle_click_on_image)

        self.hint_label = ctk.CTkLabel(
            master=self.master,
            text="",
            font=("Raleway", 14),
            height=60,
            width=300,
            corner_radius=0,
        )
        self.hint_label.place(x=172, y=510)

        self.find_lesion_button = ctk.CTkButton(
            master=self.master,
            text="Find lesion",
            font=("Raleway", 14),
            hover=True,
            height=30,
            width=95,
            border_width=2,
            corner_radius=6,
            state=tk.DISABLED,
            command=self._put_new_seg_task_to_queue,
        )
        self.find_lesion_button.place(x=220, y=575)

        self.predict_class_button = ctk.CTkButton(
            master=self.master,
            text="Predict class",
            font=("Raleway", 14),
            hover=True,
            height=30,
            width=95,
            border_width=2,
            corner_radius=6,
            state=tk.DISABLED,
            command=self._put_new_cls_task_to_queue,
        )
        self.predict_class_button.place(x=325, y=575)

        self.fit_polygon_button = ctk.CTkButton(
            master=self.master,
            text="",
            height=30,
            width=30,
            border_width=1,
            image=ctk.CTkImage(
                light_image=self.fit_img_disabled,
                dark_image=self.fit_img_disabled,
                size=(30, 30),
            ),
            hover=False,
            state=ctk.NORMAL,
            fg_color="transparent",
            command=self._fit_polygon,
        )
        self.fit_polygon_button.place(x=568, y=50)

        self.clear_polygon_label = ctk.CTkButton(
            master=self.master,
            text="",
            height=30,
            width=30,
            border_width=1,
            image=ctk.CTkImage(
                light_image=self.polygon_img_disabled,
                dark_image=self.polygon_img_disabled,
                size=(30, 30),
            ),
            hover=False,
            state=ctk.DISABLED,
            fg_color="transparent",
            command=self._clear_polygon,
        )
        self.clear_polygon_label.place(x=568, y=90)

        self.circularity_label = ctk.CTkLabel(
            master=self.master,
            text="Circularity:",
            height=20,
            width=70,
            font=("Raleway", 14)
        )
        self.circularity_label_value = ctk.CTkLabel(
            master=self.master,
            text="0.00",
            height=15,
            width=70,
            font=("Raleway", 14),
        )
        self.circularity_label.place(x=557, y=140)
        self.circularity_label_value.place(x=557, y=160)

        self.diameter_button = ctk.CTkButton(
            master=self.master,
            text="",
            height=30,
            width=30,
            border_width=1,
            image=ctk.CTkImage(
                light_image=self.diameter_img_disabled,
                dark_image=self.diameter_img_disabled,
                size=(30, 30),
            ),
            hover=False,
            state=ctk.NORMAL,
            fg_color="transparent",
            command=self._handle_diameter_button_click,
        )
        self.diameter_button.place(x=568, y=190)

        self.diameter_label = ctk.CTkLabel(
            master=self.master,
            text="Diameter:",
            height=20,
            width=70,
            font=("Raleway", 14)
        )
        self.diameter_label_value = ctk.CTkLabel(
            master=self.master,
            text="0.0 mm",
            height=15,
            width=70,
            font=("Raleway", 14),
        )
        self.diameter_label.place(x=557, y=230)
        self.diameter_label_value.place(x=557, y=250)

        # By using this event we prevent errors in _wait_for_libraries_to_load due to fast ONNX loading
        self.events.ui_loaded.set()

    def _reset_all_buttons(self):
        self.find_lesion_button.configure(state=tk.NORMAL)
        self.predict_class_button.configure(state=tk.DISABLED)
        self._disable_fit_polygon_button()
        self._disable_clear_polygon_button()
        self._disable_diameter_button()
        self._disable_razor_button()

    # User: Taku
    # Published: 27 Feb 2017
    # Title: Tkinter get mouse coordinates on click and use them as variables
    # Link: https://stackoverflow.com/a/42494066
    def _handle_click_on_image(self, eventorigin):
        x0 = eventorigin.x
        y0 = eventorigin.y

        x0 = math.ceil(x0 / CROP_SIZE * self.segmented_image.size[0])
        y0 = math.ceil(y0 / CROP_SIZE * self.segmented_image.size[1])

        if self.diameter_measure_mode:
            self.clicks.append((x0, y0))
            self.diameter_label_value.configure(text='%d/%d' % (len(self.clicks), self.clicks_needed))

            if len(self.clicks) == self.clicks_needed:
                diameter = self._measure_lesion_diameter()
                self._deactivate_diameter_mode(diameter)
        else:
            self.polygon_vertices.append((x0, y0))
            self._display_polygon(LESION_TYPE_UNKNOWN)
            self._enable_fit_polygon_button()
            self._enable_clear_polygon_button()

    def _display_polygon(self, malignant: int) -> None:
        rgba1 = self.segmented_image.copy().convert('RGBA')
        rgba2 = Image.new('RGBA', self.segmented_image.size)
        canvas = ImageDraw.Draw(rgba2)

        if malignant == LESION_TYPE_MALIGNANT:
            fill_color = (255, 0, 0, 64)
            circle_color = (255, 0, 0, 255)
            centroid_color = (255, 0, 0, 255)
        elif malignant == LESION_TYPE_BENIGN:
            fill_color = (0, 255, 0, 64)
            circle_color = (0, 255, 0, 255)
            centroid_color = (0, 255, 0, 255)
        else:
            fill_color = (255, 214, 35, 64)
            circle_color = (255, 214, 35)
            centroid_color = (255, 214, 35, 255)

        if len(self.polygon_vertices) >= 3:
            canvas.polygon(self.polygon_vertices, fill=fill_color)

            xc, yc = centroid(self.polygon_vertices)

            if xc >= 5 and yc >= 5:
                width = round(self.segmented_image.size[0] * 0.01)
                canvas.line((xc, yc - width*3, xc, yc + width*3), fill=centroid_color, width=width)
                canvas.line((xc - width*3, yc, xc + width*3, yc), fill=centroid_color, width=width)

        for i, (x, y) in enumerate(self.polygon_vertices):
            radius = self.segmented_image.size[0] * 0.01
            width = round(self.segmented_image.size[0] * 0.005)

            canvas.circle(
                xy=(x, y),
                radius=radius,
                fill=(33, 33, 33, 255),
                outline=circle_color,
                width=width if width > 0 else 1,
            )

        img = Image.alpha_composite(rgba1, rgba2)

        self.image_label.configure(
            image=ctk.CTkImage(
                light_image=img,
                dark_image=img,
                size=(448, 448),
            ),
        )

        self._circularity_label(circularity(self.polygon_vertices))

    def _circularity_label(self, c: float):
        self.circularity_label_value.configure(text="%0.2f" % c)

    def _display_rectangle(self, malignant: int) -> None:
        img = self.segmented_image.copy()

        if malignant == LESION_TYPE_MALIGNANT:
            color = (255, 0, 0)
        elif malignant == LESION_TYPE_BENIGN:
            color = (0, 255, 0)
        else:
            color = (0, 0, 0)

        ImageDraw.Draw(img).rectangle(
            xy=self.lesion_bbox,
            outline=color,
            width=round(img.size[0] * 0.005),
        )

        self.image_label.configure(
            image=ctk.CTkImage(
                light_image=img,
                dark_image=img,
                size=(448, 448),
            ),
        )

    def _clear_polygon(self):
        self.polygon_vertices = []
        self.lesion_mask = None
        self._display_polygon(LESION_TYPE_UNKNOWN)
        self._disable_clear_polygon_button()

    def _enable_clear_polygon_button(self):
        self.clear_polygon_label.configure(
            state=ctk.NORMAL,
            hover=True,
            image=ctk.CTkImage(
                light_image=self.polygon_img_normal,
                dark_image=self.polygon_img_normal,
                size=(30, 30),
            ),
        )

    def _disable_clear_polygon_button(self):
        self.clear_polygon_label.configure(
            state=ctk.DISABLED,
            hover=False,
            image=ctk.CTkImage(
                light_image=self.polygon_img_disabled,
                dark_image=self.polygon_img_disabled,
                size=(30, 30),
            ),
        )

        self.circularity_label_value.configure(text='0.00')

    # --------------------- #
    # RAZOR-RELATED METHODS #
    # --------------------- #

    def _enable_razor_button(self):
        self.razor_button.configure(
            state=ctk.NORMAL,
            hover=True,
            image=ctk.CTkImage(
                light_image=self.razor_normal,
                dark_image=self.razor_normal,
                size=(30, 30),
            ),
        )

    def _disable_razor_button(self):
        self.razor_button.configure(
            state=ctk.DISABLED,
            hover=False,
            image=ctk.CTkImage(
                light_image=self.razor_disabled,
                dark_image=self.razor_disabled,
                size=(30, 30),
            ),
        )

    def _raise_razor_frame(self):
        frame = BarberFrame(self)
        frame.set_filepath(self.original_filepath)
        frame.set_sync_method(self.update_frame_state_from_barber_frame)
        frame.tkraise()

    # ----------------------- #
    # FITTING-RELATED METHODS #
    # ----------------------- #

    def _fit_polygon(self):
        """
        A good image for demo is ISIC_0035914
        """
        if not self.lesion_bbox and not self.lesion_mask:
            return

        img = self.segmented_image.copy()
        bbox = self.lesion_bbox
        mask = deepcopy(self.lesion_mask)

        # User: Adrian Rosebrock
        # Title: OpenCV GrabCut: Foreground Segmentation and Extraction
        # Published: 27 July 2020
        # Link: https://pyimagesearch.com/2020/07/27/opencv-grabcut-foreground-segmentation-and-extraction/
        if mask is not None:
            mask[mask > 0] = cv2.GC_PR_FGD
            mask[mask == 0] = cv2.GC_BGD
            fg_model = np.zeros((1, 65), dtype=np.float64)
            bg_model = np.zeros((1, 65), dtype=np.float64)
            cv2.grabCut(np.array(img), mask, None, bg_model, fg_model, 10, cv2.GC_INIT_WITH_MASK)
        # Fit hand-made polygon
        elif self.polygon_vertices:
            mask = np.zeros(img.size[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.array(self.polygon_vertices), cv2.GC_PR_FGD)
            fg_model = np.zeros((1, 65), dtype=np.float64)
            bg_model = np.zeros((1, 65), dtype=np.float64)
            cv2.grabCut(np.array(img), mask, None, bg_model, fg_model, 10, cv2.GC_INIT_WITH_MASK)
        # If we don't have any mask, we can try finding it within lesion's boundary box
        elif bbox[2] > 0:
            mask = np.zeros(img.size[:2], dtype=np.uint8)
            fg_model = np.zeros((1, 65), dtype=np.float64)
            bg_model = np.zeros((1, 65), dtype=np.float64)
            rect = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
            cv2.grabCut(np.array(img), mask, rect, bg_model, fg_model, 10, cv2.GC_INIT_WITH_RECT)

        mask = np.array(mask == cv2.GC_PR_FGD).astype(np.uint8) * 255

        self.polygon_vertices = []
        self.lesion_mask = mask
        contours, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self._display_contours_or_bbox(contours, bbox)

    def _enable_fit_polygon_button(self):
        self.fit_polygon_button.configure(
            state=ctk.NORMAL,
            hover=True,
            image=ctk.CTkImage(
                light_image=self.fit_img_normal,
                dark_image=self.fit_img_normal,
                size=(30, 30),
            ),
        )

    def _disable_fit_polygon_button(self):
        self.fit_polygon_button.configure(
            state=ctk.DISABLED,
            hover=False,
            image=ctk.CTkImage(
                light_image=self.fit_img_disabled,
                dark_image=self.fit_img_disabled,
                size=(30, 30),
            ),
        )

    # ------------------------ #
    # DIAMETER-RELATED METHODS #
    # ------------------------ #

    def _handle_diameter_button_click(self):
        if not self.diameter_measure_mode:
            self._activate_diameter_mode()

    def _measure_lesion_diameter(self) -> float:
        mm1 = math.sqrt((self.clicks[1][0] - self.clicks[0][0]) ** 2 + (self.clicks[1][1] - self.clicks[0][1]) ** 2)
        mm2 = math.sqrt((self.clicks[2][0] - self.clicks[1][0]) ** 2 + (self.clicks[2][1] - self.clicks[1][1]) ** 2)
        mm = (mm1 + mm2) / 2

        candidates = []

        for v1 in self.polygon_vertices:
            for v2 in self.polygon_vertices:
                if v1 != v2:
                    candidates.append((v1[0], v1[1], v2[0], v2[1]))

        # TODO Thanks to numpy, we can significantly speed up calculation of n^2 number of pairs of vertices. This,
        #  however, needs to be replaced by Graham's scan and Shamos's algorithm
        #  https://en.wikipedia.org/wiki/Graham_scan
        #  https://en.wikipedia.org/wiki/Rotating_calipers#Shamos's_algorithm
        candidates = np.array(candidates)
        distances = np.sqrt((candidates[:, 0] - candidates[:, 2]) ** 2 + (candidates[:, 1] - candidates[:, 3]) ** 2)
        diameter = np.max(distances) / mm

        return diameter

    def _enable_diameter_button(self):
        self.diameter_button.configure(
            state=ctk.NORMAL,
            hover=True,
            image=ctk.CTkImage(
                light_image=self.diameter_img_normal,
                dark_image=self.diameter_img_normal,
                size=(30, 30),
            ),
        )

        self.diameter_label.configure(text='Diameter:')
        self.diameter_label_value.configure(text='0 mm')

    def _activate_diameter_mode(self):
        self.diameter_measure_mode = True
        self.diameter_label.configure(text='Points:')
        self.diameter_label_value.configure(text='%d/%d' % (len(self.clicks), self.clicks_needed))

    def _deactivate_diameter_mode(self, diameter: float):
        self.diameter_measure_mode = False
        self.diameter_label.configure(text='Diameter:')
        self.diameter_label_value.configure(text='%.0f mm' % diameter)
        self.clicks = []

    def _disable_diameter_button(self):
        self.diameter_button.configure(
            state=ctk.DISABLED,
            hover=False,
            image=ctk.CTkImage(
                light_image=self.diameter_img_disabled,
                dark_image=self.diameter_img_disabled,
                size=(30, 30),
            ),
        )

        self.diameter_label.configure(text='Diameter:')
        self.diameter_label_value.configure(text='0 mm')

    def _wait_for_libraries_to_load(self):
        if (
            self.events.ui_loaded.wait(60.0)
            and self.events.models_downloaded.wait(1800.0)
            and self.events.cls_runtime_loaded.wait(60.0)
            and self.events.yolo_loaded.wait(60.0)
        ):
            self.hint_label.configure(text="Waiting for an image...")

    def _wait_for_models_to_be_downloaded(self, meter: queue.Queue[tuple[int, int]]):
        downloaded: int = 0

        self.events.ui_loaded.wait(60.0)

        while True:
            if self.events.models_downloaded.is_set():
                break

            try:
                current = meter.get_nowait()

                downloaded += current[0]
                total_size = current[1]

                self.hint_label.configure(
                    text="AI is loading: %.1f%%. Please stand by..." % (downloaded * 100 / total_size)
                )

                if downloaded >= total_size:
                    self.events.models_downloaded.set()
            except Empty:
                time.sleep(0.05)

    def _spawn_cls_workers(self):
        """
        For every model specified in config, a separate worker (thread) and a communication queue will be spawned
        """
        self.events.models_downloaded.wait(1800.0)

        for model in self.options.cls_models:
            # Each classification worker has its own queue for tasks
            self.cls_task_queues[model.name] = Queue()
            self.cls_result_queues[model.name] = Queue()

            thread = ClassificationWorker(
                model=model,
                events=self.events,
                tasks=self.cls_task_queues[model.name],
                results=self.cls_result_queues[model.name]
            )
            thread.start()
            self.threads.append(thread)

    def _update_frame_state_on_dnd(self, e: TkinterDnD.DnDEvent) -> None:
        filepath = str(e.data)

        if filepath.startswith('{') and filepath.endswith('}'):
            filepath = filepath[1:-1]

        ext = filepath.lower()

        if not ext.endswith(".jpg") and not ext.endswith(".jpeg") and not ext.endswith(".png"):
            CTkMessagebox(
                title="Information",
                message="Expected to get a JPG or PNG file",
                font=("Raleway", 14)
            )

            return

        try:
            img = Image.open(filepath)
        except Exception as e:
            CTkMessagebox(
                icon="warning",
                title="Error",
                message="Could not read file: %s" % str(e),
                font=("Raleway", 14)
            )

            return

        # PNG files often contain alpha channel
        if img.mode != "RGB":
            img = img.convert("RGB")

        self.image_label.configure(
            text="",
        )

        self._update_frame_state(img, filepath)


    def update_frame_state_from_barber_frame(self, img: Image, filepath: str) -> None:
        self._update_frame_state(img, filepath)

    def _update_frame_state(self, img: Image, filepath: str):
        orig_width, orig_height = img.size

        # Our classification models expect images of size 224x224, so we would like to have crops of size 448x448,
        # 896x896, and so on. This way, downscaling operation is relatively easy, as it would require scaling by a power
        # of two.
        desired_crop_size = 448

        # Take central crop from an image of arbitrary size with dimensions >448
        if orig_width > desired_crop_size and orig_height > desired_crop_size:
            factor = min(orig_width // desired_crop_size, orig_height // desired_crop_size)
            desired_crop_size *= factor

            left_offset = (orig_width - desired_crop_size) // 2
            upper_offset = (orig_height - desired_crop_size) // 2
            img = img.crop((left_offset, upper_offset, left_offset + desired_crop_size, upper_offset + desired_crop_size))
        # Take central crop from an image of arbitrary size with dimensions <=448
        else:
            if orig_width > orig_height:
                left_offset = (orig_width - orig_height) // 2
                img = img.crop((left_offset, 0, left_offset + orig_height, orig_height))
            elif orig_height > orig_width:
                upper_offset = (orig_height - orig_width) // 2
                img = img.crop((0, upper_offset, orig_width, upper_offset + orig_width))

        self.original_filepath = filepath
        self.original_image = img
        self.segmented_image = None
        self.lesion_bbox = []
        self.polygon_vertices = []

        self.image_label.configure(
            image=ctk.CTkImage(
                light_image=img,
                dark_image=img,
                size=(448, 448),
            ),
        )

        self._reset_all_buttons()

        def _activate_buttons():
            self.find_lesion_button.configure(state=tk.NORMAL)
            self.predict_class_button.configure(state=tk.DISABLED)
            self._enable_razor_button()
            self.hint_label.configure(text="Waiting for analysis")

        if self.events.models_downloaded.is_set():
            _activate_buttons()
        else:
            def _wait_for_the_moment_to_activate_buttons():
                self.events.models_downloaded.wait(1800.0)
                _activate_buttons()

            thread = Thread(target=_wait_for_the_moment_to_activate_buttons)
            thread.start()
            self.threads.append(thread)

    def _put_new_cls_task_to_queue(self):
        if self.find_lesion_button.cget("state") == tk.NORMAL:
            self.find_lesion_button.configure(state=tk.DISABLED)
            self.predict_class_button.configure(state=tk.DISABLED)

            thread = Thread(target=self._draw_cls_inference_result)
            thread.start()
            self.threads.append(thread)

            # Send our image to  every known model through its own queue
            for model in self.options.cls_models:
                if model.name in self.cls_task_queues:
                    self.cls_task_queues[model.name].put(self.segmented_image.copy())

    def _put_new_seg_task_to_queue(self):
        if self.find_lesion_button.cget("state") == tk.NORMAL:
            self.find_lesion_button.configure(state=tk.DISABLED)
            self.predict_class_button.configure(state=tk.DISABLED)

            thread = Thread(target=self._draw_seg_inference_result)
            thread.start()
            self.threads.append(thread)

            self.seg_tasks.put(self.original_image.copy())

    def _draw_cls_inference_result(self):
        probabilities: list[list[float, ...]] = []

        try:
            for result_queue in self.cls_result_queues.values():
                probabilities.append(result_queue.get(timeout=15))
                result_queue.task_done()
        except Empty:
            CTkMessagebox(
                icon="warning",
                title="Error",
                message="Timeout while waiting for results",
                font=("Raleway", 14)
            )

        hint: str = ""
        first_binary_label: Optional[int] = None

        probabilities: np.ndarray = np.array(probabilities)
        probabilities_sum = np.sum(probabilities, axis=0)
        top3_labels = np.argsort(probabilities_sum)[-3:]
        probabilities_sum /= len(self.cls_task_queues)

        for label in reversed(top3_labels):
            probability = probabilities_sum[label] * 100
            stddev = np.std(probabilities[:, label]) * 100

            if label in LESION_TYPE_DICT:
                binary_label = LESION_TYPE_DICT[label]
            else:
                binary_label = LESION_TYPE_UNKNOWN

            if binary_label in LESION_TYPE_TEXT_DICT:
                binary_label_text = LESION_TYPE_TEXT_DICT[binary_label]
            else:
                binary_label_text = LESION_TYPE_TEXT_DICT[LESION_TYPE_UNKNOWN]

            if first_binary_label is None:
                first_binary_label = binary_label

            if hint:
                hint += "\n"

            hint += "%s (%s, %.0f±%.0f%%)" % (binary_label_text, LESION_CLASSES[label], probability, stddev)

        if hint == "":
            hint = "Oops. I am not sure what it is..."

        self.hint_label.configure(text=hint)

        if self.polygon_vertices:
            self._display_polygon(first_binary_label)
        elif self.lesion_bbox[2] > 0:
            self._display_rectangle(first_binary_label)

        if self.find_lesion_button.cget("state") == tk.DISABLED:
            self.find_lesion_button.configure(state=tk.NORMAL)

        if self.predict_class_button.cget("state") == tk.DISABLED:
            self.predict_class_button.configure(state=tk.NORMAL)

        self.thread_gc()

    def _draw_seg_inference_result(self):
        img: Optional[Image.Image] = None
        bbox: Optional[tuple[int, int, int, int]] = None
        mask: Optional[np.ndarray] = None
        contours = []

        try:
            (img, bbox, mask) = self.seg_results.get(timeout=15)
            self.seg_results.task_done()
        except Empty:
            CTkMessagebox(
                icon="warning",
                title="Error",
                message="Timeout while waiting for results",
                font=("Raleway", 14)
            )

        # Save square crop and box coordinates around a lesion
        self.segmented_image = img.copy()
        self.lesion_bbox = bbox
        self.lesion_mask = mask
        self.polygon_vertices = []

        if mask is not None:
            contours, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self._display_contours_or_bbox(contours, bbox)

        if self.find_lesion_button.cget("state") == tk.DISABLED:
            self.find_lesion_button.configure(state=tk.NORMAL)

        if self.predict_class_button.cget("state") == tk.DISABLED:
            self.predict_class_button.configure(state=tk.NORMAL)

        if bbox[2] > 0:
            self.hint_label.configure(text="A lesion is found.")
        else:
            self.hint_label.configure(text="No lesion is found, but you can still try predicting")

        self.thread_gc()

    def _display_contours_or_bbox(self, contours, bbox):
        if len(contours) > 0:
            # Sort objects by the number of vertices in descendant order
            # TODO What to do with other contours? Should we display them all?
            contours = sorted(contours, key=lambda x: x.shape[0], reverse=True)

            for point in contours[0].reshape(-1, 2).tolist():
                self.polygon_vertices.append((point[0], point[1]))

            self._display_polygon(LESION_TYPE_UNKNOWN)
            self._enable_fit_polygon_button()
            self._enable_clear_polygon_button()
            self._enable_diameter_button()
        elif bbox[2] > 0:
            self._display_rectangle(LESION_TYPE_UNKNOWN)
            self._enable_fit_polygon_button()
            self._disable_clear_polygon_button()
            self._disable_diameter_button()

    def redraw_page(self) -> None:
        self.reset_page()
        self.draw_page()

    def thread_gc(self):
        # Keep only alive threads in the list
        self.threads[:] = [thread for thread in self.threads if thread.is_alive()]

    def destroy(self):
        for t in self.threads:
            t.join()

        super().destroy()
