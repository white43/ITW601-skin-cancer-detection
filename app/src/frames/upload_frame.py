import queue
import tkinter as tk
from argparse import Namespace
from queue import Queue, Empty
from threading import Thread

import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
from PIL import Image
from tkinterdnd2 import TkinterDnD, DND_FILES

from ..events import Events
from ..inference import ClassificationWorker, SegmentationWorker
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
    -1: "unknown",
    0: "benign",
    1: "malignant",
}


class UploadFrame(ctk.CTkFrame):
    image_label: ctk.CTkButton | None
    analyze_button: ctk.CTkButton | None
    show_more_button: ctk.CTkButton | None
    hint_label: ctk.CTkLabel | None

    events: Events
    cls_tasks: Queue[Image.Image]
    cls_results: Queue[tuple[int, float]]
    seg_tasks: Queue[tuple[Image.Image, bool]]
    seg_results: Queue[Image.Image]

    threads: list[Thread] = []

    original_image: Image.Image | None
    last_lesion_binary_label: int
    last_lesion_label: int

    def __init__(self,
                 master: ctk.CTk,
                 options: Namespace,
                 events: Events,
                 cls_tasks: Queue[Image.Image],
                 cls_results: Queue[tuple[int, float]],
                 seg_tasks: Queue[tuple[Image.Image, bool]],
                 seg_results: Queue[Image.Image],
                 download_meter: Queue[tuple[int, int]],
                 **kwargs,
                 ):
        super().__init__(master, **kwargs)

        self.master: ctk.CTk = master
        self.events = events
        self.cls_tasks = cls_tasks
        self.cls_results = cls_results
        self.seg_tasks = seg_tasks
        self.seg_results = seg_results

        self.dnd_light_img = Image.open(resource_path("dnd-light.png"))
        self.dnd_dark_img = Image.open(resource_path("dnd-dark.png"))

        thread = ClassificationWorker(options, events, cls_tasks, cls_results)
        thread.start()
        self.threads.append(thread)

        thread = SegmentationWorker(options, events, seg_tasks, seg_results)
        thread.start()
        self.threads.append(thread)

        thread = Thread(target=lambda: self._wait_for_models_to_be_downloaded(download_meter))
        thread.start()
        self.threads.append(thread)

        thread = Thread(target=self._wait_for_libraries_to_load)
        thread.start()
        self.threads.append(thread)

    def reset_page(self) -> None:
        for child in self.grid_slaves():
            child.grid_forget()
            child.destroy()

    def draw_page(self) -> None:
        self.image_label = ctk.CTkButton(
            master=self.master,
            text="",
            text_color="#333333",
            height=224,
            width=224,
            border_width=1,
            image=ctk.CTkImage(
                light_image=self.dnd_light_img,
                dark_image=self.dnd_dark_img,
                size=(224, 224),
            ),
            hover=False,
            fg_color="transparent",
        )
        self.image_label.place(x=200, y=70)
        self.image_label.drop_target_register(DND_FILES)
        self.image_label.dnd_bind('<<Drop>>', lambda e: self._update_frame_state_on_dnd(e))

        self.hint_label = ctk.CTkLabel(
            master=self.master,
            text="",
            font=("Raleway", 14),
            height=30,
            width=264,
            corner_radius=0,
        )
        self.hint_label.place(x=190, y=300)

        self.analyze_button = ctk.CTkButton(
            master=self.master,
            text="Analyze",
            font=("Raleway", 14),
            hover=True,
            height=30,
            width=95,
            border_width=2,
            corner_radius=6,
            state=tk.DISABLED,
            command=self._put_new_cls_task_to_queue,
        )
        self.analyze_button.place(x=220, y=340)

        self.show_more_button = ctk.CTkButton(
            master=self.master,
            text="More info",
            font=("Raleway", 14),
            hover=True,
            height=30,
            width=95,
            border_width=2,
            corner_radius=6,
            state=tk.DISABLED,
            command=self._put_new_seg_task_to_queue,
        )
        self.show_more_button.place(x=325, y=340)

        # By using this event we prevent errors in _wait_for_libraries_to_load due to fast ONNX loading
        self.events.ui_loaded.set()

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
                current = meter.get()

                downloaded += current[0]
                total_size = current[1]

                self.hint_label.configure(
                    text="AI is loading: %.1f%%. Please stand by..." % (downloaded * 100 / total_size)
                )

                if downloaded >= total_size:
                    self.events.models_downloaded.set()
            except Empty:
                pass

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

        self.image_label.configure(
            text="",
        )

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

        self.original_image = img

        self.image_label.configure(
            image=ctk.CTkImage(
                light_image=img,
                dark_image=img,
                size=(224, 224),
            ),
        )

        self.analyze_button.configure(state=tk.NORMAL)
        self.show_more_button.configure(state=tk.DISABLED)
        self.hint_label.configure(text="Waiting for analysis")

    def _put_new_cls_task_to_queue(self):
        if self.analyze_button.cget("state") == tk.NORMAL:
            self.analyze_button.configure(state=tk.DISABLED)
            self.show_more_button.configure(state=tk.DISABLED)

            thread = Thread(target=self._draw_cls_inference_result)
            thread.start()
            self.threads.append(thread)

            self.cls_tasks.put(self.original_image)

    def _put_new_seg_task_to_queue(self):
        if self.show_more_button.cget("state") == tk.NORMAL:
            self.analyze_button.configure(state=tk.DISABLED)
            self.show_more_button.configure(state=tk.DISABLED)

            thread = Thread(target=self._draw_seg_inference_result)
            thread.start()
            self.threads.append(thread)

            self.seg_tasks.put((self.original_image, True if self.last_lesion_binary_label == LESION_TYPE_MALIGNANT else False))

    def _draw_cls_inference_result(self):
        label: int = -1
        probability: float = -1.0

        try:
            label, probability = self.cls_results.get(timeout=15)
            self.cls_results.task_done()
        except Empty:
            CTkMessagebox(
                icon="warning",
                title="Error",
                message="Timeout while waiting for results",
                font=("Raleway", 14)
            )

        if label in LESION_TYPE_DICT:
            binary_label = LESION_TYPE_DICT[label]
        else:
            binary_label = LESION_TYPE_UNKNOWN

        self.last_lesion_binary_label = binary_label
        self.last_lesion_label = label
        self.last_lesion_probability = probability

        if binary_label in LESION_TYPE_TEXT_DICT:
            lesion_type_text = LESION_TYPE_TEXT_DICT[binary_label]
        else:
            lesion_type_text = LESION_TYPE_TEXT_DICT[LESION_TYPE_UNKNOWN]

        self.hint_label.configure(text="This lesion is %s" % lesion_type_text)

        if self.analyze_button.cget("state") == tk.DISABLED:
            self.analyze_button.configure(state=tk.NORMAL)

        if self.show_more_button.cget("state") == tk.DISABLED:
            self.show_more_button.configure(state=tk.NORMAL)

        self.thread_gc()

    def _draw_seg_inference_result(self):
        result: Image.Image | None = None

        try:
            result = self.seg_results.get(timeout=15)
            self.seg_results.task_done()
        except Empty:
            CTkMessagebox(
                icon="warning",
                title="Error",
                message="Timeout while waiting for results",
                font=("Raleway", 14)
            )

        self.image_label.configure(
            image=ctk.CTkImage(
                light_image=result,
                dark_image=result,
                size=(224, 224),
            ),
        )

        if self.analyze_button.cget("state") == tk.DISABLED:
            self.analyze_button.configure(state=tk.NORMAL)

        if self.show_more_button.cget("state") == tk.DISABLED:
            self.show_more_button.configure(state=tk.NORMAL)

        self.hint_label.configure(text="This is %s (%.0f%%)" % (LESION_CLASSES[self.last_lesion_label], self.last_lesion_probability))

        self.thread_gc()

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



