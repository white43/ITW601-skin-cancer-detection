import os
import tkinter as tk
from argparse import Namespace
from queue import Queue, Empty
from threading import Thread

import customtkinter as ctk
from PIL import Image
from ..events import Events
from ..inference import Worker
from tkinterdnd2 import TkinterDnD, DND_FILES

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


class UploadFrame(ctk.CTkFrame):
    image_label: ctk.CTkButton | None
    analyze_button: ctk.CTkButton | None
    hint_label: ctk.CTkLabel | None

    events: Events
    tasks: Queue[tuple[Image.Image, ctk.CTkButton, ctk.CTkLabel]]
    results: Queue[int]

    threads: list[Thread] = []

    def __init__(self,
                 master: ctk.CTk,
                 options: Namespace,
                 events: Events,
                 tasks: Queue[tuple[Image.Image, ctk.CTkButton, ctk.CTkLabel]],
                 results: Queue[int],
                 **kwargs,
                 ):
        super().__init__(master, **kwargs)

        self.master: ctk.CTk = master
        self.events = events
        self.tasks = tasks
        self.results = results

        self.dnd_img = Image.open(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "dnd.png"))

        thread = Worker(options.model, events, tasks, results)
        thread.start()
        self.threads.append(thread)

        thread = Thread(target=self._wait_for_tensorflow_to_load)
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
            # font=("Arial", 14), # TODO Railway
            text_color="#333333",
            height=224,
            width=224,
            border_width=1,
            image=ctk.CTkImage(
                light_image=self.dnd_img,
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
            text="Tensorflow is loading. Please stand by...",
            # font=("Arial", 14), # TODO Railway
            height=30,
            width=224,
            corner_radius=0,
        )
        self.hint_label.place(x=200, y=300)

        self.analyze_button = ctk.CTkButton(
            master=self.master,
            text="Analyze",
            # font=("undefined", 14), # TODO Railway
            # text_color="#000000",
            hover=True,
            height=30,
            width=95,
            border_width=2,
            corner_radius=6,
            state=tk.DISABLED,
            command=self._put_new_task_to_queue_on_click,
        )
        self.analyze_button.place(x=265, y=340)

    def _wait_for_tensorflow_to_load(self):
        if self.events.tensorflow_loaded.wait(60.0):
            self.hint_label.configure(text="Waiting for an image...")

    def _update_frame_state_on_dnd(self, e: TkinterDnD.DnDEvent) -> None:
        filepath = e.data

        if not filepath.endswith(".jpg") and not filepath.endswith(".jpeg"):
            raise Exception("Not a JPG")

        try:
            img = Image.open(filepath)
        except Exception:
            raise Exception("Couldn't read an image")

        self.image_label.configure(
            text="",
        )

        self.image_label.configure(
            image=ctk.CTkImage(
                light_image=img,
                dark_image=img,
                size=(224, 224),
            ),
        )

        self.analyze_button.configure(state=tk.NORMAL)
        self.hint_label.configure(text="Waiting for analysis")

    def _put_new_task_to_queue_on_click(self):
        if self.analyze_button.cget("state") == tk.NORMAL:
            self.analyze_button.configure(state=tk.DISABLED)

            image: ctk.CTkImage = self.image_label.cget("image")
            image: Image.Image = image.cget("light_image")

            thread = Thread(target=self._draw_inference_result_on_ready)
            thread.start()
            self.threads.append(thread)

            self.tasks.put((image, self.analyze_button, self.hint_label))

    def _draw_inference_result_on_ready(self):
        result: int = -1

        try:
            result = self.results.get(timeout=15)
            self.results.task_done()
        except Empty:
            pass

        self.hint_label.configure(text="In this image: %s" % LESION_CLASSES[result])

        if self.analyze_button.cget("state") == tk.DISABLED:
            self.analyze_button.configure(state=tk.NORMAL)

        # Keep only alive threads in the list
        self.threads[:] = [thread for thread in self.threads if thread.is_alive()]

    def redraw_page(self) -> None:
        self.reset_page()
        self.draw_page()

    def destroy(self):
        for t in self.threads:
            t.join()

        super().destroy()



