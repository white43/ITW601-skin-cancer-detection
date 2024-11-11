import time
import tkinter
from queue import Queue
from threading import Thread

import customtkinter as ctk
from PIL import Image
from .events import Events


class Worker(Thread):
    def __init__(self,
                 model: str,
                 events: Events,
                 tasks: Queue[tuple[Image.Image, ctk.CTkButton, ctk.CTkLabel]],
                 results: Queue,
                 ):
        Thread.__init__(self)
        self.events = events
        self.model: str = model
        self.tasks: Queue[tuple[Image.Image, ctk.CTkButton, ctk.CTkLabel]] = tasks
        self.results: Queue = results

    def run(self):
        # Moving loading TF and other libraries off the main thread
        import keras as k
        import numpy as np

        model: k.models.Model = k.models.load_model(self.model, compile=True)
        self.events.tensorflow_loaded.set()

        while True:
            if self.tasks.qsize() > 0:
                img, button, hint = self.tasks.get()

                if not isinstance(img, Image.Image):
                    continue

                img = img.resize((224, 224), resample=Image.BICUBIC)
                prediction = np.argmax(model.predict(np.asarray(img)[np.newaxis], verbose=0), axis=1)[0]

                self.tasks.task_done()
                self.results.put(prediction)

                button.configure(state=tkinter.NORMAL)
            else:
                time.sleep(0.05)

        print("Inference thread finished")
