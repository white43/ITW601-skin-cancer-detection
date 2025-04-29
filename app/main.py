import argparse
from queue import Queue
from threading import Thread
from typing import Optional

import customtkinter as ctk
import numpy as np
from PIL import Image

from app.src.app import App
from app.src.events import Events
from app.src.frames.barber_frame import BarberFrame
from app.src.frames.upload_frame import UploadFrame
from app.src.options import Options
from app.src.overrides import Tk
from app.src.update import download_models
from app.src.utils import resource_path

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("dark-blue")

cli_opts = argparse.ArgumentParser()
cli_opts.add_argument("--cls-model")
cli_opts.add_argument("--seg-model")
cli_opts.add_argument("--download-from")
cli_opts.add_argument("--debug", action='store_true')
cli_opts.add_argument("--debug-asymmetry", action='store_true')
cli_opts.add_argument("--debug-diameter", action='store_true')

options = Options()
cli_opts.parse_known_args(namespace=options)

window = Tk()
window.geometry("640x640")
window.title("AI for Skin Cancer Detection")

ctk.FontManager.load_font(resource_path("fonts", "Raleway-Regular.ttf"))

seg_tasks: Queue[Image.Image] = Queue()
# A queue for results of inference from the segmentation model: a square crop and lesion boundaries within the crop
seg_results: Queue[tuple[Image.Image, tuple[int, int, int, int], Optional[np.ndarray]]] = Queue()
# A queue for displaying current progress to users while models are being downloading
download_meter: Queue[tuple[int, int]] = Queue()
events = Events()

# A separate thread to download models and avoid main thread blocking
downloading = Thread(target=lambda: download_models(options, events, download_meter), name="download_models")
downloading.start()

upload_frame = UploadFrame(window, options, events, seg_tasks, seg_results, download_meter)
barber_page = BarberFrame(window)
app = App(upload_frame)
barber_page.tkraise()


def graceful_shutdown():
    events.stop_everything.set()

    if events.cls_runtime_stopped.wait(15) and events.yolo_stopped.wait(15):
        window.destroy()


window.protocol("WM_DELETE_WINDOW", lambda: graceful_shutdown())
window.mainloop()
