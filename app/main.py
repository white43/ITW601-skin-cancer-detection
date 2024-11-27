import argparse
import os
from queue import Queue

import customtkinter as ctk
from PIL import Image

from .src.app import App
from .src.events import Events
from .src.frames.upload_frame import UploadFrame
from .src.overrides import Tk

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("dark-blue")

cli_opts = argparse.ArgumentParser()
cli_opts.add_argument("--cls-model", required=True)
cli_opts.add_argument("--seg-model", required=True)
options = cli_opts.parse_args()

window = Tk()
window.geometry("640x480")
window.title("AI for Skin Cancer Detection")

ctk.FontManager.load_font(os.path.join(os.path.dirname(__file__), "assets", "fonts", "Raleway-Regular.ttf"))

cls_tasks: Queue[Image.Image] = Queue()
cls_results: Queue[tuple[int, float]] = Queue()
seg_tasks: Queue[tuple[Image.Image, bool]] = Queue()
seg_results: Queue[Image.Image] = Queue()
events = Events()

upload_frame = UploadFrame(window, options, events, cls_tasks, cls_results, seg_tasks, seg_results)
app = App(upload_frame)


def graceful_shutdown():
    events.stop_everything.set()

    if events.cls_runtime_stopped.wait(15) and events.yolo_stopped.wait(15):
        window.destroy()


window.protocol("WM_DELETE_WINDOW", lambda: graceful_shutdown())
window.mainloop()
