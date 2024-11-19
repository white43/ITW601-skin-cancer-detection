import argparse
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

cls_tasks: Queue[Image.Image] = Queue()
cls_results: Queue[int] = Queue()
seg_tasks: Queue[tuple[Image.Image, bool]] = Queue()
seg_results: Queue[Image.Image] = Queue()
events = Events()

upload_frame = UploadFrame(window, options, events, cls_tasks, cls_results, seg_tasks, seg_results)
app = App(upload_frame)

window.mainloop()
