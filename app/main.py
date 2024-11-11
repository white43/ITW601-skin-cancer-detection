from queue import Queue
import argparse

import customtkinter as ctk
from PIL import Image

from .src.app import App
from .src.events import Events
from .src.frames.upload_frame import UploadFrame
from .src.overrides import Tk

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("dark-blue")

cli_opts = argparse.ArgumentParser()
cli_opts.add_argument("--model", required=True)
options = cli_opts.parse_args()

window = Tk()
window.geometry("640x480")
window.title("AI for Skin Cancer Detection")

tasks: Queue[tuple[Image.Image, ctk.CTkButton, ctk.CTkLabel]] = Queue()
results: Queue[int] = Queue()
events = Events()

upload_frame = UploadFrame(window, options, events, tasks, results)
app = App(upload_frame)

window.mainloop()
