import argparse
from queue import Queue
from threading import Thread

import customtkinter as ctk
from PIL import Image

from app.src.app import App
from app.src.events import Events
from app.src.frames.upload_frame import UploadFrame
from app.src.overrides import Tk
from app.src.update import download_models
from app.src.utils import resource_path

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("dark-blue")

cli_opts = argparse.ArgumentParser()
cli_opts.add_argument("--cls-model")
cli_opts.add_argument("--seg-model")
cli_opts.add_argument("--download-from", default="https://torrens-files.s3.ap-southeast-2.amazonaws.com/ITA602")
options = cli_opts.parse_args()

window = Tk()
window.geometry("640x480")
window.title("AI for Skin Cancer Detection")

ctk.FontManager.load_font(resource_path("fonts", "Raleway-Regular.ttf"))

cls_tasks: Queue[Image.Image] = Queue()
cls_results: Queue[tuple[int, float]] = Queue()
seg_tasks: Queue[tuple[Image.Image, bool]] = Queue()
seg_results: Queue[Image.Image] = Queue()
# A queue for displaying current progress to users while models are being downloading
download_meter: Queue[tuple[int, int]] = Queue()
events = Events()

# A separate thread to download models and avoid main thread blocking
downloading = Thread(target=lambda: download_models(options, events, download_meter))
downloading.start()

upload_frame = UploadFrame(window, options, events, cls_tasks, cls_results, seg_tasks, seg_results, download_meter)
app = App(upload_frame)


def graceful_shutdown():
    events.stop_everything.set()

    if events.cls_runtime_stopped.wait(15) and events.yolo_stopped.wait(15):
        window.destroy()


window.protocol("WM_DELETE_WINDOW", lambda: graceful_shutdown())
window.mainloop()
