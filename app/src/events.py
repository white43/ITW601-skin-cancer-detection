from threading import Event


class Events:
    def __init__(self):
        self.cls_runtime_loaded = Event()
        self.cls_runtime_stopped = Event()
        self.yolo_loaded = Event()
        self.yolo_stopped = Event()
        self.stop_everything = Event()

