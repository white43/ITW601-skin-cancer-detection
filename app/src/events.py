from threading import Event


class Events:
    def __init__(self):
        self.tensorflow_loaded = Event()
        self.yolo_loaded = Event()
