from .frames.upload_frame import UploadFrame


class App:
    def __init__(self, upload_frame: UploadFrame):
        self.upload_frame: UploadFrame = upload_frame
        self.show_upload_frame()

    def show_upload_frame(self):
        self.upload_frame.redraw_page()
        self.upload_frame.tkraise()
