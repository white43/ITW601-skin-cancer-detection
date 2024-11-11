import customtkinter as ctk
from tkinterdnd2 import TkinterDnD


# https://stackoverflow.com/a/75527642
class Tk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)
