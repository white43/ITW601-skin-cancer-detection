import os
import sys


def resource_path(*relative_path):
    try:
        base_path = os.path.join(sys._MEIPASS, "assets")
    except Exception:
        base_path = os.path.join(os.getcwd(), "app", "assets")

    return os.path.join(base_path, *relative_path)
