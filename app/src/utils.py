import os
import sys


def cwd() -> str:
    try:
        return sys._MEIPASS
    except Exception:
        return os.path.join(os.getcwd(), "app")


def resource_path(*relative_path) -> str:
    return os.path.join(cwd(), "assets", *relative_path)
