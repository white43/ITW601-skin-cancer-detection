import os
from argparse import Namespace
from queue import Queue

import requests
from CTkMessagebox import CTkMessagebox

from app.src.events import Events
from app.src.utils import cwd


def download_models(options: Namespace, events: Events, meter: Queue[tuple[int, int]]) -> dict[str, str] | None:
    response = requests.get(options.download_from + "/models.json")

    if not response.ok:
        CTkMessagebox(
            icon="warning",
            title="Error",
            message="Could not download model updates data",
            font=("Raleway", 14)
        )

        return None

    latest = response.json()

    result: dict[str, str] = dict()
    total_size: int = 0

    if not options.cls_model and "cls" in latest:
        total_size += int(latest["cls"]["size"])

    if not options.cls_model and "seg" in latest:
        total_size += int(latest["seg"]["size"])

    if not options.cls_model and "cls" in latest:
        _download_model(options, latest["cls"]["file"], meter, total_size)
        options.cls_model = os.path.join(cwd(), "models", latest["cls"]["file"])

    if not options.cls_model and "seg" in latest:
        _download_model(options, latest["seg"]["size"], meter, total_size)
        options.cls_model = os.path.join(cwd(), "models", latest["seg"]["file"])

    events.models_downloaded.set()

    return result


def _download_model(options: Namespace, filename: str, meter: Queue[tuple[int, int]], total_size: int) -> None:
    models_dir = os.path.join(cwd(), "models")

    if not os.path.exists(models_dir):
        os.mkdir(models_dir, mode=0o755)

    cls_path = os.path.join(cwd(), "models", filename)

    if not os.path.exists(cls_path):
        cls_response = requests.get(options.download_from + "/" + filename, stream=True)

        if not cls_response.ok:
            return

        with open(cls_path, mode="xb") as fp:
            for chunk in cls_response.iter_content(chunk_size=4096):
                if chunk:
                    fp.write(chunk)
                    fp.flush()
                    os.fsync(fp.fileno())

                    meter.put((4096, total_size))
