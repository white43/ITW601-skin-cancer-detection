import os
from queue import Queue

import appdirs
import requests
from CTkMessagebox import CTkMessagebox

from app.src.events import Events
from app.src.options import Options, ClsModel


def download_models(options: Options, events: Events, meter: Queue[tuple[int, int]]):
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

    cache_dir = os.path.join(appdirs.user_cache_dir(appname="ITA602"), "models")

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, mode=0o755)

    to_download: list[str] = []
    total_size: int = 0

    if not options.seg_model and "seg" in latest:
        seg_path = os.path.join(cache_dir, latest["seg"]["file"])
        options.seg_model = seg_path

        if not os.path.exists(seg_path):
            to_download.append(latest["seg"]["file"])
            total_size += int(latest["seg"]["size"])

    if "cls" in latest:
        for cls_model in latest["cls"]:
            cls_path = os.path.join(cache_dir, cls_model["file"])

            options.cls_models.append(ClsModel(
                name=cls_path,
                augmentations=cls_model["augmentations"] if "augmentations" in cls_model else {},
            ))

            if not os.path.exists(cls_path):
                to_download.append(cls_model["file"])
                total_size += int(cls_model["size"])

    for model in to_download:
        _download_model(options, model, os.path.join(cache_dir, model), meter, total_size)

    events.models_downloaded.set()


def _download_model(
    options: Options,
    filename: str,
    dest: str,
    meter: Queue[tuple[int, int]],
    total_size: int,
) -> None:
    if not os.path.exists(dest):
        cls_response = requests.get(options.download_from + "/" + filename, stream=True)

        if not cls_response.ok:
            return

        with open(dest, mode="xb") as fp:
            for chunk in cls_response.iter_content(chunk_size=4096):
                if chunk:
                    fp.write(chunk)
                    fp.flush()
                    os.fsync(fp.fileno())

                    meter.put((4096, total_size))
