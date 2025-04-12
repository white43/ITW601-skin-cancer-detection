import os
from queue import Queue

import appdirs
import requests
from CTkMessagebox import CTkMessagebox

from app.src.events import Events
from app.src.options import Options


def download_models(options: Options, events: Events, meter: Queue[tuple[int, int]]) -> dict[str, str] | None:
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

    cls_path: str | None = None
    seg_path: str | None = None
    result: dict[str, str] = dict()
    total_size: int = 0

    if not options.cls_model and "cls" in latest:
        cls_path = os.path.join(cache_dir, latest["cls"]["file"])
        options.cls_model = cls_path

        if not os.path.exists(cls_path):
            total_size += int(latest["cls"]["size"])

    if not options.seg_model and "seg" in latest:
        seg_path = os.path.join(cache_dir, latest["seg"]["file"])
        options.seg_model = seg_path

        if not os.path.exists(seg_path):
            total_size += int(latest["seg"]["size"])

    if cls_path is not None and not os.path.exists(cls_path):
        _download_model(options, latest["cls"]["file"], cls_path, meter, total_size)

    if seg_path is not None and not os.path.exists(seg_path):
        _download_model(options, latest["seg"]["file"], seg_path, meter, total_size)

    events.models_downloaded.set()

    return result


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
