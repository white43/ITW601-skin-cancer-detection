import argparse
import os

import numpy as np
from PIL import Image, JpegImagePlugin
from tqdm import tqdm

from common.shades_of_grey import shades_of_grey

cli_opts = argparse.ArgumentParser()

cli_opts.add_argument("--input", type=str, required=True)
cli_opts.add_argument("--output", type=str, required=True)

options = cli_opts.parse_args()

work: list[tuple[str, str]] = []

for kind in os.listdir(options.input):
    for label in os.listdir(os.path.join(options.input, kind)):
        if not os.path.exists(os.path.join(options.output, kind, label)):
            os.makedirs(os.path.join(options.output, kind, label))

        for image in os.listdir(os.path.join(options.input, kind, label)):
            if not image.endswith(".jpg"):
                continue

            if os.path.exists(os.path.join(options.output, kind, label, image)):
                continue

            work.append((
                os.path.join(options.input, kind, label, image),
                os.path.join(options.output, kind, label, image),
            ))

for source, target in tqdm(work):
    with Image.open(source) as img:
        sampling = JpegImagePlugin.get_sampling(img)
        quant = img.quantization

        imgnp = np.array(img)
        imgnp = shades_of_grey(imgnp, norm_p=6)
        Image.fromarray(imgnp, "RGB").save(
            target,
            subsampling=sampling,
            qtables=quant,
            quality=100,
        )
