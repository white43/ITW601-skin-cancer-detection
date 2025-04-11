import argparse

from ultralytics import YOLO

cli_opts = argparse.ArgumentParser()
cli_opts.add_argument("--model", type=str, required=True)
cli_opts.add_argument("--imgsz", type=int, required=True)
options = cli_opts.parse_args()

model = YOLO(options.model)
print(model.export(format="onnx", imgsz=[options.imgsz, options.imgsz], opset=20, simplify=True))

