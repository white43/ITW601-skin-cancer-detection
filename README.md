# Desktop App 

## Installation 

1. `virtualenv venv`
2. `source venv/bin/activate`
3. `pip install -r app/requirements.txt`

## How to run the app

```shell
python -m app.main --cls-model classification-model.onnx --seg-model path-to-model.pt
```

## How to package the app

```shell
pyinstaller -F -s \
    --collect-all tkinterdnd2.tkdnd.linux-x64 \
    --collect-all PIL._tkinter_finder \
    --add-data "./app/assets/*.png:./assets/" \
    --add-data "./app/assets/fonts/*.ttf:./assets/fonts/" \
    ./app/main.py
```

# Training

## Installation

At the moment, Tensorflow 2.14 requires Python 3.11 to work. 

1. `virtualenv venv`
2. `source venv/bin/activate`
3. `pip install -r training/requirements.txt`

## How to download datasets for classification

```shell
python -m training.classification.download --cache isic2018-datasets --target isic2018-classification
```

Where the `cache` and `target` arguments are the directories (relative or 
absolute) to store zipped and unzipped data, respectfully.
