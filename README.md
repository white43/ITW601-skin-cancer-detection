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
