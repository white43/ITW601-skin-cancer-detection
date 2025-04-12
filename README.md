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

## How to download datasets

The dataset for classification needs about ~6GB of free space on your disk 
(including its archive), while the dataset for segmentation will need ~25GB
of free space (including its archive).

```shell
python -m training.classification.download --cache isic2018-datasets --target isic2018-classification
python -m training.segmentation.download --cache isic2018-datasets --target isic2018-segmentation
```

Where the `cache` and `target` arguments are the directories (relative or 
absolute) to store zipped and unzipped data, respectfully.

## How to evaluate trained models

Trained models are stored in the directory named `runs` with two subdirectories
for classification models and segmentation. Each model has a distinct command 
for evaluation.

The following command prints a few metrics (categorical loss and mean recall) 
to stdout: 

```shell
python -m training.evaluation.tf --models path/to/model1.leras path/to/model2.leras --quick
```

A more comprehensive approach is to use other command that evaluates roughly a 
dozen of metrics including but not limited to per-class accuracy, sensitivity 
(recall), specificity, dice score and AUC, as well as their mean values. This 
command requires providing a file containing ground truth. This file is 
downloaded among other files using `training.classification.download` and 
typically localed under `isic2018-datasets` directory. A file-accumulator will
be used to average individual results in multimodel mode.

```shell
python -m training.evaluation.tf --models path/to/model1.leras path/to/model2.leras \
    --ground-truth isic2018-datasets/ISIC2018_Task3_Test_GroundTruth.csv
    --reduce ~/Desktop/accumulator.csv
```
