import argparse
import csv
import os
import pathlib
from argparse import Namespace
from operator import itemgetter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import numpy as np
import tensorflow as tf
from isic_challenge_scoring import ClassificationScore, ClassificationMetric
from keras.src.metrics import CategoricalCrossentropy
from tqdm import tqdm

from training.classification.constants import LABELS
from training.classification.datasets import get_test_dataset
from training.classification.metrics import MeanRecall
from training.classification.models import get_input_shape_for
from training.classification.options import Options

cli_opts = argparse.ArgumentParser()
cli_opts.add_argument("--models", nargs='+', required=True, help="A list of models to run predictions against")
cli_opts.add_argument("--quick", action='store_true', default=False, help="Use built-in evaluate() method on a model")
cli_opts.add_argument("--ground-truth", type=str, default=None, help="A CSV file with the ground truth")
args = cli_opts.parse_args()

if not args.quick and not args.ground_truth:
    print("Ground truth required")
    exit(1)


class Evaluation:
    def evaluate(self, model_path: str, opts: Options, ground_truth: str | None = None) -> None:
        pass


def evaluation_factory(args: Namespace) -> Evaluation:
    if args.quick is True:
        return QuickEvaluation()
    else:
        return ComprehensiveEvaluation()


class QuickEvaluation(Evaluation):
    """
    This class represents a quick way of how to evaluate your model after training.
    It prints two metrics values (loss and mean recall) to stdout
    """

    def evaluate(self, model_path: str, opts: Options, ground_truth: str | None = None) -> None:
        _, ds = get_test_dataset(options, get_input_shape_for(options.model), LABELS)

        model_class = keras.models.load_model(model_path)
        model_class.compile(metrics=[CategoricalCrossentropy(name='loss'), MeanRecall(num_labels=len(LABELS))])
        model_class.evaluate(ds)


class ComprehensiveEvaluation(Evaluation):
    """
    This class represents another way of how to evaluate your model's performance
    by using pip package called isic-challenge-scoring. This package prints
    significantly more information to stdout including but not limited to per-class
    accuracy, sensitivity (recall), specificity, dice score and AUC, as well as
    their mean values.
    """

    def evaluate(self, model_path: str, opts: Options, ground_truth: str | None = None) -> None:
        results = []

        files, ds = get_test_dataset(opts, get_input_shape_for(options.model), LABELS)

        model_class = keras.models.load_model(model_path)
        model_class.compile(metrics=[CategoricalCrossentropy(name='loss'), MeanRecall(num_labels=len(LABELS))])

        ds = tf.data.Dataset.zip(files, ds).as_numpy_iterator()
        length = len(files)

        for file_batch, [image_batch, _] in tqdm(ds, total=length):
            file_batch.astype('U')
            predictions = model_class.predict_on_batch(image_batch)
            predictions = np.argmax(predictions, axis=1)

            for file, prediction in zip(file_batch, predictions):
                one_hot = np.zeros(len(LABELS))
                one_hot[prediction] = 1.0
                results.append([os.path.basename(str(file))[:-5]] + one_hot.tolist())

        with open(model_path + ".csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image'] + LABELS)
            writer.writerows(sorted(results, key=itemgetter(0)))

        print("Model: %s" % os.path.basename(model_path))

        score = ClassificationScore.from_file(
            pathlib.Path(ground_truth),
            pathlib.Path(model_path + ".csv"),
            ClassificationMetric(ClassificationMetric.BALANCED_ACCURACY.value),
        )

        print(score.to_string())


test = evaluation_factory(args)

for model in args.models:
    options: Options = Options.load_from(model)
    print("Testing [", model, "]  with the following config {", options, "}")

    test.evaluate(
        model_path=model,
        opts=options,
        ground_truth=args.ground_truth,
    )
