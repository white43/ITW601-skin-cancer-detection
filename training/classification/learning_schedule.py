import math
from typing import Union

import keras
from keras.src.optimizers.schedules.learning_rate_schedule import LearningRateSchedule

from training.classification.options import Options


def get_learning_rate_schedule(
        options: Options,
        ds_len: int,
) -> Union[float, keras.optimizers.schedules.LearningRateSchedule]:
    if options.lr == "flat":
        lr = float(options.lr0)
        print("Flat LR: %g" % lr)
    elif options.lr == "cosine" or options.cosine_lr is True:
        lr = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=float(options.lr0),
            alpha=float(options.lrf),
            decay_steps=math.ceil(ds_len / options.batch) * options.epochs,
        )

        print("Cosine LR schedule: %s" % str(lr.get_config()))
    elif options.lr == "one-step":
        lr = OneStep(
            lr0=options.lr0,
            lrf=options.lrf,
            step=math.ceil(ds_len / options.batch) * options.lr_step,
        )
        print("One-step LR schedule: %s" % str(lr.get_config()))
    else:
        raise ValueError("Unknown options.lr argument")

    return lr


class OneStep(LearningRateSchedule):
    """
    A simple learning schedule with one step. `lr0` is used until the `step` is reached,
    `lrf` - afterwards.
    """

    def __init__(self, lr0: float, lrf: float, step: int):
        self.lr0: float = lr0
        self.lrf: float = lrf
        self.step: int = step

    def __call__(self, step):
        if step < self.step:
            return self.lr0
        else:
            return self.lrf

    def get_config(self):
        return {
            "lr0": self.lr0,
            "lrf": self.lrf,
            "step": self.step,
        }


class FixedStep(LearningRateSchedule):
    def __init__(self, lr0: float, lrf: float, step: int, decay_rate: float):
        self.lr0: float = lr0
        self.lrf: float = lrf
        self.step: int = step
        self.decay_rate: float = decay_rate

    def __call__(self, step):
        if step > 0 and step % self.step == 0:
            if self.lr0 <= self.lrf:
                return self.lrf

            self.lr0 = max(self.lr0 * self.decay_rate, self.lrf)

            return self.lr0
        else:
            return self.lrf

    def get_config(self):
        return {
            "lr0": self.lr0,
            "lrf": self.lrf,
            "step": self.step,
            "decay_rate": self.decay_rate
        }
