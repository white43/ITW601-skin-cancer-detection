from typing import Union

import keras

from training.classification.options import Options


def get_optimizer(
        options: Options,
        lr: Union[float, keras.optimizers.schedules.LearningRateSchedule],
) -> Union[keras.optimizers.Optimizer, keras.optimizers.legacy.Optimizer]:
    if options.optimizer == "SGD":
        if options.unfreeze_at > 0:
            return keras.optimizers.legacy.SGD(
                learning_rate=lr,
                momentum=options.sgd_momentum,
                nesterov=options.nesterov,
            )
        else:
            return keras.optimizers.SGD(
                learning_rate=lr,
                momentum=options.sgd_momentum,
                weight_decay=options.weight_decay if options.weight_decay > 0 else None,
                nesterov=options.nesterov,
                use_ema=True if options.ema > 0 else False,
                ema_momentum=options.ema,
            )
    elif options.optimizer == "Adam":
        if options.unfreeze_at > 0:
            return keras.optimizers.legacy.Adam(
                learning_rate=lr,
                momentum=options.sgd_momentum,
                nesterov=options.nesterov,
            )
        else:
            return keras.optimizers.Adam(
                learning_rate=lr,
                momentum=options.sgd_momentum,
                weight_decay=options.weight_decay if options.weight_decay > 0 else None,
                nesterov=options.nesterov,
                use_ema=True if options.ema > 0 else False,
                ema_momentum=options.ema,
            )
    elif options.optimizer == "AdamW":
        return keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=options.weight_decay if options.weight_decay > 0 else None,
            beta_1=options.adam_b1,
            beta_2=options.adam_b2,
            amsgrad=options.amsgrad,
            use_ema=True if options.ema > 0 else False,
            ema_momentum=options.ema,
        )
    elif options.optimizer == "Lion":
        return keras.optimizers.Lion(
            learning_rate=lr,
            weight_decay=options.weight_decay if options.weight_decay > 0 else None,
            use_ema=True if options.ema > 0 else False,
            ema_momentum=options.ema,
        )
