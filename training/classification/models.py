import keras
import tensorflow as tf
from keras.regularizers import L1L2
from keras.src.applications.efficientnet_v2 import EfficientNetV2

from training.classification.constants import INPUT_SIZES
from training.classification.options import Options


def get_input_shape_for(code: str) -> tuple[int, int, int]:
    return INPUT_SIZES[code], INPUT_SIZES[code], 3

def get_model(options: Options, labels: int) -> keras.Model:
    if options.weights == "imagenet" or options.weights is None:
        model = get_imagenet_or_random_model(options, labels)
    else:
        model = get_self_trained_model(options)

    unfreeze_layers(model, options, target=0)

    backbone = model.get_layer(options.model)

    print("Number of layers in backbone: %d" % len(backbone.layers))
    print("Total number of layers: %d" % (len(backbone.layers) + len(model.layers) - 1))

    # Print the feature extractor's layers to validate what is frozen
    model.get_layer(name=options.model).summary(show_trainable=True)

    # Print the model's layers to validate what is frozen (in Functional API we
    # don't see the feature extractor's layers here)
    model.summary(show_trainable=True)

    return model

def unfreeze_layers(model: keras.Model, options: Options, target: float) -> None:
    backbone = model.get_layer(options.model)

    # # Freeze/unfreeze the whole feature extractor if needed
    backbone.trainable = options.unfreeze > 0 and target >= options.unfreeze_at

    # Freeze/unfreeze every/some layers in the feature extractor if needed
    # https://keras.io/guides/transfer_learning/
    # https://www.tensorflow.org/tutorials/images/transfer_learning
    for i, layer in enumerate(reversed(backbone.layers)):
        if options.unfreeze > 0:
            trainable = options.unfreeze > i and target >= options.unfreeze_at
        else:
            trainable = False

        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = trainable and options.unfreeze_bn is True
        else:
            layer.trainable = trainable

    # Validate the training state of various layers
    # The layers of the feature extractor should be frozen when the option is set
    assert backbone.trainable is (True if options.unfreeze > 0 and target >= options.unfreeze_at else False)
    assert backbone.layers[-1].trainable is (True if options.unfreeze > 0 and target >= options.unfreeze_at else False)
    # The layers added on top of the feature extractor should be trainable regardless of options
    assert model.trainable is True

def get_imagenet_or_random_model(options: Options, labels: int) -> keras.Model:
    # The very first layer (input layer)
    inputs = keras.layers.Input(shape=get_input_shape_for(options.model), name='input_layer')

    # Optional preprocessing layer for some models
    preprocessing = None

    if options.model == "efficientnetv2-b0":
        # Blocks:
        # 6th: 13+15+15+15+15+15+15+18 = 121 layer
        backbone = EfficientNetV2(
            # Default values for keras.applications.EfficientNetV2B0
            width_coefficient=1.0,
            depth_coefficient=1.0,
            default_size=224,
            activation="swish",
            model_name="efficientnetv2-b0",
            input_tensor=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
            include_preprocessing=True,

            # TODO For test
            drop_connect_rate=0.2,
            bn_momentum=0.9,

            # Custom values
            include_top=False,
            weights=options.weights,
            input_shape=get_input_shape_for(options.model),
        )
    elif options.model == "efficientnetv2-s":
        backbone = keras.applications.EfficientNetV2S(
            include_top=False,
            input_shape=get_input_shape_for(options.model),
            weights=options.weights,
        )
    elif options.model == "resnet50v2":
        # https://keras.io/api/applications/resnet/#resnet50v2-function
        preprocessing = keras.applications.resnet_v2.preprocess_input(inputs)

        backbone = keras.applications.ResNet50V2(
            include_top=False,
            input_shape=get_input_shape_for(options.model),
            weights=options.weights,
        )
    elif options.model == "densenet121":
        # https://keras.io/api/applications/densenet/#densenet121-function
        preprocessing = keras.applications.densenet.preprocess_input(inputs)

        backbone = keras.applications.DenseNet121(
            include_top=False,
            input_shape=get_input_shape_for(options.model),
            weights=options.weights,
        )
    elif options.model == "densenet169":
        # https://keras.io/api/applications/densenet/#densenet169-function
        preprocessing = keras.applications.densenet.preprocess_input(inputs)

        backbone = keras.applications.DenseNet169(
            include_top=False,
            input_shape=get_input_shape_for(options.model),
            weights=options.weights,
        )
    elif options.model == "convnext_tiny":
        backbone = keras.applications.ConvNeXtTiny(
            include_top=False,
            input_shape=get_input_shape_for(options.model),
            weights=options.weights,
        )
    else:
        raise Exception("Wrong model code")

    if preprocessing is not None:
        x = backbone(preprocessing)
    else:
        x = backbone(inputs)

    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)

    if options.dropout > 0:
        x = keras.layers.Dropout(options.dropout, name="top0_dropout")(x)

    if options.neurons > 0:
        x = keras.layers.Dense(
            units=options.neurons,
            activation="swish",
            kernel_regularizer=L1L2(l1=options.l2, l2=options.l2) if options.l1 > 0 or options.l2 > 0 else None,
            bias_regularizer=L1L2(l1=options.l2, l2=options.l2) if options.l1 > 0 or options.l2 > 0 else None,
            name='top1_dense')(x)

        if options.dropout > 0:
            x = keras.layers.Dropout(options.dropout, name='top1_dropout')(x)

    outputs = keras.layers.Dense(
        units=labels,
        activation='softmax',
        dtype=tf.float32,
        name='predictions')(x)

    return keras.Model(inputs, outputs)

def get_self_trained_model(options: Options) -> keras.Model:
    return keras.models.load_model(options.weights)
