import keras
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import approximate_equal


class MeanRecall(tf.keras.metrics.Metric):
    def __init__(self, num_labels: int, name=None, dtype=None, **kwargs):
        super().__init__(name, dtype, **kwargs)

        self.num_labels: int = num_labels
        self.y_true = self.add_weight(name="y_true", shape=num_labels, initializer="zeros", dtype=self.dtype)
        self.y_pred = self.add_weight(name="y_pred", shape=num_labels, initializer="zeros", dtype=self.dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self._ensure_sample_weight_is_none(sample_weight)
        self._ensure_rank_sizes(y_pred, y_true)
        tf.ensure_shape(y_true, y_pred.shape)

        y_true = tf.cast(y_true, dtype=self.dtype)
        y_pred = tf.cast(y_pred, dtype=self.dtype)

        # Example 2D tensor ground truth (y_true):
        # [[0, 0, 1, 0, 0]
        #  [0, 0, 1, 0, 0]]

        # Example 2D tensor predictions (y_pred):
        # [[0.1, 0.2, 0.7, 0.2, 0.1]
        #  [0.1, 0.7, 0.2, 0.2, 0.1]]

        # Create a scalar from 1D tensor or 1D tensor from 2D tensor from the
        # predictions using argmax (the most probable classes)
        # [2, 1]
        y_pred = tf.math.argmax(y_pred, axis=y_pred.shape.rank - 1)
        # Convert the most probable classes to one-hot encoded 1D or 2D vectors
        # [[0, 0, 1, 0, 0]
        #  [0, 1, 0, 0, 0]]
        y_pred = tf.one_hot(y_pred, self.num_labels, dtype=self.dtype)

        # Find predictions that are correct (in False/True format) and convert
        # them to 0/1 format
        # [[1, 1, 1, 1, 1]
        #  [1, 0, 0, 1, 1]]
        match = tf.cast(approximate_equal(y_true, y_pred), dtype=self.dtype)

        # Leave only true positives (where 1*1). This zeroes true negatives (0*0),
        # false positives (0*1) and false negatives (1*0)
        # Multiply element-wisely two tensors
        #
        # [[0, 0, 1, 0, 0]
        #  [0, 1, 0, 0, 0]]
        #
        # [[1, 1, 1, 1, 1]
        #  [1, 0, 0, 1, 1]]
        #
        # The product is:
        # [[0, 0, 1, 0, 0]
        #  [0, 0, 0, 0, 0]]
        y_pred = tf.multiply(y_pred, match)

        # If we have a 2D tensor, sum ground truth classes by columns
        # [0, 0, 2, 0, 0]
        if y_true.shape.rank == 2:
            y_true = tf.reduce_sum(y_true, axis=0)

        # If we have a 2D tensor, sum prediction classes by columns
        # [0, 0, 1, 0, 0]
        if y_pred.shape.rank == 2:
            y_pred = tf.reduce_sum(y_pred, axis=0)

        self.y_true.assign_add(y_true)
        self.y_pred.assign_add(y_pred)

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype, "num_labels": self.num_labels}

    def result(self):
        per_class_accuracy = tf.math.divide_no_nan(self.y_pred, self.y_true)
        return tf.reduce_mean(per_class_accuracy)

    def reset_state(self):
        keras.backend.batch_set_value(
            [(v, tf.zeros(v.shape.as_list())) for v in self.variables]
        )

    @staticmethod
    def _ensure_sample_weight_is_none(sample_weight):
        if sample_weight is not None:
            raise ValueError("For MeanRecall, sample_weight argument must be " +
                             "None, since we compute mean recall across all " +
                             "labels regardless of their size/weight")

    @staticmethod
    def _ensure_rank_sizes(y_pred, y_true):
        if y_true.shape.rank not in [1, 2] or y_pred.shape.rank not in [1, 2]:
            raise ValueError("Ranks of y_true and y_pred must be 1 or 2. " +
                             "%d and %d given, respectively" % (y_true.shape.rank, y_pred.shape.rank))
