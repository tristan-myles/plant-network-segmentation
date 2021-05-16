import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, nn_ops


# *========================= weighted cross-entropy ==========================*
class WeightedCE(tf.keras.losses.Loss):
    def __init__(self, alpha: float = 0.5, name="WeightedCE"):
        super().__init__(name=name)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # For numerical stability
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())

        Pt = ((y_pred * y_true) + ((1 - y_true) * (1 - y_pred)))
        CE = -tf.math.log(Pt)

        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)

        balanced_CE = alpha_t * CE

        return tf.reduce_mean(balanced_CE, axis=-1)


# *=============================== focal loss ================================*
class FocalLossV2(tf.keras.losses.Loss):
    def __init__(self, alpha: float = 0.25, gamma=2, name="FocalLossV2"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        logits = ops.convert_to_tensor(y_pred.op.inputs[0])
        labels = ops.convert_to_tensor(y_true)
        labels.get_shape().merge_with(logits.get_shape())
        # log_weight = 1 + (20 - 1) * labels
        CE = math_ops.add((1 - labels) * logits, (
                math_ops.log1p(math_ops.exp(-math_ops.abs(logits)))
                + nn_ops.relu(-logits)), name="test")

        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())

        Pt = ((y_pred * y_true) + ((1 - y_true) * (1 - y_pred)))
        # CE = -tf.math.log(Pt)

        # weight balanced cross entropy loss
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)

        balanced_CE = alpha_t * CE

        modulating_factor = tf.pow((1 - Pt), self.gamma)

        focal_loss_custom = modulating_factor * balanced_CE

        return K.mean(focal_loss_custom, axis=-1)


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha: float = 0.25, gamma=2, name="FocalLoss"):
        """
        :param alpha: controls the weighting of the positive class (1) to the
         negative class (0) in the cross entropy term
        :param gamma:
        :return:

        """
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        """
        Function matching the structure expected by Keras

        :param y_true: the true label
        :param y_pred: the predicted label
        :return:
        """
        # For numerical stability
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())

        Pt = ((y_pred * y_true) + ((1 - y_true) * (1 - y_pred)))
        CE = -tf.math.log(Pt)

        # weight balanced cross entropy loss
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)

        balanced_CE = alpha_t * CE

        modulating_factor = tf.pow((1 - Pt), self.gamma)

        focal_loss_custom = modulating_factor * balanced_CE

        # use sum over mean?; using a mean in an imbalanced problem may
        # under-represent doing poorly on the minority class
        # focal_loss_per_image = tf.reduce_mean(focal_loss_custom, axis=1)

        return tf.reduce_mean(focal_loss_custom, axis=-1)


# *============================= soft dice loss ==============================*
class SoftDiceLoss(tf.keras.losses.Loss):
    def __init__(self, name="SoftDiceLoss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # reshaping y_pred and y_true
        # more robust to work with flat arrays
        dim = tf.reduce_prod(tf.shape(y_pred)[1:])
        y_pred = tf.reshape(y_pred, [-1, dim])

        dim = tf.reduce_prod(tf.shape(y_true)[1:])
        y_true = tf.reshape(y_true, [-1, dim])

        numerator = 2 * tf.reduce_sum(tf.multiply(y_true, y_pred), axis=1)
        numerator = tf.clip_by_value(numerator, tf.keras.backend.epsilon(),
                                     tf.float32.max)

        # Squaring the cardinality is less strict since a smaller denominator
        # would mean a larger dice coefficient => loss closer to 0
        denominator = tf.add(tf.reduce_sum(y_true, axis=1),
                             tf.reduce_sum(y_pred, axis=1))
        denominator = tf.clip_by_value(denominator, tf.keras.backend.epsilon(),
                                       tf.float32.max)

        return tf.reduce_mean(1 - (numerator / denominator), axis=-1)
# *===========================================================================*
