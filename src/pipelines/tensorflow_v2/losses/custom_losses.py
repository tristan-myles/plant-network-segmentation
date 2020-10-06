import tensorflow as tf

# *========================= weighted cross-entropy ==========================*
def weighted_CE(alpha: float = 0.5):
    def weighted_cross_entropy(y_true, y_pred):
        dim = tf.reduce_prod(tf.shape(y_pred)[1:])
        y_pred = tf.reshape(y_pred, [-1, dim])

        dim = tf.reduce_prod(tf.shape(y_true)[1:])
        y_true = tf.reshape(y_true, [-1, dim])
        # For numerical stability
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())

        Pt = ((y_pred * y_true) + ((1 - y_true) * (1 - y_pred)))
        CE = -tf.math.log(Pt)

        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)

        balanced_CE = alpha_t * CE

        balanced_CE_per_image = tf.reduce_mean(balanced_CE, axis=1)

        return tf.reduce_mean(balanced_CE_per_image, axis=0)

    # average loss across batches
    return weighted_cross_entropy


# *=============================== focal loss ================================*
def focal_loss(alpha: float = 0.5, gamma: float = 2):
    """
    Function closure of custom_focal_loss

    :param alpha: controls the weighting of the positive class (1) to the
    negative class (0) in the cross entropy term
    :param gamma:
    :return:
    """

    def binary_focal_loss(y_true, y_pred):
        """
        Function matching the structure expected by Keras

        :param y_true: the true label
        :param y_pred: the predicted label
        :return:
        """
        # 16 bit should be sufficient
        alpha = 0.6
        gamma = 2

        dim = tf.reduce_prod(tf.shape(y_pred)[1:])
        y_pred = tf.reshape(y_pred, [-1, dim])

        dim = tf.reduce_prod(tf.shape(y_true)[1:])
        y_true = tf.reshape(y_true, [-1, dim])
        # For numerical stability
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())

        Pt = ((y_pred * y_true) + ((1 - y_true) * (1 - y_pred)))
        CE = -tf.math.log(Pt)

        # weight balanced cross entropy loss
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)

        balanced_CE = alpha_t * CE

        modulating_factor = tf.pow((1 - Pt), gamma)

        focal_loss_custom = modulating_factor * balanced_CE

        # use sum over mean?; using a mean in an imbalanced problem may
        # under-represent doing poorly on the minority class
        focal_loss_per_image = tf.reduce_mean(focal_loss_custom, axis=1)

        # average loss for the batch
        return tf.reduce_mean(focal_loss_per_image, axis=0)
    return binary_focal_loss

# *============================= soft dice loss ==============================*
def soft_dice_loss(y_true, y_pred):
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

    # batch is first => average over the batches
    return tf.reduce_mean(1 - (numerator / denominator), axis=0)
# *===========================================================================*
