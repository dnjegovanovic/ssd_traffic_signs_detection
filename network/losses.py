import tensorflow as tf


def hardNegativeMining(loss, gt_confs, neg_ratio):
    """
    Instead of using all the negative examples, we sort them using the highest confidence
    loss for each default box and pick the top ones so that the ratio between the negatives and positives is at most 3:1.
    This can lead to faster optimization and a more stable training.
    :param loss: list of classification losses of all default boxes (B, num_default)
    :param gt_confs: classification targets (B, num_default)
    :param neg_ratio: negative / positive ratio
    :return:
    """
    # loss: B x N
    # gt_confs: B x N
    pos_idx = gt_confs > 0
    num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.int32), axis=1)
    num_neg = num_pos * neg_ratio

    rank = tf.argsort(loss, axis=1, direction='DESCENDING')
    rank = tf.argsort(rank, axis=1)
    neg_idx = rank < tf.expand_dims(num_neg, 1)

    return pos_idx, neg_idx


class SSDLosses(object):
    """
    Class for SSD Losses
    """

    def __init__(self, neg_ratio, num_classes):
        self.neg_ratio = neg_ratio
        self.num_classes = num_classes

    def __call__(self, confs, locs, gt_confs, gt_locs):
        """
        Compute losses for SSD
        regression loss: smooth L1
        classification loss: cross entropy
        :param confs: outputs of classification heads (B, num_default, num_classes)
        :param locs: outputs of regression heads (B, num_default, 4)
        :param gt_confs: classification targets (B, num_default)
        :param gt_locs: regression targets (B, num_default, 4)
        :return: classification loss, regression loss
        """
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        # compute classification losses
        # without reduction
        temp_loss = cross_entropy(
            gt_confs, confs)
        pos_idx, neg_idx = hardNegativeMining(
            temp_loss, gt_confs, self.neg_ratio)

        # classification loss will consist of positive and negative examples

        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='sum')
        smooth_l1_loss = tf.keras.losses.Huber(reduction='sum')

        conf_loss = cross_entropy(
            gt_confs[tf.math.logical_or(pos_idx, neg_idx)],
            confs[tf.math.logical_or(pos_idx, neg_idx)])

        # regression loss only consist of positive examples
        loc_loss = smooth_l1_loss(
            gt_locs[pos_idx],
            locs[pos_idx])

        num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.float32))

        conf_loss = conf_loss / num_pos
        loc_loss = loc_loss / num_pos

        return conf_loss, loc_loss


def createLosses(neg_ratio, num_classes):
    criterion = SSDLosses(neg_ratio, num_classes)

    return criterion
