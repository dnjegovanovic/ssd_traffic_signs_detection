import tensorflow as tf

def computeArea(top_left, bot_right):
    """
    Compute area given top_left and bottom_right coordinates
    :param top_left: tensor (num_boxes, 2)
    :param bot_right: tensor (num_boxes, 2)
    :return: tensor (num_boxes,)
    """
    # top_left: N x 2
    # bot_right: N x 2
    hw = tf.clip_by_value(bot_right - top_left, 0.0, 512.0)
    area = hw[..., 0] * hw[..., 1]

    return area

def computeIOU(boxes_a, boxes_b):
    """
    Compute overlap between boxes_a and boxes_b
    :param boxes_a: tensor (num_boxes_a, 4)
    :param boxes_b: tensor (num_boxes_b, 4)
    :return:
    """
    # boxes_a => num_boxes_a, 1, 4
    boxes_a = tf.expand_dims(boxes_a, 1)

    # boxes_b => 1, num_boxes_b, 4
    boxes_b = tf.expand_dims(boxes_b, 0)
    top_left = tf.math.maximum(boxes_a[..., :2], boxes_b[..., :2])
    bot_right = tf.math.minimum(boxes_a[..., 2:], boxes_b[..., 2:])

    overlap_area = computeArea(top_left, bot_right)
    area_a = computeArea(boxes_a[..., :2], boxes_a[..., 2:])
    area_b = computeArea(boxes_b[..., :2], boxes_b[..., 2:])

    overlap = overlap_area / (area_a + area_b - overlap_area)

    return overlap

def computeTarget(default_boxes, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Compute regression and classification targets
    :param default_boxes: tensor (num_default, 4)
                          of format (cx, cy, w, h)
    :param gt_boxes: tensor (num_gt, 4)
                     of format (xmin, ymin, xmax, ymax)
    :param gt_labels: tensor (num_gt,)
    :param iou_threshold:
    :return: gt_confs: classification targets, tensor (num_default,)
             gt_locs: regression targets, tensor (num_default, 4)
    """
    # Convert default boxes to format (xmin, ymin, xmax, ymax)
    # in order to compute overlap with gt boxes
    transformed_default_boxes = transformCenterToCorner(default_boxes)
    iou = computeIOU(transformed_default_boxes, gt_boxes)

    best_gt_iou = tf.math.reduce_max(iou, 1)
    best_gt_idx = tf.math.argmax(iou, 1)

    best_default_iou = tf.math.reduce_max(iou, 0)
    best_default_idx = tf.math.argmax(iou, 0)

    best_gt_idx = tf.tensor_scatter_nd_update(
        best_gt_idx,
        tf.expand_dims(best_default_idx, 1),
        tf.range(best_default_idx.shape[0], dtype=tf.int64))

    best_gt_iou = tf.tensor_scatter_nd_update(
        best_gt_iou,
        tf.expand_dims(best_default_idx, 1),
        tf.ones_like(best_default_idx, dtype=tf.float32))

    gt_confs = tf.gather(gt_labels, best_gt_idx)
    gt_confs = tf.where(
        tf.less(best_gt_iou, iou_threshold),
        tf.zeros_like(gt_confs),
        gt_confs)

    gt_boxes = tf.gather(gt_boxes, best_gt_idx)
    gt_locs = encode(default_boxes, gt_boxes)

    return gt_confs, gt_locs

def encode(default_boxes, boxes, variance=[0.1, 0.2]):
    """
    Compute regression values
    :param default_boxes: tensor (num_default, 4)
                          of format (cx, cy, w, h)
    :param boxes: tensor (num_default, 4)
                  of format (xmin, ymin, xmax, ymax)
    :param variance: variance for center point and size
    :return: regression values, tensor (num_default, 4)
    """
    # Convert boxes to (cx, cy, w, h) format
    transformed_boxes = transformCornerToCenter(boxes)

    locs = tf.concat([
        (transformed_boxes[..., :2] - default_boxes[:, :2]
         ) / (default_boxes[:, 2:] * variance[0]),
        tf.math.log(transformed_boxes[..., 2:] / default_boxes[:, 2:]) / variance[1]],
        axis=-1)

    return locs

def decode(default_boxes, locs, variance=[0.1, 0.2]):
    """
    Decode regression values back to coordinates
    :param default_boxes: tensor (num_default, 4)
                          of format (cx, cy, w, h)
    :param locs: (batch_size, num_default, 4)
                 of format (cx, cy, w, h)
    :param variance: variance for center point and size
    :return: tensor (num_default, 4)
             of format (xmin, ymin, xmax, ymax)
    """
    test =  locs[..., :2]
    a = default_boxes[:, 2:]
    locs = tf.concat([
        locs[..., :2] * variance[0] *
        default_boxes[:, 2:] + default_boxes[:, :2],
        tf.math.exp(locs[..., 2:] * variance[1]) * default_boxes[:, 2:]], axis=-1)

    boxes = transformCenterToCorner(locs)

    return boxes

def transformCornerToCenter(boxes):
    """
    Transform boxes of format (xmin, ymin, xmax, ymax)
    to format (cx, cy, w, h)
    :param boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    :return: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    """
    center_box = tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]], axis=-1)

    return center_box

def transformCenterToCorner(boxes):
    """
    Transform boxes of format (cx, cy, w, h)
        to format (xmin, ymin, xmax, ymax)
    :param boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    :return: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    corner_box = tf.concat([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2], axis=-1)

    return corner_box

def computeNms(boxes, scores, nms_threshold, limit=200):
    """
    Perform Non Maximum Suppression algorithm
        to eliminate boxes with high overlap
    :param boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    :param scores: tensor (num_boxes,)
    :param nms_threshold:
    :param limit: maximum number of boxes to keep
    :return:
    """
    if boxes.shape[0] == 0:
        return tf.constant([], dtype=tf.int32)

    selected = [0]
    idx = tf.argsort(scores, direction='DESCENDING')
    idx = idx[:limit]
    boxes = tf.gather(boxes, idx)

    iou = computeIOU(boxes, boxes)

    while True:
        row = iou[selected[-1]]
        next_indices = row <= nms_threshold
        iou = tf.where(
            tf.expand_dims(tf.math.logical_not(next_indices), 0),
            tf.ones_like(iou, dtype=tf.float32),
            iou)

        if not tf.math.reduce_any(next_indices):
            break

        selected.append(tf.argsort(
            tf.dtypes.cast(next_indices, tf.int32), direction='DESCENDING')[0].numpy())

    return tf.gather(idx, selected)