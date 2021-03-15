from PIL import Image
import random
import numpy as np
from .boxutils import computeIOU
import tensorflow as tf

def generatePatch(boxes, threshold):
    """

    @param boxes:
    @param threshold:
    @return:
    """
    while True:
        patch_w = random.uniform(0.1, 1)
        scale = random.uniform(0.5, 2)
        patch_h = patch_w * scale
        patch_xmin = random.uniform(0, 1 - patch_w)
        patch_ymin = random.uniform(0, 1 - patch_h)
        patch_xmax = patch_xmin + patch_w
        patch_ymax = patch_ymin + patch_h
        patch = np.array(
            [[patch_xmin, patch_ymin, patch_xmax, patch_ymax]],
            dtype=np.float32)
        patch = np.clip(patch, 0.0, 1.0)
        ious = computeIOU(tf.constant(patch), boxes)
        if tf.math.reduce_any(ious >= threshold):
            break

    return patch[0], ious[0]

def randomPatching(img, boxes, labels):
    """

    @param img:
    @param boxes:
    @param labels:
    @return:
    """
    threshold = np.random.choice(np.linspace(0.1, 0.7, 4))

    patch, ious = generatePatch(boxes, threshold)

    box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    keep_idx = (
        (ious > 0.3) &
        (box_centers[:, 0] > patch[0]) &
        (box_centers[:, 1] > patch[1]) &
        (box_centers[:, 0] < patch[2]) &
        (box_centers[:, 1] < patch[3])
    )

    if not tf.math.reduce_any(keep_idx):
        return img, boxes, labels

    img = img.crop(patch)

    boxes = boxes[keep_idx]
    patch_w = patch[2] - patch[0]
    patch_h = patch[3] - patch[1]
    boxes = tf.stack([
        (boxes[:, 0] - patch[0]) / patch_w,
        (boxes[:, 1] - patch[1]) / patch_h,
        (boxes[:, 2] - patch[0]) / patch_w,
        (boxes[:, 3] - patch[1]) / patch_h], axis=1)
    boxes = tf.clip_by_value(boxes, 0.0, 1.0)

    labels = labels[keep_idx]

    return img, boxes, labels


def horizontalFlip(img, boxes, labels):
    """

    @param img:
    @param boxes:
    @param labels:
    @return:
    """
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    boxes = tf.stack([
        1 - boxes[:, 2],
        boxes[:, 1],
        1 - boxes[:, 0],
        boxes[:, 3]], axis=1)

    return img, boxes, labels