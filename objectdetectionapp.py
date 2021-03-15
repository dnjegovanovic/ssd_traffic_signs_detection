import os
import sys

import argparse
import tensorflow as tf
import sys
import numpy as np
import yaml
import cv2

from utils.defaultboxes import generate_default_boxes
from utils.boxutils import decode, computeNms

from network.ssdnet import createSSD

parser = argparse.ArgumentParser()

parser.add_argument('-config')#, default='../config.yml'
parser.add_argument('-checkpointPath')
#default=r"D:\FAX\MASTER\repo\ssd_traffic_signs_detection\checkpointsNewData\ssd_epoch_70.h5"

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
NUM_CLASSES = 247
BATCH_SIZE = 1


def initNetwork():
    print("Config path: {}".format(args.config))
    with open(args.config) as f:

        cfg = yaml.load(f)

    try:
        config = cfg['SSD300']
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format('ssd300'))

    default_boxes = generate_default_boxes(config)

    try:
        ssd, latestEpoch = createSSD(NUM_CLASSES, 'ssd300',
                                     'specified',
                                     '',
                                     args.checkpointPath)

        print('Latest epoch: {}'.format(latestEpoch))

    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()

    return default_boxes, ssd


def predict(ssd, imgs, default_boxes):
    confs, locs = ssd(imgs)

    confs = tf.squeeze(confs, 0)
    locs = tf.squeeze(locs, 0)

    confs = tf.math.softmax(confs, axis=-1)
    classes = tf.math.argmax(confs, axis=-1)
    scores = tf.math.reduce_max(confs, axis=-1)

    boxes = decode(default_boxes, locs)

    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, NUM_CLASSES):
        cls_scores = confs[:, c]

        score_idx = cls_scores > 0.6
        # cls_boxes = tf.boolean_mask(boxes, score_idx)
        # cls_scores = tf.boolean_mask(cls_scores, score_idx)
        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]

        nms_idx = computeNms(cls_boxes, cls_scores, 0.45, 200)
        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)
        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()

    return boxes, classes, scores


def run():
    db, ssd = initNetwork()
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        retVal, img = cam.read()
        if cv2.waitKey(1) == 27:
            break

        img_resOrg = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
        img_res = np.array(img_resOrg,dtype=np.float32)
        img_res=img_res.reshape(-1, 300, 300, 3)
        img_res = (img_res / 127.0) - 1
        boxes, classes, scores = predict(ssd, img_res, db)
        boxes *= (img.shape[1], img.shape[0]) * 2

        for i, box in enumerate(boxes):
            top_left = (box[0], box[1])
            bot_right = (box[2], box[3])
            cv2.rectangle(img, top_left, bot_right, (0, 0, 255), 2)
            sc = round(float(scores[i]),2)
            cv2.putText(img,"sc:{}-cls:{}".format(sc,classes[i]),(int(top_left[0]),int(top_left[1]-10)),font,0.5,(255, 0, 0), 2, cv2.LINE_AA)
        # cv2.imwrite("test.jpg", testImg)
        cv2.imshow('my webcam', img)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
