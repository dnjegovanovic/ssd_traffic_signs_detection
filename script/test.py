import argparse
import tensorflow as tf
import os
import sys
import numpy as np
import yaml
from tqdm import tqdm
from PIL import Image
import pickle

from utils.trafficdata import createTraffBatchGenerator
from utils.defaultboxes import generate_default_boxes
from utils.boxutils import decode, computeNms
from network.ssdnet import createSSD
from network.ssdneteval import createSSD as createSSDEval
from utils.imagevisualize import ImageVisualizer

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='D:\FAX\MASTER\LISADFG')
parser.add_argument('--img-dir', default='TData2')
parser.add_argument('--arch', default='ssd300')
parser.add_argument('--num-examples', default=-1, type=int)
parser.add_argument('--pretrained-type', default='specified')
parser.add_argument('--checkpoint-dir', default='')
parser.add_argument('--checkpoint-path', default=r"D:\FAX\MASTER\repo\ssd_traffic_signs_detection\checkpointsNewData\ssd_epoch_70.h5")
parser.add_argument('--gpu-id', default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

NUM_CLASSES = 247
BATCH_SIZE = 1

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

def test_net():
    # with open(r"D:\FAX\MASTER\LISADFG\classToIds.p", 'rb') as f:
    #     dataLoad = pickle.load(f)

    with open('./config.yml') as f:
        cfg = yaml.load(f)

    try:
        config = cfg[args.arch.upper()]
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format(args.arch))

    default_boxes = generate_default_boxes(config)
    batch_generator, info = createTraffBatchGenerator(args.data_dir,
                                                      args.img_dir,
                                                      default_boxes,
                                                      config['image_size'],
                                                      BATCH_SIZE,
                                                      args.num_examples,
                                                      augmentation=True,
                                                      mode='test')

    try:
        ssd, latestEpoch = createSSD(NUM_CLASSES, args.arch,
                                     args.pretrained_type,
                                     args.checkpoint_dir,
                                     args.checkpoint_path)

        print('Latest epoch: {}'.format(latestEpoch))

    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()

    os.makedirs('outputsMergeData/images', exist_ok=True)
    os.makedirs('outputsMergeData/detects', exist_ok=True)
    visualizer = ImageVisualizer(info['idx_to_name'], save_dir='outputsMergeData/images')

    for i, (filename, imgs, gt_confs, gt_locs) in enumerate(
            tqdm(batch_generator, total=info['length'],
                 desc='Testing...', unit='images')):

        boxes, classes, scores = predict(ssd, imgs, default_boxes)

        filename = filename.numpy()[0].decode()
        original_image = Image.open(
            os.path.join(args.data_dir, info['image_dir'], filename))
        boxes *= original_image.size * 2
        visualizer.save_image(
            original_image, boxes, classes, '{}.jpg'.format(filename))

        log_file = os.path.join('outputsMergeData/detects', '{}.txt')

        for cls, box, score in zip(classes, boxes, scores):
            cls_name = info['idx_to_name'][cls]
            with open(log_file.format(cls_name), 'a') as f:
                f.write('{} {} {} {} {} {}\n'.format(
                    filename,
                    score,
                    *[coord for coord in box]))
