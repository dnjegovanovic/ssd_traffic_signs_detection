import argparse
import tensorflow as tf
import os
import sys
from network.ssdneteval import createSSD
from utils.defaultboxes import generate_default_boxes
from utils.trafficdata import TraffDataset
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='ssd300')
parser.add_argument('--pretrained-type', default='specified')
parser.add_argument('--checkpoint-dir', default='')
parser.add_argument('--checkpoint-path',
                    default=r"D:\FAX\MASTER\repo\ssd_traffic_signs_detection\checkpointsNewData\ssd_epoch_70.h5")
parser.add_argument('--gpu-id', default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

NUM_CLASSES = 247
BATCH_SIZE = 1

def exportTFLite():
    with open('../config.yml') as f:
        cfg = yaml.load(f)

    try:
        config = cfg[args.arch.upper()]
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format(args.arch))

    try:

        ssd, latestEpoch = createSSD(NUM_CLASSES, args.arch,
                                     args.pretrained_type,
                                     args.checkpoint_dir,
                                     args.checkpoint_path)
        print('Latest epoch: {}'.format(latestEpoch))

        # Convert the model.
        converter = tf.lite.TFLiteConverter.from_keras_model(ssd)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # converter.target_spec.supported_types = [tf.float16]
        # converter.representative_dataset = representative_dataset(x)
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        tflite_model = converter.convert()

        # Save the TF Lite model.
        with tf.io.gfile.GFile('../modelv3PredictBCS3.tflite', 'wb') as f:
            f.write(tflite_model)

    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()


if __name__ == '__main__':
    exportTFLite()
