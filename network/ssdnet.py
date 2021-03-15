from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
import os

from .layers import createVGG16Layers, createExtraLayers, createConfHeadLayers, createLocHeadLayers


class SSD(Model):
    """
    Class for SSD model
    """

    def __init__(self, numClasses, arch='ssd300'):
        super(SSD, self).__init__()
        self.numClasses = numClasses
        self.vgg16Conv4, self.vgg16Conv7 = createVGG16Layers()
        self.batchNorm = layers.BatchNormalization(
            beta_initializer='glorot_uniform',
            gamma_initializer='glorot_uniform'
        )
        self.extraLayers = createExtraLayers()
        self.confHeadLayers = createConfHeadLayers(numClasses)
        self.locHeadLayers = createLocHeadLayers()

        if arch == 'ssd300':
            self.extraLayers.pop(-1)
            self.confHeadLayers.pop(-2)
            self.locHeadLayers.pop(-2)

    def computeHeads(self, x, idx):
        """
        Compute outputs of classification and regression heads
        :param x: the input feature map
        :param idx: index of the head layer
        :return:conf: output of the idx-th classification head
                loc: output of the idx-th regression head
        """
        conf = self.confHeadLayers[idx](x)
        conf = tf.reshape(conf, [conf.shape[0], -1, self.numClasses])

        loc = self.locHeadLayers[idx](x)
        loc = tf.reshape(loc, [loc.shape[0], -1, 4])

        return conf, loc

    def initVGG16(self):
        """
        Initialize the VGG16 layers from pretrained weights
        and the rest from scratch using xavier initializer
        """
        originVGG = VGG16(weights='imagenet')
        for i in range(len(self.vgg16Conv4.layers)):
            self.vgg16Conv4.get_layer(index=i).set_weights(
                originVGG.get_layer(index=i).get_weights())

        fc1Weights, fc1Biases = originVGG.get_layer(index=-3).get_weights()
        fc2Weights, fc2Biases = originVGG.get_layer(index=-2).get_weights()

        conv6Weights = np.random.choice(
            np.reshape(fc1Weights, (-1,)), (3, 3, 512, 1024))
        conv6Biases = np.random.choice(
            fc1Biases, (1024,))

        conv7Weights = np.random.choice(
            np.reshape(fc2Weights, (-1,)), (1, 1, 1024, 1024))
        conv7Biases = np.random.choice(
            fc2Biases, (1024,))

        self.vgg16Conv7.get_layer(index=2).set_weights(
            [conv6Weights, conv6Biases])
        self.vgg16Conv7.get_layer(index=3).set_weights(
            [conv7Weights, conv7Biases])

    def call(self, x):
        """
        The forward pass
        :param x: the input image
        :return: confs: list of outputs of all classification heads
                locs: list of outputs of all regression heads
        """
        confs = []
        locs = []
        head_idx = 0
        for i in range(len(self.vgg16Conv4.layers)):
            x = self.vgg16Conv4.get_layer(index=i)(x)
            if i == len(self.vgg16Conv4.layers) - 5:
                conf, loc = self.computeHeads(self.batchNorm(x), head_idx)
                confs.append(conf)
                locs.append(loc)
                head_idx += 1

        x = self.vgg16Conv7(x)

        conf, loc = self.computeHeads(x, head_idx)

        confs.append(conf)
        locs.append(loc)
        head_idx += 1

        for layer in self.extraLayers:
            x = layer(x)
            conf, loc = self.computeHeads(x, head_idx)
            confs.append(conf)
            locs.append(loc)
            head_idx += 1

        confs = tf.concat(confs, axis=1)
        locs = tf.concat(locs, axis=1)

        return confs, locs


def createSSD(numClasses, arch, pretrainedType,
              checkpointDir=None,
              checkpointPath=None):
    """
    Create SSD model and load pretrained weights
    :param numClasses:
    :param arch:
    :param pretrainedType:
    :param checkpointDir:
    :param checkpointPath:
    :return:
    """
    net = SSD(numClasses, arch)
    net(tf.random.normal((1, 300, 300, 3)))
    if pretrainedType == 'base':
        net.initVGG16()
        latestEpoch = 0
    elif pretrainedType == 'latest':
        try:
            paths = [os.path.join(checkpointDir, path)
                     for path in os.listdir(checkpointDir)]
            latest = sorted(paths, key=os.path.getmtime)[-1]
            name = latest.split('_')[2].split('.')
            latestEpoch = int(name[0])
            net.load_weights(latest)
        except AttributeError as e:
            print('Please make sure there is at least one checkpoint at {}'.format(
                checkpointDir))
            print('The model will be loaded from base weights.')
            net.initVGG16()
        except ValueError as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    latest, arch))
        except Exception as e:
            print(e)
            raise ValueError('Please check if checkpoint_dir is specified')
    elif pretrainedType == 'specified':
        if not os.path.isfile(checkpointPath):
            raise ValueError(
                'Not a valid checkpoint file: {}'.format(checkpointPath))

        try:
            net.load_weights(checkpointPath)
            latestEpoch = 0
        except Exception as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    checkpointPath, arch))
    else:
        raise ValueError('Unknown pretrained type: {}'.format(pretrainedType))
    return net, latestEpoch
