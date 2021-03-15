import tensorflow as tf
import os
import numpy as np
from PIL import Image
import re
import shutil
from functools import partial
import pickle
from .boxutils import computeTarget
import random
from .augmenteddata import *

class TraffDataset():
    """
    Generate and prepare data.
    """

    def __init__(self, rootDir, imgDir, defaultBoxes,
                 newSize, numExamples=-1, augmentation = False):
        """

        :param rootDir: Dataset root dir
        :param imgDir: Image root dir
        :param defaultBoxes: Created default boxes
        :param newSize: New image size
        :param numExamples:
        """
        self.dataDir = rootDir
        self.imgDir = imgDir
        self.defaultBoxes = defaultBoxes
        self.newSize = newSize
        self.numExamples = numExamples
        self.augmentation = augmentation
        self.augmentCounter = 0
        self.idxToName = self.__loadClassNameIds()
        self.data = self._loadData('data.p')
        self.nameToIdx = self._nameToIdx()
        self.augBoundry = int(len(self.data)*0.20)

    def _loadData(self, dName='prepData.p'):
        """
        Load created data
        :return: data
        """
        dataPath = os.path.join(self.dataDir, dName)
        with open(dataPath, 'rb') as f:
            dataLoad = pickle.load(f)

        keys = list(dataLoad.keys())
        random.shuffle(keys)

        shuffleData = {}
        for k in keys:
            shuffleData[k] = dataLoad[k]

        return shuffleData

    def __loadClassNameIds(self, clsIdsName='classToIds.p'):
        """
        Load class name - id dict
        @return:
        """
        dataPath = os.path.join(self.dataDir, clsIdsName)
        with open(dataPath, 'rb') as f:
            dataLoad = pickle.load(f)
        return dataLoad

    @classmethod
    def loadPData(self, dataPath):
        """
        Load created data
        :return: data
        """
        dataPath = os.path.join(dataPath, 'prepData.p')
        with open(dataPath, 'rb') as f:
            dataLoad = pickle.load(f)
        return dataLoad

    def _nameToIdx(self):
        """
        Class name to idx
        :return:
        """
        tmpNameToIdx = {}
        for k, v in self.idxToName.items():
            tmpNameToIdx[v] = k

        return tmpNameToIdx

    @classmethod
    def prepareData(cls, pathToAnnotationFile, pathToDirData, orgDirFile):
        """
        prepare data for training and store bbox in pickle
        :param pathToAnnotationFile: Annotation file
        :param pathToDirData: Pat to new image dir
        :param orgDirFile: path to data dir
        :return:
        """

        if not os.path.isdir(pathToDirData):
            os.mkdir(pathToDirData)
        with open(pathToAnnotationFile, 'r') as f:
            lines = [line for line in f]
            data = {}
            counterImg = 0
            for i, line in enumerate(lines[1:]):
                splitLine = re.split(";|/", line)
                imgPath = os.path.join(orgDirFile, splitLine[0], splitLine[1], splitLine[2])
                # newName = '{}_{}'.format(counterImg, splitLine[2])
                newImgName = os.path.join(pathToDirData, splitLine[2])

                try:
                    id = cls.idxToName[splitLine[3]]

                    if splitLine[2] in data:
                        tmpBbox = [float(splitLine[4]), float(splitLine[5]), float(splitLine[6]), float(splitLine[7])]
                        data[splitLine[2]]['bbox'].append(tmpBbox)
                        data[splitLine[2]]['clsName'].append(splitLine[3])
                        data[splitLine[2]]['clsId'].append(id)
                    else:
                        tmpBbox = [float(splitLine[4]), float(splitLine[5]), float(splitLine[6]), float(splitLine[7])]
                        tmpDict = {'clsName': [splitLine[3]], 'clsId': [id],
                                   'bbox': [tmpBbox]}  # 'imgName': newImgName,
                        data[splitLine[2]] = tmpDict
                        try:
                            shutil.copy(imgPath, newImgName)
                        except Exception as e:
                            print('Can not copy ima')
                            continue

                    counterImg += 1

                    with open('prepData.p', 'wb') as f:
                        pickle.dump(data, f)

                except Exception as e:
                    print('Dont have this label:{}'.format(e))
                    continue

    def _get_image(self, filename):
        """
        Method to read image from file
        then resize to (300, 300)
        """
        img_path = os.path.join(self.dataDir, self.imgDir, filename)
        img = Image.open(img_path)

        return img

    def _prepBoxCord(self, boxCord, w, h):
        """
        Scale box cordinate in range 0-1
        :param boxCord: boc cord
        :param w: image width
        :param h: image height
        :return:
        """
        xmin = (float(boxCord[0]) - 1) / w
        ymin = (float(boxCord[1]) - 1) / h
        xmax = (float(boxCord[2]) - 1) / w
        ymax = (float(boxCord[3]) - 1) / h

        return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

    def generate(self, mode='train'):
        """
        Generate data
        :param mode: train or test
        :return:
        """
        numOfData = len(self.data.keys())

        if mode == 'train':
            start = 0
            numOfind = round(numOfData * 0.95)
            print('Number of train data: {}'.format(numOfind - start))

        if mode == 'val':
            start = round(numOfData * 0.90)
            numOfind = numOfData
            print('Number of val data: {}'.format(numOfind - start))

        for i in range(start, numOfData):

            key = list(self.data.keys())[i]

            imgData = self.data[key]
            img = self._get_image(key)
            w, h = img.size

            # Get box and labels
            # scale box corrd
            boxCord = imgData["bbox"]
            labels = []  # np.array(imgData['clsId'], dtype=np.int64)
            boxes = []
            for n, b in enumerate(boxCord):
                boxes.append(self._prepBoxCord(b, w, h))
                labels.append(imgData['clsId'][n])
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

            boxes = tf.constant(boxes, dtype=tf.float32)
            labels = tf.constant(labels, dtype=tf.int64)

            if self.augmentation:
                flag = random.choice([True, False])
                if flag and self.augmentCounter < self.augBoundry:
                    img, boxes, labels = horizontalFlip(img, boxes, labels)
                    self.augmentCounter += 1
                    if self.augmentCounter > self.augBoundry:
                        self.augmentation = False

            img = np.array(img.resize((self.newSize, self.newSize)), dtype=np.float32)
            img = (img / 127.0) - 1.0
            img = tf.constant(img, dtype=tf.float32)

            gt_confs, gt_locs = computeTarget(self.defaultBoxes, boxes, labels)

            #print(i)
            yield key, img, gt_confs, gt_locs  # yield

    def generateExportModel(self, mode='train'):
        """
        Generate data
        :param mode: train or test
        :return:
        """
        numOfData = len(self.data.keys())

        if mode == 'train':
            start = 0
            numOfind = round(numOfData * 0.95)
            print('Number of train data: {}'.format(numOfind - start))

        if mode == 'val':
            start = round(numOfData * 0.90)
            numOfind = numOfData
            print('Number of val data: {}'.format(numOfind - start))

        for i in range(start, numOfData):

            key = list(self.data.keys())[i]

            imgData = self.data[key]
            img = self._get_image(key)
            w, h = img.size

            # Get box and labels
            # scale box corrd
            boxCord = imgData["bbox"]
            labels = []  # np.array(imgData['clsId'], dtype=np.int64)
            boxes = []
            for n, b in enumerate(boxCord):
                boxes.append(self._prepBoxCord(b, w, h))
                labels.append(imgData['clsId'][n])
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

            boxes = tf.constant(boxes, dtype=tf.float32)
            labels = tf.constant(labels, dtype=tf.int64)

            if self.augmentation:
                flag = random.choice([True, False])
                if flag and self.augmentCounter < self.augBoundry:
                    img, boxes, labels = horizontalFlip(img, boxes, labels)
                    self.augmentCounter += 1
                    if self.augmentCounter > self.augBoundry:
                        self.augmentation = False

            img = np.array(img.resize((self.newSize, self.newSize)), dtype=np.float32)
            img = (img / 127.0) - 1.0
            #img = tf.constant(img, dtype=tf.float32)

            gt_confs, gt_locs = computeTarget(self.defaultBoxes, boxes, labels)

            #print(i)
            return key, img, gt_confs, gt_locs  # yield


def createTraffBatchGenerator(
        rootDir,
        imgDir,
        defaultBoxes,
        newSize,
        batchSize,
        numBatches,
        augmentation,
        mode):
    """
    Batch generator
    :param rootDir:
    :param imgDir:
    :param defaultBoxes:
    :param newSize:
    :param batchSize:
    :param numBatches:
    :param mode:
    :return:
    """

    num_examples = batchSize * numBatches if numBatches > 0 else -1
    traff = TraffDataset(rootDir, imgDir, defaultBoxes, newSize, num_examples, augmentation)

    # traff.generate(mode="train")

    info = {
        'idx_to_name': traff.nameToIdx,
        'length': round(len(traff.data) * 0.85),
        'image_dir': imgDir
    }

    if mode == "train":
        train_gen = partial(traff.generate, mode="train")
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, (tf.string, tf.float32, tf.int64, tf.float32)
        )
        val_gen = partial(traff.generate, mode="val")
        val_dataset = tf.data.Dataset.from_generator(
            val_gen, (tf.string, tf.float32, tf.int64, tf.float32)
        )

        train_dataset = train_dataset.shuffle(40).batch(batchSize)
        val_dataset = val_dataset.batch(batchSize)

        return train_dataset.take(numBatches), val_dataset.take(-1), info

    else:
        dataset = tf.data.Dataset.from_generator(
            traff.generate, (tf.string, tf.float32, tf.int64, tf.float32))
        dataset = dataset.batch(batchSize)
        return dataset.take(numBatches), info
