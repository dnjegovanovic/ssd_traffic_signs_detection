import argparse
import os
import numpy as np
from utils.trafficdata import TraffDataset
e = {}
evalD =  {'addedLane': 0,
         'slow': 0,
         'curveLeft': 0,
         'speedLimit15': 0,
         'curveRight': 0,
         'speedLimit25': 0,
         'dip': 0,
         'speedLimit30': 0,
         'doNotEnter': 0,
         'speedLimit35': 0,
         'doNotPass': 0,
         'speedLimit40': 0,
         'intersection': 0,
         'speedLimit45': 0,
         'keepRight': 0,
         'speedLimit50': 0,
         'laneEnds': 0,
         'speedLimit55': 0,
         'merge': 0,
         'speedLimit65': 0,
         'noLeftTurn': 0,
         'speedLimitUrdbl': 0,
         'noRightTurn': 0,
         'stop': 0,
         'pedestrianCrossing': 0,
         'stopAhead': 0,
         'rampSpeedAdvisory20': 0,
         'thruMergeLeft': 0,
         'rampSpeedAdvisory35': 0,
         'thruMergeRight': 0,
         'rampSpeedAdvisory40': 0,
         'thruTrafficMergeLeft': 0,
         'rampSpeedAdvisory45': 0,
         'truckSpeedLimit55': 0,
         'rampSpeedAdvisory50': 0,
         'turnLeft': 0,
         'rampSpeedAdvisoryUrdbl': 0,
         'turnRight': 0,
         'rightLaneMustTurn': 0,
         'yield': 0,
         'roundabout': 0,
         'yieldAhead': 0,
         'school': 0,
         'zoneAhead25': 0,
         'schoolSpeedLimit25': 0,
         'zoneAhead45': 0,
         'signalAhead': 0,
         'bg': 0,
         'mAP': []}


def calcScore(data, detPath, clsName, iouTres=0.5, use07metric=False):
    detPath = detPath.format(clsName)
    with open(detPath, 'r') as f:
        lines = f.readlines()

    lines = [x.strip().split(' ') for x in lines]
    imageIds = [x[0] for x in lines]
    confs = np.array([float(x[1]) for x in lines])
    boxes = np.array([[float(z) for z in x[2:]] for x in lines])

    gts = {}
    clsGts = {}
    npos = 0

    for imgId in imageIds:
        gts[imgId] = data[imgId]
        gtBbox = gts[imgId]["bbox"]
        clsGts[imgId] = {'gtBoxes': gtBbox}

    sortedIds = np.argsort(-confs)
    sortedScores = np.sort(-confs)
    boxes = boxes[sortedIds, :]
    imageIds = [imageIds[x] for x in sortedIds]

    dataNumber = len(imageIds)
    tp = np.zeros(dataNumber)
    fp = np.zeros(dataNumber)

    for i in range(dataNumber):
        R = clsGts[imageIds[i]]
        dataGTClass = data[imageIds[i]]['clsName']
        box = boxes[i, :].astype(float)
        iouMax = -np.inf
        gtBox = np.array(R['gtBoxes'])  # .astype(float)

        if gtBox.size > 0:
            ixmin = np.maximum(gtBox[0], box[0])
            ixmax = np.maximum(gtBox[2], box[2])
            iymin = np.maximum(gtBox[1], box[1])
            iymax = np.maximum(gtBox[3], box[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            uni = ((box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0) +
                   (gtBox[2] - gtBox[0] + 1.0) *
                   (gtBox[3] - gtBox[1] + 1.0) - inters)

            ious = inters / uni
            iouMax = np.max(ious)
            jmax = np.argmax(ious)

        if iouMax > iouTres:
            pass


def evaluate():
    dataPath = r'D:\FAX\MASTER\data'
    data = TraffDataset.loadPData(dataPath)

    detectData = r'../outputsMergeData/detects'
    for clsName in evalD.keys():
        detPath = os.path.join(detectData, '{}.txt')

        if os.path.exists(detPath.format(clsName)):
            recall, precision, ap = calcScore(data, detPath, clsName)

    print(dataPath)


if __name__ == '__main__':
    evaluate()
