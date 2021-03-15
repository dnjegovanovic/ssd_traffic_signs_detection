import numpy as np
from PIL import Image, ImageDraw
import pickle
import json
import os

def loadJSONfile(path, typeData='train'):
    """

    @param path:
    @param typeData:
    @return:
    """
    if typeData == 'train':
        with open(os.path.join(path, 'train.json')) as f:
            trainData = json.load(f)
        return trainData
    elif typeData == 'test':

        with open(os.path.join(path, 'test.json')) as f:
            testData = json.load(f)
        return testData


def findeImgAnnById(imgId, ann):
    """

    @param imgId:
    @param ann:
    @return:
    """
    matchedAnn = []
    for a in ann:
        if a['image_id'] == imgId:
            matchedAnn.append(a)

    return matchedAnn


def matchCategAnn(category, matchedAnnid):
    """

    @param category:
    @param matchedAnnid:
    @return:
    """
    catName = []
    for anID in matchedAnnid:
        for c in category:
            if c['id'] == anID:
                catName.append(c['name'])

    return catName


def exportIdxTiNameClass(imgCategories):
    """
    exportIdxNameClass
    @param imgCategories:
    @return:
    """
    tmpCID = {'addedLane': 200,
             'slow': 201,
             'curveLeft': 202,
             'speedLimit15': 203,
             'curveRight': 204,
             'speedLimit25': 205,
             'dip': 206,
             'speedLimit30': 207,
             'doNotEnter': 208,
             'speedLimit35': 209,
             'doNotPass': 210,
             'speedLimit40': 211,
             'intersection': 212,
             'speedLimit45': 213,
             'keepRight': 214,
             'speedLimit50': 215,
             'laneEnds': 216,
             'speedLimit55': 217,
             'merge': 218,
             'speedLimit65': 219,
             'noLeftTurn': 220,
             'speedLimitUrdbl': 221,
             'noRightTurn': 222,
             'stop': 223,
             'pedestrianCrossing': 224,
             'stopAhead': 225,
             'rampSpeedAdvisory20': 226,
             'thruMergeLeft': 227,
             'rampSpeedAdvisory35': 228,
             'thruMergeRight': 229,
             'rampSpeedAdvisory40': 230,
             'thruTrafficMergeLeft': 231,
             'rampSpeedAdvisory45': 232,
             'truckSpeedLimit55': 233,
             'rampSpeedAdvisory50': 234,
             'turnLeft': 235,
             'rampSpeedAdvisoryUrdbl': 236,
             'turnRight': 237,
             'rightLaneMustTurn': 238,
             'yield': 239,
             'roundabout': 240,
             'yieldAhead': 241,
             'school': 242,
             'zoneAhead25': 243,
             'schoolSpeedLimit25': 244,
             'zoneAhead45': 245,
             'signalAhead': 246,
             'bg': 247}

    for c in imgCategories:
        # print("c: {},{}".format(c['name'], id))
        tmpCID[c['name']] = c['id']

    with open('classToIds.p', 'wb') as f:
        pickle.dump(tmpCID, f)


def loadImgAndParsData(imgPath, jsonData):
    """

    @param imgPath:
    @param jsonData:
    @return:
    """
    imagesDic = jsonData['images']
    imagesAnn = jsonData['annotations']
    imagesCat = jsonData['categories']

    exportIdxTiNameClass(imagesCat)

    data = {}
    counter = 0
    for imgd in imagesDic:
        imgNameTmp = imgd['file_name']
        imgTmpID = imgd['id']

        matchIMgmp = findeImgAnnById(imgTmpID, imagesAnn)  # pronadji annotation za sliku
        imgsBbox = [bb['bbox'] for bb in matchIMgmp]
        if len(imgsBbox) == 0:
            continue
        clasId = [cl['category_id'] for cl in matchIMgmp]
        matchAnnId = [a['id'] for a in matchIMgmp]  # za matchin sa kategorijom
        clasName = matchCategAnn(imagesCat, matchAnnId)

        tmpBox = []
        for b in imgsBbox:
            s = [float(b[0]), float(b[1]), float(b[0]+b[2]), float(b[1]+b[3])]
            tmpBox.append(s)

        tmpDict = {'clsName': clasName, 'clsId': clasId, 'bbox': tmpBox}
        data[imgNameTmp] = tmpDict
        # Load image and draw
        # imgtmp = Image.open(os.path.join(imgPath,imgNameTmp))
        # imgD = ImageDraw.Draw(imgtmp)
        # for b in imgsBbox:
        #     shape = [(b[0],b[1]), (b[0]+b[2],b[1]+b[3])]
        #     imgD.rectangle(shape, outline='red', width=2)
        # imgtmp.save("test.png")
        counter += 1
        print(counter)

    with open('prepDataDFGTrain.p', 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    path = r'D:\FAX\MASTER\DFG-tsd-annot-json'
    imgPath = r'D:\FAX\MASTER\JPEGImages\JPEGImages'
    labels = loadJSONfile(path, 'train')

    loadImgAndParsData(imgPath, labels)


    
    # with open(r"D:\FAX\MASTER\JPEGImages\prepDataDFGTrain.p", 'rb') as f:
    #     dfg = pickle.load(f)
    #
    # with open(r"D:\FAX\MASTER\data2\prepData.p", 'rb') as f:
    #     lisa = pickle.load(f)
    #
    # for k,v in lisa.items():
    #     for i in range(len(lisa[k]['clsId'])):
    #         lisa[k]['clsId'][i] = lisa[k]['clsId'][i]+200
    #
    # dfg.update(lisa)
    #
    # with open('data.p', 'wb') as f:
    #     pickle.dump(dfg, f)
        

    print('Finish')
