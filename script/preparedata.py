from utils.trafficdata import TraffDataset

if __name__ == '__main__':
    annfile = r'F:\RACUNAR\Master_data_Set\signDatabasePublicFramesOnly\allAnnotationsValidColumn.csv'
    createDir = r"D:\FAX\MASTER\data2\TData2"
    imgDir = r'F:\RACUNAR\Master_data_Set\signDatabasePublicFramesOnly'

    TraffDataset.prepareData(annfile,createDir,imgDir)