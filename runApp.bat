set CONFIG="D:\FAX\MASTER\repo\ssd_traffic_signs_detection\config.yml"
set CHECKPOINTPATH="D:\FAX\MASTER\repo\ssd_traffic_signs_detection\checkpointsNewData\ssd_epoch_70.h5"
"D:\envs\tf2.2gpu\Scripts\python.exe" "D:\FAX\MASTER\repo\ssd_traffic_signs_detection\objectdetectionapp.py" -config=%CONFIG% -checkpointPath=%CHECKPOINTPATH%
