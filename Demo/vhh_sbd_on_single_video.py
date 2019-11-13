from sbd.SBD import SBD
from sbd.utils import *
from sbd.Video import Video

import os

printCustom("Welcome to the sbd framework!", STDOUT_TYPE.INFO);

printCustom("Setup environment variables ... ", STDOUT_TYPE.INFO)
print("------------------------------------------")
print("LD_LIBRARY_PATH: ", str(os.environ['LD_LIBRARY_PATH']))
print("CUDA_HOME: ", str(os.environ['CUDA_HOME']))
print("PATH: ", str(os.environ['PATH']))
print("CUDA_VISIBLE_DEVICES: ", str(os.environ['CUDA_VISIBLE_DEVICES']))
print("PYTHONPATH: ", str(os.environ['PYTHONPATH']))
print("------------------------------------------")


print("start")
sbd_instance = SBD();

# run shot boundary detection process
video_filename = "/caa/Projects02/vhh/private/database_nobackup/IMediaCities_datasets/IMC_database/videos/OFM_WStLA_Budget-Film-Rettungswesen-und-Krankenhaus_H264-1440x1080-High-CAVLC-12Mbps-KFIAuto-mp-Stereo48-256Kbps_24fps_128G.mp4"
sbd_instance.runOnSingleVideo(video_filename);
