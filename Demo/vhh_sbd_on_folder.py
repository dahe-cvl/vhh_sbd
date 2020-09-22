from vhh_sbd.SBD import SBD
from vhh_sbd.utils import *
from vhh_sbd.Model import PyTorchModel
import os
import numpy as np

printCustom("Welcome to the sbd framework!", STDOUT_TYPE.INFO);
printCustom("Setup environment variables ... ", STDOUT_TYPE.INFO)
#print("------------------------------------------")
#print("LD_LIBRARY_PATH: ", str(os.environ['LD_LIBRARY_PATH']))
#print("CUDA_HOME: ", str(os.environ['CUDA_HOME']))
#print("PATH: ", str(os.environ['PATH']))
#print("CUDA_VISIBLE_DEVICES: ", str(os.environ['CUDA_VISIBLE_DEVICES']))
#print("PYTHONPATH: ", str(os.environ['PYTHONPATH']))
print("------------------------------------------")

#python Demo/vhh_sbd_on_folder.py /data/share/maxrecall_vhh_mmsi/develop/videos/downloaded/ /home/dhelm/VHH_Develop/installed_pkg/vhh_pkgs/vhh_sbd/config/config_vhh_test.yaml

printCustom("start process ... ", STDOUT_TYPE.INFO)

# read commandline arguments
params = getCommandLineParams()

# run shot boundary detection process
video_folder = params[1]
config_file = params[2]

file_list = os.listdir(video_folder)

for file in file_list:
    print(video_folder + file)

    video_filename = video_folder + file

    # initialize and run sbd process
    sbd_instance = SBD(config_file)
    sbd_instance.runOnSingleVideo(video_filename, max_recall_id=int(file.split('.')[0]))

printCustom("process finished!", STDOUT_TYPE.INFO)
