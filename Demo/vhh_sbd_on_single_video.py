from sbd.SBD import SBD
from sbd.utils import *
from sbd.Model import PyTorchModel
import os
import numpy as np

printCustom("Welcome to the sbd framework!", STDOUT_TYPE.INFO);
printCustom("Setup environment variables ... ", STDOUT_TYPE.INFO)
print("------------------------------------------")
print("LD_LIBRARY_PATH: ", str(os.environ['LD_LIBRARY_PATH']))
print("CUDA_HOME: ", str(os.environ['CUDA_HOME']))
print("PATH: ", str(os.environ['PATH']))
print("CUDA_VISIBLE_DEVICES: ", str(os.environ['CUDA_VISIBLE_DEVICES']))
print("PYTHONPATH: ", str(os.environ['PYTHONPATH']))
print("------------------------------------------")


printCustom("start process ... ", STDOUT_TYPE.INFO)

# read commandline arguments
params = getCommandLineParams();

# run shot boundary detection process
video_filename = params[1];
config_file = params[2];

# initialize and run sbd process
sbd_instance = SBD(config_file);
sbd_instance.runOnSingleVideo(video_filename);

printCustom("process finished!", STDOUT_TYPE.INFO)
