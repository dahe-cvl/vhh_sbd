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
config_file = "../config/config.yaml"
sbd_instance = SBD(config_file);

# run shot boundary detection process
sbd_instance.run();
