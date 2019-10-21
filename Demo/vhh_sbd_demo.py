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

vid_instance = Video();
#vid_instance.load("test1.mpg");
vid_path = "/caa/Projects02/vhh/private/dzafirova/sbd_efilms_db_20190621/videos_converted/EF-NS_009_OeFM.mp4";
vid_instance.load(vid_path);
sbd_instance = SBD(vid_instance);
shot_list = sbd_instance.run();
print(len(shot_list))