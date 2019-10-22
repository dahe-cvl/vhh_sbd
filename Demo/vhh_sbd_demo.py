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
sbd_instance = SBD(vid_instance);

src_path = "/caa/Projects02/vhh/private/dzafirova/sbd_efilms_db_20190621/videos_converted/";
filename_l = os.listdir("/caa/Projects02/vhh/private/dzafirova/sbd_efilms_db_20190621/videos_converted/")
print(filename_l)


print("start")
vidname_list = filename_l;
vidname_list = ['EF-NS_026_OeFM.mp4'];
for vidname in vidname_list:
    printCustom("--------------------------", STDOUT_TYPE.INFO)
    printCustom("Process video: " + str(vidname) + " ... ", STDOUT_TYPE.INFO)
    vid_instance.load(src_path + "/" + vidname);
    #vid_path = "/caa/Projects02/vhh/private/dzafirova/sbd_efilms_db_20190621/videos_converted/EF-NS_009_OeFM.mp4";
    #vid_path = "/caa/Projects02/vhh/private/dzafirova/sbd_efilms_db_20190621/videos_converted/EF-NS_026_OeFM.mp4";
    #vid_instance.load(vid_path);

    shot_list = sbd_instance.run();
    print(len(shot_list))