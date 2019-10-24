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
#vidname_list = filename_l;

vidname_list = ['EF-NS_032_OeFM.mp4',
                'EF-NS_083_USHMM.mp4', 'EF-NS_052_USHMM.mp4', 'EF-NS_067_OeFM.mp4', 'EF-NS_020_OeFM.mp4',
                'EF-NS_049_USHMM.mp4', 'EF-NS_029_OeFM_R02-01.mp4', 'EF-NS_018_OeFM.mp4', 'EF-NS_041_USHMM.mp4',
                'EF-NS_035_OeFM.mp4', 'EF-NS_098_OeFM_R02-01.mp4', 'EF-NS_098_OeFM_R02-02.mp4', 'EF-NS_030_OeFM.mp4',
                'EF-NS_027_OeFM.mp4', 'EF-NS_072_OeFM.mp4', 'EF-NS_026_OeFM.mp4', 'EF-NS_028_OeFM.mp4', 'EF-NS_034_OeFM.mp4',
                'EF-NS_065_OeFM.mp4', 'EF-NS_033_OeFM.mp4', 'EF-NS_031_OeFM.mp4', 'EF-NS_001_OeFM.mp4',
                'EF-NS_096_NCJF.mp4', 'EF-NS_060_OeFM_R03-02.mp4', 'EF-NS_019_OeFM.mp4', 'EF-NS_025_OeFM_1.mp4',
                'EF-NS_092_OeFM.mp4', 'EF-NS_068_OeFM.mp4', 'EF-NS_094_NARA.mp4', 'EF-NS_023_OeFM.mp4',
                'EF-NS_057_OeFM.mp4', 'EF-NS_059_OeFM.mp4', 'EF-NS_017_OeFM.mp4', 'EF-NS_011_OeFM.mp4',
                'EF-NS_060_OeFM_R03-03.mp4', 'EF-NS_010_OeFM.mp4', 'EF-NS_090_OeFM.mp4', 'EF-NS_064_OeFM.mp4',
                'EF-NS_097_OeFM_R01-01.mp4', 'EF-NS_022_OeFM.mp4', 'EF-NS_063_OeFM.mp4', 'EF-NS_093_USHMM.mp4',
                'EF-NS_021_OeFM.mp4', 'EF-NS_077_OeFM.mp4', 'EF-NS_069_OeFM.mp4', 'EF-NS_079_OeFM.mp4',
                'EF-NS_076_OeFM.mp4', 'EF-NS_071_OeFM.mp4', 'EF-NS_085_USHMM.mp4', 'EF-NS_008_OeFM.mp4',
                'EF-NS_053_USHMM.mp4', 'EF-NS_100_OeFM.mp4', 'EF-NS_029_OeFM_R02-02.mp4', 'EF-NS_091_OeFM.mp4',
                'EF-NS_066_OeFM.mp4', 'EF-NS_061_OeFM.mp4', 'EF-NS_005_OeFM.mp4', 'EF-NS_099_OeFM.mp4',
                'EF-NS_084_USHMM.mp4']


for vidname in vidname_list:
    printCustom("--------------------------", STDOUT_TYPE.INFO)
    printCustom("Process video: " + str(vidname) + " ... ", STDOUT_TYPE.INFO)
    vid_instance.load(src_path + "/" + vidname);
    #vid_path = "/caa/Projects02/vhh/private/dzafirova/sbd_efilms_db_20190621/videos_converted/EF-NS_009_OeFM.mp4";
    #vid_path = "/caa/Projects02/vhh/private/dzafirova/sbd_efilms_db_20190621/videos_converted/EF-NS_026_OeFM.mp4";
    #vid_instance.load(vid_path);

    shot_list = sbd_instance.run();
    print(len(shot_list))