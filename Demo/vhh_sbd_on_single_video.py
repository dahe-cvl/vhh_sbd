from vhh_sbd.SBD import SBD
import os

# run shot boundary detection process
config_file = "./config/config_vhh_test.yaml"
sbd_instance = SBD(config_file)

videos_path = sbd_instance.config_instance.path_videos
videos_path_list = os.listdir(videos_path)
videos_path_list.sort()

for file in videos_path_list:
    print(videos_path + file)
    max_recall_id = int(file.split('.')[0])
    sbd_instance.runOnSingleVideo(videos_path + file, max_recall_id=max_recall_id)
