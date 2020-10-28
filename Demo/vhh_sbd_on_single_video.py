from vhh_sbd.SBD import SBD
import os

# run shot boundary detection process
config_file = "/caa/Homes01/dhelm/working/vhh/develop/vhh_sbd/config/config_vhh_test.yaml"
sbd_instance = SBD(config_file)

videos_path = "/data/share/maxrecall_vhh_mmsi/release/videos/downloaded/"
videos_path_list = os.listdir(videos_path)
print(videos_path_list)

for file in videos_path_list[:1]:
    print(videos_path + file)

    max_recall_id = int(file.split('.')[0])
    sbd_instance.runOnSingleVideo(videos_path + file, max_recall_id=max_recall_id)
