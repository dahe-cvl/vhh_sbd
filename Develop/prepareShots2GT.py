import numpy as np

print("convert annoatation list to usable gt data ...")

# read csv file
fp = open("imediacities_shot_annotations_20190805_prepared.csv", 'r')
lines = fp.readlines()
fp.close()

print("prepare annotation file ... ")

lines_np = []
for i in range(1, len(lines)):
    tmp = lines[i].replace('\r', '')
    tmp = tmp.replace('\n', '')
    tmp_split = tmp.split(';')
    #print(tmp_split)
    lines_np.append(tmp_split)
lines_np = np.array(lines_np)
shot_info_np = lines_np[:, 1:4];
vid_names_np = np.unique(shot_info_np[:, :1]);

#print(vid_names_np)

print("process ... ")

sbd_entries = []
for vid_name in vid_names_np:
    #print(vid_name)

    idx = np.where(vid_name == shot_info_np)[0]
    #print(idx)
    shot_info_per_video = shot_info_np[idx]
    #print(shot_info_per_video)
    #print(len(shot_info_per_video))
    if (len(shot_info_per_video) > 1):
        for i in range(1, len(shot_info_per_video)):
            sbd_start = shot_info_per_video[i-1][2]
            sbd_end = shot_info_per_video[i][1]

            #print(sbd_start)
            #print(sbd_end)

            diff = abs(int(sbd_start) - int(sbd_end))
            if(diff == 1):
                #print("-----------")
                sbd_entries.append([str(i), str(vid_name), str(sbd_start), str(sbd_end), str(diff), "HARDCUT"])
                #print("(" + str(i) + ") " + str(vid_name) + ": " + str(sbd_start) + " -> " + str(sbd_end))
            else:
                sbd_entries.append([str(i), str(vid_name), str(sbd_start), str(sbd_end), str(diff), "GRADUAL"])

sbd_entries_np = np.array(sbd_entries)


print("write to csv file ... ")
fp = open("imc_sbd.csv", 'w')
for i in range(0, len(sbd_entries_np)):
    entry_str = sbd_entries_np[i][0] + ";" + sbd_entries_np[i][1] + ";" + sbd_entries_np[i][2] + ";" + sbd_entries_np[i][3] + ";" + sbd_entries_np[i][4] + ";" + sbd_entries_np[i][5]
    fp.write(entry_str + '\n')
fp.close()

print("finished!")