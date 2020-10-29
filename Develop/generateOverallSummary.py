import numpy as np
import os

print("generate overall summary file ...")
path = "/data/share/maxrecall_vhh_mmsi/develop/videos/results/sbd/develop/sbd_deana/sbd_resnet_3f_deana_fixed/"

experiment_name = path.split('/')[-2]
net = path.split('/')[-3]

print(experiment_name)
print(net)

file_list = [f for f in os.listdir(path) if f.endswith('.csv') and f.startswith("final_")]
file_list.sort()
print(file_list)

final_results = []
for file in file_list:
    th = float(file.split('-')[-1][:-4])

    # read csv file
    fp = open(path + "/" + file, 'r')
    lines = fp.readlines()
    fp.close()

    last_line_np = lines[-1].replace('\n', '').split(';')
    #print(last_line_np)
    final_results.append([th, last_line_np[1], last_line_np[2], last_line_np[3], last_line_np[4],
                          last_line_np[5], last_line_np[6], last_line_np[7], last_line_np[8], last_line_np[9],
                          last_line_np[10]])

final_results_np = np.array(final_results)
#print(final_results_np.shape)

print("write to csv file ... ")
fp = open(path + "/summary_" + str(net) + "_" + str(experiment_name) + ".csv", 'w')
# write header
fp.write("th_" + str(net) + "_" + str(experiment_name) + 
         ";tp_" + str(net) + "_" + str(experiment_name) + 
         ";fp_" + str(net) + "_" + str(experiment_name) + 
         ";tn_" + str(net) + "_" + str(experiment_name) + 
         ";fn_" + str(net) + "_" + str(experiment_name) + 
         ";p_" + str(net) + "_" + str(experiment_name) + 
         ";r_" + str(net) + "_" + str(experiment_name) + 
         ";acc_" + str(net) + "_" + str(experiment_name) + 
         ";f1_score_" + str(net) + "_" + str(experiment_name) + 
         ";tp_rate_" + str(net) + "_" + str(experiment_name) + 
         ";fp_rate_" + str(net) + "_" + str(experiment_name) + "\n")

for i in range(0, len(final_results_np)):
    entry_str = final_results_np[i][0].replace('.', ',') + ";" + final_results_np[i][1] + ";" + final_results_np[i][2] + ";" + \
                final_results_np[i][3] + ";" + final_results_np[i][4] + ";" + final_results_np[i][5].replace('.', ',') + ";" + \
                final_results_np[i][6].replace('.', ',') + ";" + final_results_np[i][7].replace('.', ',') + ";" + final_results_np[i][8].replace('.', ',') + ";" + \
                final_results_np[i][9].replace('.', ',') + ";" + final_results_np[i][10].replace('.', ',')
    fp.write(entry_str + '\n')
fp.close()

print("finished!")
