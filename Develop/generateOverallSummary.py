import numpy as np
import os

print("generate overall summary file ...")
path = "/caa/Projects02/vhh/private/Results/ShotBoundaryDetection/20191212_vhh_sbd/squeezenet/Exp09_none/imc/Evaluation/"
#path = "/caa/Homes01/dhelm/working/pycharm_vhh_sbd/Develop/Evaluation/"

file_list = [f for f in os.listdir(path) if f.endswith('.csv') and f.startswith("final_")]
file_list.sort()
print(file_list)

final_results = []
for file in file_list:
    # read csv file
    fp = open(path + "/" + file, 'r')
    lines = fp.readlines()
    fp.close()

    last_line_np = lines[-1].replace('\n', '').split(';')
    #print(last_line_np)
    final_results.append([last_line_np[0], last_line_np[1], last_line_np[2], last_line_np[3], last_line_np[4],
                          last_line_np[5], last_line_np[6], last_line_np[7], last_line_np[8], last_line_np[9],
                          last_line_np[10]])

final_results_np = np.array(final_results)
#print(final_results_np.shape)


print("write to csv file ... ")
fp = open(path + "/overall_summary.csv", 'w')
for i in range(0, len(final_results_np)):
    entry_str = final_results_np[i][0] + ";" + final_results_np[i][1] + ";" + final_results_np[i][2] + ";" + \
                final_results_np[i][3] + ";" + final_results_np[i][4] + ";" + final_results_np[i][5] + ";" + \
                final_results_np[i][6] + ";" + final_results_np[i][7] + ";" + final_results_np[i][8] + ";" + \
                final_results_np[i][9] + ";" + final_results_np[i][10]
    fp.write(entry_str + '\n')
fp.close()

print("finished!")
