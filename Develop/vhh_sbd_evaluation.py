from sbd.Evaluation import Evaluation
from sbd.SBD import SBD
from sbd.utils import *
from sbd.Video import Video
import numpy as np

printCustom("Welcome to the sbd evauation framework!", STDOUT_TYPE.INFO);

tp_sum = 0;
fp_sum = 0;
tn_sum = 0;
fn_sum = 0;

sbd_eval_instance = Evaluation();

# load raw results
#vid_name = "EF-NS_004_OeFM"; 'EF-NS_026_OeFM', 'EF-NS_004_OeFM',
#vid_name_list = ['EF-NS_004_OeFM', 'EF-NS_016_OeFM', 'EF-NS_060_OeFM_R03-01',
#                 'EF-NS_009_OeFM', 'EF-NS_032_OeFM', 'EF-NS_083_USHMM',
#                 'EF-NS_013_OeFM', 'EF-NS_043_USHMM', 'EF-NS_095_OeFM']

# calculate precision recall curve
vid_name_list = ['EF-NS_095_OeFM']
final_results_np = sbd_eval_instance.calculatePrecisionRecallCurve(src_path="../Demo", vid_name_list=vid_name_list, prefix="results_raw_");

# export results to csv file
sbd_eval_instance.export2CSV(final_results_np,
                             "threshold;p;r;acc;f1_score" + "\n",
                             "final_evaluation_pr_curve",
                             "../Develop/");


'''
THRESHOLD = 0.85;
results_l = [];
for vid_name in vid_name_list:

    results_np = sbd_eval_instance.loadRawResultsAsCsv("../Demo/results_raw_" + str(vid_name) + ".csv")

    # calculate similarity measures of consecutive frames and threshold it
    shot_boundaries_np1 = sbd_eval_instance.calculateSimilarityMetric(results_np, threshold=THRESHOLD);
    if(len(shot_boundaries_np1) != 0):
        tp, fp, tn, fn = sbd_eval_instance.evaluation(shot_boundaries_np1);
        p, r, acc, f1_score = sbd_eval_instance.calculateEvalMetrics(tp, fp, tn, fn);
    else:
        tp = 0;
        fp = 0;
        tn = 0;
        fn = 0;
        p = 0;
        r = 0;
        acc = 0;
        f1_score = 0;
    results_l.append([vid_name, tp, fp, tn, fn, p, r, acc, f1_score])

    tp_sum = tp_sum + tp;
    fp_sum = fp_sum + fp;
    tn_sum = tn_sum + tn;
    fn_sum = fn_sum + fn;


p, r, acc, f1_score = sbd_eval_instance.calculateEvalMetrics(tp_sum, fp_sum, tn_sum, fn_sum);
results_l.append(["overall", tp_sum, fp_sum, tn_sum, fn_sum, p, r, acc, f1_score])
results_np = np.array(results_l);



print("---------------------------")
print("overall results - " + str(THRESHOLD))
print("TP: " + str(tp_sum))
print("FP: " + str(fp_sum))
print("TN: " + str(tn_sum))
print("FN: " + str(fn_sum))
print("precision: " + str(p))
print("recall: " + str(r))
print("accuracy: " + str(acc))
print("f1_score: " + str(f1_score))

## save to csv file
filename = "final_evaluation_results_" + str(THRESHOLD);
fp = open("../Develop/" + str(filename) + ".csv", 'w');
fp.write("vidname;tp;fp;tn;fn;p;r;acc;f1_score" + "\n")
for i in range(0, len(results_np)):
    tmp_str = results_np[i][0];
    for j in range(1, len(results_np[0])):
        tmp_str = tmp_str + ";" + results_np[i][j]
        #print(str(i) + "/" + str(j) + " - " + tmp_str)
    fp.write(tmp_str + "\n")
fp.close()

#shot_boundaries_np2 = sbd_eval_instance.calculateSimilarityMetric(results_np, threshold=0.65);
#shot_boundaries_np3 = sbd_eval_instance.calculateSimilarityMetric(results_np, threshold=0.75);

#sbd_eval_instance.evaluation(shot_boundaries_np2)
#sbd_eval_instance.evaluation(shot_boundaries_np3)


#print("postprocess results ... ")
# export shot boundaries as csv
#sbd_instance.exportResultsAsCsv(shot_boundaries_np);
# convert shot boundaries to shots
#shot_l = sbd_instance.convertShotBoundaries2Shots(shot_boundaries_np);
'''