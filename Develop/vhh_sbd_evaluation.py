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
vid_name_list = ['EF-NS_004_OeFM', 'EF-NS_013_OeFM']
for vid_name in vid_name_list:

    results_np = sbd_eval_instance.loadRawResultsAsCsv("../Demo/results_raw_" + str(vid_name) + ".csv")

    # calculate similarity measures of consecutive frames and threshold it
    shot_boundaries_np1 = sbd_eval_instance.calculateSimilarityMetric(results_np, threshold=0.55);
    tp, fp, tn, fn = sbd_eval_instance.evaluation(shot_boundaries_np1);

    tp_sum = tp_sum + tp;
    fp_sum = fp_sum + fp;
    tn_sum = tn_sum + tn;
    fn_sum = fn_sum + fn;

p, r, acc, f1_score = sbd_eval_instance.calculateEvalMetrics(tp_sum, fp_sum, tn_sum, fn_sum)

print("---------------------------")
print("overall results")
print("TP: " + str(tp_sum))
print("FP: " + str(fp_sum))
print("TN: " + str(tn_sum))
print("FN: " + str(fn_sum))
print("precision: " + str(p))
print("recall: " + str(r))
print("accuracy: " + str(acc))
print("f1_score: " + str(f1_score))


#shot_boundaries_np2 = sbd_eval_instance.calculateSimilarityMetric(results_np, threshold=0.65);
#shot_boundaries_np3 = sbd_eval_instance.calculateSimilarityMetric(results_np, threshold=0.75);

#sbd_eval_instance.evaluation(shot_boundaries_np2)
#sbd_eval_instance.evaluation(shot_boundaries_np3)


#print("postprocess results ... ")
# export shot boundaries as csv
#sbd_instance.exportResultsAsCsv(shot_boundaries_np);
# convert shot boundaries to shots
#shot_l = sbd_instance.convertShotBoundaries2Shots(shot_boundaries_np);