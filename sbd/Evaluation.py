import numpy as np
import csv
from time import gmtime, strftime
import json
from scipy.spatial import distance
from sbd.Video import Video
import matplotlib.pyplot as plt
from inspect import signature
from sbd.Configuration import Configuration
import os


class Evaluation:
    def __init__(self, config_file: str):
        print("create instance of evaluation ...")
        self.config_instance = Configuration(config_file);
        self.config_instance.loadConfig();

        self.precision = 0;
        self.recall = 0;
        self.f1score = 0;


    def exportResultsToCSV(self, res_np):
        print("start csv export ...");

        timestamp = str(strftime("%Y-%m-%d_%H%M%S", gmtime()));
        fp = open(self.results_path + "/" + "results_"+ str(timestamp) +".csv", mode='w');
        #csv_writer = csv.writer(fp, delimiter=';');

        for i in range(0, len(res_np)):
            #print(sb_np[i])
            vidname = res_np[i][0];
            sb_np = res_np[i][1];
            tmp_str = str(vidname) + ";";
            for j in range(0, len(sb_np)):
                tmp_str = tmp_str + str(int(sb_np[j])) + ";"
            fp.write(tmp_str + "\n")
            #csv_writer.writerow(row);

        fp.close();

    def exportMovieResultsToCSV(self, fName, res_np):
        print("start csv export ...");
        fName = fName.split('.')[0]
        print(fName)

        fp = open(str(fName) + "_movie_based.csv", mode='w');

        tmp_str = "vidname;precision;recall;f1score";
        fp.write(tmp_str + "\n")
        for i in range(0, len(res_np)):
            #print(sb_np[i])
            vidname = res_np[i][0];
            p = res_np[i][1];
            r = res_np[i][2];
            f1 = res_np[i][3];
            tmp_str = vidname + ";" + str(p) + ";" + str(r) + ";" + str(f1)
            fp.write(tmp_str + "\n")
            #csv_writer.writerow(row);

        fp.close();

    def loadFromJson(self, filepath):
        print("load data from json file ...");
        fp = open(filepath, mode='r');
        res_json = json.load(fp);

        #print(res_json.items())
        sbd_list = [];
        for videoname, labels in res_json.items():
            #print("-----------------")
            #print(videoname)

            _gts = res_json[videoname]['cut'];
            # print(_gts)

            for start, end in _gts:
                if (abs(start - end) > 1):
                    sbd_list.append([videoname, start, end, 'gradual']);
                if (abs(start - end) == 1):
                    sbd_list.append([videoname, start, end, 'abrupt']);

                #print("videoname: " + str(videoname))
                #print(start)
                #print(end)
                #print("--------")

        sbd_np = np.array(sbd_list)

        idx = np.where(sbd_np[:, 3:4] == 'abrupt')[0];
        p_abrupt_np = sbd_np[idx];

        idx = np.where(sbd_np[:, 3:4] == 'gradual')[0];
        p_gradual_np = sbd_np[idx];

        # print(self.gt_cuts_np)
        # print(self.gt_graduals_np)
        return p_abrupt_np, p_gradual_np

    def loadFromJsonDeepSBD(self, filepath):
        print("load data from json file ...");
        fp = open(filepath, mode='r');
        res_json = json.load(fp);

        #print(res_json.items())
        sbd_cut_list = [];
        sbd_gradual_list = [];
        for videoname, labels in res_json.items():
            #print("-----------------")
            #print(videoname)

            _p_cut = res_json[videoname]['cut'];
            _p_gradual = res_json[videoname]['gradual'];
            #print(_p_cut)
            #print(_p_gradual)

            for start, end in _p_cut:
                sbd_cut_list.append([videoname, start, end, 'cut']);


            for start, end in _p_gradual:
                sbd_gradual_list.append([videoname, start, end, 'gradual']);

                #print("videoname: " + str(videoname))
                #print(start)
                #print(end)
                #print("--------")

        sbd_cut_np = np.array(sbd_cut_list)
        sbd_gradual_np = np.array(sbd_gradual_list)


        # print(self.gt_cuts_np)
        # print(self.gt_graduals_np)
        return sbd_cut_np, sbd_gradual_np

    def loadResultsFromCSV(self, filepath):
        print("load data from csv file ...");
        fp = open(filepath, mode='r');
        csv_reader = csv.reader(fp, delimiter=';');
        res_l = [];

        for line in csv_reader:
            res_l.append(line);
        res_np = np.array(res_l)
        #print(res_np)

        fp.close();
        return res_np;

    def calcPRF(self, tp, fp, fn, tn):
        print("calculate precision, recall and f1 score ...");

        # TP, TN, FP, FN

        self.precision = float(tp) / (float(tp) + float(fp))
        self.recall = tp / (tp + fn)

        if(self.precision > 0 and self.recall > 0):
            self.f1score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1score = 0.0;

        print("precision: " + str(round(self.precision, 3)));
        print("recall: " + str(round(self.recall, 3)));
        print("f1score: " + str(round(self.f1score, 3)));

    def loadRawResultsAsCsv(self, filepath):
        # save raw results to file
        fp = open(filepath, mode='r');
        lines = fp.readlines();
        fp.close();
        # print(lines)

        dist_l = [];
        for i in range(0, len(lines)):
            line = lines[i].replace('\n', '')
            line = line.replace('[', '')
            line = line.replace(']', '')
            line = line.replace('\'', '')
            line = line.replace(',', '.')
            line = line.replace(' ', '')
            line_split = line.split(';')
            # print(line_split)
            vidname = line_split[0];
            distance = float(line_split[1]);
            dist_l.append([vidname, distance]);

        dist_np = np.array(dist_l)
        #print(dist_np.shape)
        return dist_np

    def loadRawResultsAsFromNumpy(self, filepath):
        # save raw results to file
        print("load raw results from numpy ...")
        vid_name = filepath.split('/')[-1].split('.')[0];
        raw_results = np.load(filepath, allow_pickle=True)

        start = raw_results[0][0]
        stop = raw_results[0][1]
        dist_l = raw_results[0][2]
        dist_np = np.array(dist_l).astype('float')

        array_list = []
        array_list.append([vid_name.replace("results_raw_", ""), start, stop, dist_np])
        array_np = np.array(array_list)
        return array_np;

    def loadRawResultsAsCsv_New(self, filepath):
        # save raw results to file
        fp = open(filepath, mode='r');
        lines = fp.readlines();
        fp.close();
        #print(len(lines))

        final_l = [];
        for i in range(0, len(lines)):
            line = lines[i].replace('\n', '')
            line = line.replace('[', '')
            line = line.replace(']', '')
            line = line.replace(')', '')
            line = line.replace('(', '')
            #line = line.replace('list', '')
            line = line.replace('\'', '')
            line = line.replace(',', '.')
            #line = line.replace(' ', '')
            line_split = line.split(';')
            vidname = line_split[0];
            start = int(line_split[1]);
            end = int(line_split[2]);
            dist_l = line_split[3:];
            dist_np = np.array(dist_l).astype('float')
            final_l.append([vidname, start, end, dist_np]);

        final_np = np.array(final_l)
        return final_np

    def calculateSimilarityMetric(self, results_np: np.ndarray, threshold=0.8):
        '''
        vid_name = results_np[0][0];
        start = results_np[i][1]
        distances_np = results_np[:, 1:2].astype('float');

        idx_max = np.where(distances_np > threshold)[0]
        shot_boundaries_l = []
        for i in range(0, len(idx_max)):
            shot_boundaries_l.append([vid_name, idx_max[i], idx_max[i] + 1])
            #cv2.imwrite("./test_result" + str(i) + "_1.png", self.vid_instance.getFrame(idx_max[i]))
            #cv2.imwrite("./test_result" + str(i) + "_2.png", self.vid_instance.getFrame(idx_max[i] + 1))
        shot_boundaries_np = np.array(shot_boundaries_l)
        #print(shot_boundaries_np.shape)
        #print(shot_boundaries_np)
        '''

        shot_boundaries_l = []
        for i in range(0, len(results_np)):
            vid_name = results_np[i][0]
            start = results_np[i][1]
            end = results_np[i][2]
            distances_l = results_np[i][3]
            distances_np = np.array(distances_l).astype('float');

            if(self.config_instance.activate_candidate_selection == 0):
                # just take all frame positions over specified threshold
                idx_max = np.where(distances_np > threshold)[0]
                print(idx_max)

                if(len(idx_max) == 1) :
                    final_idx = idx_max #+ start
                    shot_boundaries_l.append([vid_name, final_idx, final_idx + 1])
                elif(len(idx_max) > 1):
                    final_idx = idx_max #+ start
                    print(final_idx)
                    #exit()
                    for a in range(0, len(final_idx)):
                        shot_boundaries_l.append([vid_name, final_idx[a], final_idx[a] + 1])
            elif(self.config_instance.activate_candidate_selection == 1):
                # just take all frame positions over specified threshold
                idx_max = np.argmax(distances_np)
                final_idx = idx_max + start
                shot_boundaries_l.append([vid_name, final_idx, final_idx + 1])

            # print(final_idx)

            # cv2.imwrite("./test_result" + str(i) + "_1.png", self.vid_instance.getFrame(idx_max[i]))
            # cv2.imwrite("./test_result" + str(i) + "_2.png", self.vid_instance.getFrame(idx_max[i] + 1))

        shot_boundaries_np = np.array(shot_boundaries_l)
        #print(shot_boundaries_np)
        return shot_boundaries_np;


    def calculateSimilarityMetric_new(self, results_np: np.ndarray, threshold=0.8):
        shot_boundaries_l = []
        for i in range(0, len(results_np)):
            vid_name = results_np[i][0]
            start = results_np[i][1]
            end = results_np[i][2]
            distances_l = results_np[i][3]
            distances_np = np.array(distances_l).astype('float');

            active = False;
            if(active == True):
                # just take the maximum of each range
                min_val = np.min(distances_np);
                max_val = np.max(distances_np);
                max_real = 1.0;
                min_real = 0.0;
                distances_scaled = ((max_real - min_real) / (max_val - min_val)) * (distances_np - min_val) + min_real;
                distances_np = distances_scaled

                idx_max = np.argmax(distances_np) + start

                if (distances_np[np.argmax(distances_np)] > threshold):
                    final_idx = idx_max
                    shot_boundaries_l.append([vid_name, final_idx, final_idx + 1])
            else:
                # just take all frame positions over specified threshold
                idx_max = np.argmax(distances_np) + start

                if(distances_np[np.argmax(distances_np)] > threshold):
                    final_idx = idx_max
                    shot_boundaries_l.append([vid_name, final_idx, final_idx + 1])
                    #print(final_idx)

            # cv2.imwrite("./test_result" + str(i) + "_1.png", self.vid_instance.getFrame(idx_max[i]))
            # cv2.imwrite("./test_result" + str(i) + "_2.png", self.vid_instance.getFrame(idx_max[i] + 1))

        shot_boundaries_np = np.array(shot_boundaries_l)
        return shot_boundaries_np;

    def evaluation(self, result_np, vid_name):
        #print("NOT IMPLEMENTED YET");

        src_path = self.config_instance.path_videos;
        gt_data = self.config_instance.path_gt_data;

        if (self.config_instance.path_postfix_raw_results == 'csv'):
            vid_name = vid_name.replace('results_raw_', '')
            vid_name = vid_name.replace('.csv', '')
        elif (self.config_instance.path_postfix_raw_results == 'npy'):
            vid_name = vid_name.replace('results_raw_', '')
            vid_name = vid_name.replace('.npy', '')

        #print(vid_name)
        #vid_name = result_np[0][0];
        video_obj = Video();
        video_obj.load(src_path + "/" + str(vid_name) + ".mp4");

        # load gt
        fp = open(gt_data, 'r');
        lines = fp.readlines();
        fp.close();
        #print(lines)

        gt_l = [];
        for i in range(1, len(lines)):
            line = lines[i].replace('\n', '')
            line = line.replace('[', '')
            line = line.replace(']', '')
            line = line.replace('\'', '')
            line = line.replace(',', ';')
            line = line.replace(' ', '')
            line_split = line.split(';')
            #print(line_split)
            idx = int(line_split[0]);
            vidname = line_split[1];
            start = int(line_split[2])
            stop = int(line_split[3])
            diff = int(line_split[4])
            sbd_type = line_split[5]

            if(sbd_type == "HARDCUT"):
                gt_l.append([vidname, (start, stop)]);

        gt_np = np.array(gt_l)
        #print(gt_np)

        '''
        # load pred
        fp = open(pred_data, 'r');
        lines = fp.readlines();
        fp.close();
        # print(lines)
        '''
        #print("AAA")
        #print(result_np)
        pred_l = [];
        for i in range(0, len(result_np)):
            vidname = result_np[i][0];
            start = int(result_np[i][1])
            stop = int(result_np[i][2])
            pred_l.append([vidname, (start, stop)]);

        pred_np = np.array(pred_l)
        #print(pred_np)
        print(vid_name)
        idx = np.where(vid_name == pred_np)[0]
        sb_pred_np = pred_np[idx];
        #print("---------")
        #print(sb_pred_np)

        idx = np.where(vid_name == gt_np)[0]
        sb_gt_np = gt_np[idx];
        #print("---------")
        #print(sb_gt_np)

        #exit()

        # video-based predictions
        tp_cnt = 0;
        fp_cnt = 0;
        tn_cnt = 0;
        fn_cnt = 0;
        for j in range(0, int(video_obj.number_of_frames)):
            curr_pos = j
            prev_pos = j - 1;

            search_tuple = (prev_pos, curr_pos);
            if(len(sb_pred_np) > 0 ):
                list_pred_tmp = np.squeeze(sb_pred_np[:, 1:]).tolist();
            else:
                list_pred_tmp = []
            list_gt_tmp = np.squeeze(sb_gt_np[:, 1:]).tolist();

            gt_flag = False;
            pred_flag = False;

            try:
                # found tuple in pred
                res_pred = list_pred_tmp.index(search_tuple)
                #print(list_pred_tmp.index(search_tuple))
                pred_flag = True;
            except:
                res_pred = 0;

            try:
                # found tuple in gt
                res_gt = list_gt_tmp.index(search_tuple)
                #print(list_gt_tmp.index(search_tuple))
                gt_flag = True;
            except:
                res_gt = 0;

            #print(str(gt_flag) + " == " + str(pred_flag))
            tp_cond = gt_flag and pred_flag; # find tuple in pred && find tuple in gt --> true
            fn_cond = gt_flag and not pred_flag; # not find tuple in pred && find tuple in gt --> true
            fp_cond = not gt_flag and pred_flag;
            tn_cond = not gt_flag and not pred_flag;

            if(tp_cond == True):
                tp_cnt = tp_cnt + 1;
            if (fp_cond == True):
                fp_cnt = fp_cnt + 1;
            if (tn_cond == True):
                tn_cnt = tn_cnt + 1;
            if (fn_cond == True):
                fn_cnt = fn_cnt + 1;

        #precision, recall, accuracy, f1_score = self.calculateEvalMetrics(tp_cnt, fp_cnt, tn_cnt, fn_cnt);
        #tmp_str = str(vid_name) + ";" + str(tp_cnt) + ";" + str(fp_cnt) + ";" + str(tn_cnt) + ";" + str(
        #    fn_cnt) + ";" + str(precision) + ";" + str(recall) + ";" + str(accuracy) + ";" + str(f1_score)
        #print(tmp_str)
        '''
        print("---------------------------")
        print("video-based results")
        print("TP: " + str(tp_cnt))
        print("FP: " + str(fp_cnt))
        print("TN: " + str(tn_cnt))
        print("FN: " + str(fn_cnt))
        
        print("precision: " + str(precision))
        print("recall: " + str(recall))
        print("accuracy: " + str(accuracy))
        print("f1_score: " + str(f1_score))
        '''

        return tp_cnt, fp_cnt, tn_cnt, fn_cnt;

    def calculateMetrics(self, tp_cnt, fp_cnt, tn_cnt, fn_cnt):
        # calculate precision, recall,  accuracy
        if(tp_cnt + fp_cnt != 0):
            precision = tp_cnt / (tp_cnt + fp_cnt);
        else:
            precision = 0;

        if (tp_cnt + fn_cnt != 0):
            recall = tp_cnt / (tp_cnt + fn_cnt);
        else:
            recall = 0;

        if ((tp_cnt + tn_cnt + fp_cnt + fn_cnt) != 0):
            accuracy = (tp_cnt + tn_cnt) / (tp_cnt + tn_cnt + fp_cnt + fn_cnt);
        else:
            accuracy = 0;

        if ((precision + recall) != 0):
            f1_score = 2 * (precision * recall) / (precision + recall);
        else:
            f1_score = 0;

        if ((tp_cnt + fn_cnt) != 0):
            tp_rate = tp_cnt / (tp_cnt + fn_cnt);
        else:
            tp_rate = 0;

        if ((tn_cnt + fp_cnt) != 0):
            #Specificity = True Negatives / (True Negatives + False Positives)
            fp_rate = 1 - (tn_cnt / (tn_cnt + fp_cnt));
        else:
            fp_rate = 0;

        return precision, recall, accuracy, f1_score, tp_rate, fp_rate;

    def export2CSV(self, data_np: np.ndarray, header: str, filename: str, path: str):
        # save to csv file
        fp = open(path + "/" + str(filename) + ".csv", 'w');
        fp.write(header)
        for i in range(0, len(data_np)):
            tmp_str = data_np[i][0];
            for j in range(1, len(data_np[0])):
                tmp_str = tmp_str + ";" + data_np[i][j]
                # print(str(i) + "/" + str(j) + " - " + tmp_str)
            fp.write(tmp_str + "\n")
        fp.close();

    def calculateEvaluationMetrics(self, src_path: str, vid_name_list: list, prefix="results_raw_"):
        tp_sum = 0;
        fp_sum = 0;
        tn_sum = 0;
        fn_sum = 0;

        #vid_name_list = ['EF-NS_095_OeFM']
        final_results = []
        #thresholds_l = [0.95, 0.90, 0.85, 0.8, 0.75, 0.70, 0.65, 0.60]
        #for t in thresholds_l:
        for s in range(0, 1000):
            THRESHOLD = s * 0.001;
            #THRESHOLD = t;
            #print("step: " + str(s))
            results_l = [];
            for vid_name in vid_name_list:
                results_np = self.loadRawResultsAsCsv(str(src_path) + "/" + prefix + str(vid_name) + ".csv")

                # calculate similarity measures of consecutive frames and threshold it
                shot_boundaries_np1 = self.calculateSimilarityMetric(results_np, threshold=THRESHOLD);
                if (len(shot_boundaries_np1) != 0):
                    tp, fp, tn, fn = self.evaluation(shot_boundaries_np1);
                    p, r, acc, f1_score, tp_rate, fp_rate = self.calculateEvalMetrics(tp, fp, tn, fn);
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

            p, r, acc, f1_score, tp_rate, fp_rate = self.calculateEvalMetrics(tp_sum, fp_sum, tn_sum, fn_sum);

            final_results.append([str(THRESHOLD), p, r, acc, f1_score, tp_rate, fp_rate])

            results_l.append(["overall", tp_sum, fp_sum, tn_sum, fn_sum, p, r, acc, f1_score])
            results_np = np.array(results_l);

        final_results_np = np.array(final_results);
        return final_results_np;


    def calculateEvaluationMetrics_New(self):
        if (self.config_instance.path_postfix_raw_results == 'csv'):
            vid_name_list = os.listdir(str(self.config_instance.path_raw_results_eval))
            vid_name_list = [i for i in vid_name_list if i.endswith('.csv')]
        elif (self.config_instance.path_postfix_raw_results == 'npy'):
            vid_name_list = os.listdir(str(self.config_instance.path_raw_results_eval))
            vid_name_list = [i for i in vid_name_list if i.endswith('.npy')]
        print(vid_name_list)

        final_results = []
        fp_video_based = None;
        thresholds_l = [1.0, 0.90, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55,
                        0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.0]
        #thresholds_l = [0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.92,0.91,0.90,0.89, 0.88, 0.87, 0.86,  0.85, 0.84, 0.83, 0.82,0.80,  0.8]
        #thresholds_l = [0.81]
        for t in thresholds_l:
        #for s in range(0, 1000):
            tp_sum = 0;
            fp_sum = 0;
            tn_sum = 0;
            fn_sum = 0;
            #THRESHOLD = s * 0.001;
            THRESHOLD = t;
            #print("step: " + str(s))

            if(int(self.config_instance.save_eval_results) == 1):
                fp_video_based = open(self.config_instance.path_eval_results + "/final_results_th-" + str(THRESHOLD) + ".csv", 'w');
                header = "vid_name;tp;fp;tn;fn;p;r;acc;f1_score;tp_rate;fp_rate";
                fp_video_based.write(header + "\n")

            results_l = [];
            for vid_name in vid_name_list:
                if(self.config_instance.path_postfix_raw_results == 'csv'):
                    results_np = self.loadRawResultsAsCsv_New(self.config_instance.path_raw_results_eval + "/" + vid_name)
                elif (self.config_instance.path_postfix_raw_results == 'npy'):
                    results_np = self.loadRawResultsAsFromNumpy(self.config_instance.path_raw_results_eval + "/" + vid_name)

                # calculate similarity measures of consecutive frames and threshold it
                shot_boundaries_np1 = self.calculateSimilarityMetric(results_np, threshold=THRESHOLD);
                print(shot_boundaries_np1)
                #continue

                #if (len(shot_boundaries_np1) != 0):
                tp, fp, tn, fn = self.evaluation(shot_boundaries_np1, vid_name);
                p, r, acc, f1_score, tp_rate, fp_rate = self.calculateMetrics(tp, fp, tn, fn);

                if (int(self.config_instance.save_eval_results) == 1):
                    tmp_str = str(vid_name.replace('results_raw_', '').split('.')[0]) + ";" + str(tp) + ";" + str(fp) + \
                              ";" + str(tn) + ";" + str(fn) + ";" + str(p) + ";" + str(r) + ";" + str(acc) + ";" + \
                              str(f1_score) + ";" + str(tp_rate) + ";" + str(fp_rate);
                    print(tmp_str);
                    fp_video_based.write(tmp_str + "\n")
                #else:
                #    tp = 0;
                #    fp = 0;
                #    tn = 0;
                #    fn = 0;
                #    p = 0;
                #    r = 0;
                #    acc = 0;
                #    f1_score = 0;
                results_l.append([vid_name, tp, fp, tn, fn, p, r, acc, f1_score])

                tp_sum = tp_sum + tp;
                fp_sum = fp_sum + fp;
                tn_sum = tn_sum + tn;
                fn_sum = fn_sum + fn;

            p, r, acc, f1_score, tp_rate, fp_rate = self.calculateMetrics(tp_sum, fp_sum, tn_sum, fn_sum);

            if (int(self.config_instance.save_eval_results) == 1):
                tmp_str = str("overall" + ";" + str(tp_sum) + ";" + str(fp_sum) + \
                          ";" + str(tn_sum) + ";" + str(fn_sum) + ";" + str(p) + ";" + str(r) + ";" + str(acc) + ";" + \
                          str(f1_score) + ";" + str(tp_rate) + ";" + str(fp_rate));
                print(tmp_str);

                fp_video_based.write(tmp_str + "\n");
                fp_video_based.close();

            final_results.append([str(THRESHOLD), tp_sum, fp_sum, tn_sum, fn_sum, p, r, acc, f1_score, tp_rate, fp_rate])
        final_results_np = np.array(final_results);
        return final_results_np;

    def plotPRCurve(self, results_np):
        print("plot precision recall curve ... ")
        precision = np.squeeze(results_np[1:, 5:6].astype('float')).tolist()
        recall = np.squeeze(results_np[1:, 6:7].astype('float')).tolist()

        plt.figure()
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title("2-class Precision-Recall curve");
        plt.savefig(self.config_instance.path_eval_results + "/pr_curve.png")

    def plotROCCurve(self, results_np):
        print("plot roc curve ... ")
        tp_rate = np.squeeze(results_np[1:, 8:9].astype('float')).tolist()
        fp_rate = np.squeeze(results_np[1:, 9:10].astype('float')).tolist()

        plt.figure()
        plt.plot(fp_rate, tp_rate, color='orange', label='ROC')
        #plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.savefig(self.config_instance.path_eval_results + "/roc_curve.png")

    def run(self):
        print("evaluation ... ");

        final_results_np = self.calculateEvaluationMetrics_New();
        print(final_results_np)
        self.plotPRCurve(final_results_np);
        self.plotROCCurve(final_results_np);

        '''
        # export results to csv file
        self.export2CSV(final_results_np,
                                     "threshold;p;r;acc;f1_score" + "\n",
                                     "final_evaluation_pr_curve",
                                     "../Develop/");
        '''