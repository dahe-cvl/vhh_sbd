import numpy as np
import csv
from time import gmtime, strftime
import json
from scipy.spatial import distance
from sbd.Video import Video


class Evaluation:
    def __init__(self):
        print("create instance of ...")

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

    def calculateSimilarityMetric(self, results_np: np.ndarray, threshold=0.8):
        vid_name = results_np[0][0];
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

        return shot_boundaries_np;

    def evaluation(self, result_np):
        #print("NOT IMPLEMENTED YET");

        src_path = "/caa/Projects02/vhh/private/dzafirova/sbd_efilms_db_20190621/videos_converted/";
        gt_data = "/caa/Projects02/vhh/private/database_nobackup/VHH_datasets/annotations/ShotBoundaryDetection/efilms/shotBoundaries_annotations_v8_20190523.csv";

        vid_name = result_np[0][0];
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

        idx = np.where(vid_name == pred_np)[0]
        sb_pred_np = pred_np[idx];


        idx = np.where(vid_name == gt_np)[0]
        sb_gt_np = gt_np[idx];



        # video-based predictions
        tp_cnt = 0;
        fp_cnt = 0;
        tn_cnt = 0;
        fn_cnt = 0;
        for j in range(0, int(video_obj.number_of_frames)):
            curr_pos = j
            prev_pos = j - 1;

            search_tuple = (prev_pos, curr_pos);
            list_pred_tmp = np.squeeze(sb_pred_np[:, 1:]).tolist();
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

    def calculateEvalMetrics(self, tp_cnt, fp_cnt, tn_cnt, fn_cnt):
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

        return precision, recall, accuracy, f1_score;

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

    def calculatePrecisionRecallCurve(self, src_path: str, vid_name_list: list, prefix="results_raw_"):
        tp_sum = 0;
        fp_sum = 0;
        tn_sum = 0;
        fn_sum = 0;

        #vid_name_list = ['EF-NS_095_OeFM']
        final_results = []
        # thresholds_l = [0.95, 0.90, 0.85, 0.8, 0.75]
        # for t in thresholds_l:
        for s in range(350, 950):
            THRESHOLD = s * 0.001;
            # THRESHOLD = t;
            print("step: " + str(s))
            results_l = [];
            for vid_name in vid_name_list:
                results_np = self.loadRawResultsAsCsv(str(src_path) + "/" + prefix + str(vid_name) + ".csv")

                # calculate similarity measures of consecutive frames and threshold it
                shot_boundaries_np1 = self.calculateSimilarityMetric(results_np, threshold=THRESHOLD);
                if (len(shot_boundaries_np1) != 0):
                    tp, fp, tn, fn = self.evaluation(shot_boundaries_np1);
                    p, r, acc, f1_score = self.calculateEvalMetrics(tp, fp, tn, fn);
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

            p, r, acc, f1_score = self.calculateEvalMetrics(tp_sum, fp_sum, tn_sum, fn_sum);

            final_results.append([str(THRESHOLD), p, r, acc, f1_score])

            results_l.append(["overall", tp_sum, fp_sum, tn_sum, fn_sum, p, r, acc, f1_score])
            results_np = np.array(results_l);

        final_results_np = np.array(final_results);
        return final_results_np;