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

    def evaluation(self, video_obj: Video):
        print("NOT IMPLEMENTED YET");

        vid_name = video_obj.vidName.split('.')[0];

        gt_data = "/caa/Projects02/vhh/private/database_nobackup/VHH_datasets/annotations/ShotBoundaryDetection/efilms/shotBoundaries_annotations_v8_20190523.csv";
        pred_data = "/caa/Homes01/dhelm/working/pycharm_vhh_sbd/Demo/results_" + vid_name + ".csv";

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

        # load pred
        fp = open(pred_data, 'r');
        lines = fp.readlines();
        fp.close();
        # print(lines)

        pred_l = [];
        for i in range(0, len(lines)):
            line = lines[i].replace('\n', '')
            line = line.replace('[', '')
            line = line.replace(']', '')
            line = line.replace('\'', '')
            line = line.replace(',', ';')
            line = line.replace(' ', '')
            line_split = line.split(';')
            #print(line_split)

            vidname = line_split[0];
            start = int(line_split[1])
            stop = int(line_split[2])

            pred_l.append([vidname, (start, stop)]);

        pred_np = np.array(pred_l)
        #print(pred_np)


        idx = np.where(vid_name == pred_np)[0]
        sb_pred_np = pred_np[idx];
        #print(sb_pred_np)

        idx = np.where(vid_name == gt_np)[0]
        sb_gt_np = gt_np[idx];
        #print(sb_gt_np)

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


        print(tp_cnt)
        print(fp_cnt)
        print(tn_cnt)
        print(fn_cnt)

        # calculate precision, recall,  accuracy
        precision = tp_cnt / (tp_cnt + fp_cnt);
        recall = tp_cnt / (tp_cnt + fn_cnt);
        accuracy = (tp_cnt + tn_cnt) / (tp_cnt + tn_cnt + fp_cnt + fn_cnt);
        print(precision)
        print(recall)
        print(accuracy)