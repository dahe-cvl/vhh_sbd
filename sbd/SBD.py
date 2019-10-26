from sbd.Video import Video
from sbd.utils import *
from sbd.Configuration import Configuration
from sbd.Shot import Shot
from sbd.PreProcessing import PreProcessing
from sbd.Model import *
from sbd.Evaluation import Evaluation
#import os
from matplotlib import pyplot as plt
from scipy.spatial import distance


class SBD:
    def __init__(self, vid_instance: Video):
        #printCustom("INFO: create instance of sbd ... ", STDOUT_TYPE.INFO);

        if(vid_instance == None):
            printCustom("object of type Video is None!", STDOUT_TYPE.ERROR);
            exit();

        self.vid_instance = vid_instance;
        self.pre_proc_instance = PreProcessing();
        self.evaluation_instance = Evaluation();

        self.net = None;

        # parse configuration
        #cwd = os.getcwd()
        #print(cwd)

        config_file = "../config/config.yaml";
        self.config_instance = Configuration(config_file);
        self.config_instance.loadConfig();

    def calculateCosineSimilarity(self, x, y):
        dst = distance.cosine(x, y)
        return dst;

    def run(self):
        #printCustom("process shot detection ... ", STDOUT_TYPE.INFO);

        self.net = Squeezenet();
        # self.net = VGG19a();

        number_of_frames = int(self.vid_instance.number_of_frames);
        #print(number_of_frames)
        results_l = [];
        for i in range(1, number_of_frames):
            #print("-------------------")
            #print("process " + str(i))
            idx_curr = i;
            idx_prev = i-1;

            frm_prev = self.vid_instance.getFrame(idx_prev);
            frm_curr = self.vid_instance.getFrame(idx_curr);

            #print("preprocess images ... ")
            #dim = (int(self.vid_instance.width / 2), int(self.vid_instance.height / 2));
            #print(frm.shape)
            frm_trans_prev = self.pre_proc_instance.applyTransformOnImg(frm_prev)
            frm_trans_curr = self.pre_proc_instance.applyTransformOnImg(frm_curr)
            #print(frm_trans.shape)

            #print("process core part ... ")
            feature_prev = self.net.getFeatures(frm_trans_prev)
            feature_curr = self.net.getFeatures(frm_trans_curr)

            result = self.calculateCosineSimilarity(feature_prev, feature_curr)
            #print(result)
            results_l.append(result)

        results_np = np.array(results_l)
        #print(results_np.shape)

        # save raw results to file
        self.exportRawResultsAsCsv(results_np)

        # calculate similarity measures of consecutive frames and threshold it
        #shot_boundaries_np = self.calculateSimilarityMetric(results_np, threshold=0.8);

        #print("postprocess results ... ")

        # export shot boundaries as csv
        #self.exportResultsAsCsv(shot_boundaries_np);

        # convert shot boundaries to shots
        #shot_l = self.convertShotBoundaries2Shots(shot_boundaries_np);
        shot_l = [];
        #printCustom("successfully finished!", STDOUT_TYPE.INFO);
        return shot_l;

    def runOnRange(self, candidates_np):
        #printCustom("process shot detection ... ", STDOUT_TYPE.INFO);

        results_l = [];
        shot_l = []
        for i in range(0, len(candidates_np)):
            start = candidates_np[i][0]
            end = candidates_np[i][1] - 1

            #print(start)
            #print(end)
            results_per_range = []
            for j in range(start+1, end):
                # print("-------------------")
                # print("process " + str(i))
                idx_curr = j;
                idx_prev = j - 1;

                frm_prev = self.vid_instance.getFrame(idx_prev);
                frm_curr = self.vid_instance.getFrame(idx_curr);

                if(len(frm_prev) == 0 or len(frm_curr) == 0):
                    break;
                #print(idx_prev)
                #print(idx_curr)
                #print(frm_prev.shape)
                #print(frm_curr.shape)

                # print("preprocess images ... ")
                # dim = (int(self.vid_instance.width / 2), int(self.vid_instance.height / 2));
                # print(frm.shape)
                frm_trans_prev = self.pre_proc_instance.applyTransformOnImg(frm_prev)
                frm_trans_curr = self.pre_proc_instance.applyTransformOnImg(frm_curr)
                # print(frm_trans.shape)

                # print("process core part ... ")
                feature_prev = self.net.getFeatures(frm_trans_prev)
                feature_curr = self.net.getFeatures(frm_trans_curr)

                result = self.calculateCosineSimilarity(feature_prev, feature_curr)
                if(result > 0.8):
                    print(idx_prev)
                    print(idx_curr)
                    shot_l.append([self.vid_instance.vidName, (idx_prev, idx_curr)])

                results_per_range.append(result)

            results_l.append([start, end, results_per_range])

        results_np = np.array(results_l)
        #print(results_np)

        # save raw results to file
        self.exportRawResultsAsCsv_New(results_np)

        # calculate similarity measures of consecutive frames and threshold it
        #shot_boundaries_np = self.calculateSimilarityMetric(results_np, threshold=0.8);

        #print("postprocess results ... ")

        # export shot boundaries as csv
        #self.exportResultsAsCsv(shot_boundaries_np);

        # convert shot boundaries to shots
        #shot_l = self.convertShotBoundaries2Shots(shot_boundaries_np);
        shots_np = np.array(shot_l)
        #printCustom("successfully finished!", STDOUT_TYPE.INFO);
        return shots_np;

    def exportRawResultsAsCsv(self, results_np: np.ndarray):
        # save raw results to file
        fp = open(self.config_instance.path_raw_results +
                  self.config_instance.path_prefix_raw_results +
                  str(self.vid_instance.vidName.split('.')[0]) +
                  self.config_instance.path_postfix_raw_results, mode='w');

        for i in range(0, len(results_np)):
            # for j in range(0, len(results_np[0])):
            # print(sb_np[i])
            fp.write(self.vid_instance.vidName.split('.')[0] + ";" + str(results_np[i]) + "\n")
            # csv_writer.writerow(row);
        fp.close();

    def exportRawResultsAsCsv_New(self, results_np: np.ndarray):
        # save raw results to file
        fp = open(self.config_instance.path_raw_results +
                  self.config_instance.path_prefix_raw_results +
                  str(self.vid_instance.vidName.split('.')[0]) +
                  self.config_instance.path_postfix_raw_results, mode='w');

        for i in range(0, len(results_np)):
            start, end, distances_l = results_np[i]
            tmp_str = str(start) + ";" + str(end)
            for j in range(0, len(distances_l)):
                tmp_str = tmp_str + ";" + str(distances_l[j])
            fp.write(self.vid_instance.vidName.split('.')[0] + ";" + str(tmp_str) + "\n")
            # csv_writer.writerow(row);
        fp.close();

    def exportResultsAsCsv(self, shot_boundaries_np):
        # save final results to file
        fp = open(self.config_instance.path_final_results +
                  self.config_instance.path_prefix_final_results +
                  str(self.vid_instance.vidName.split('.')[0]) +
                  self.config_instance.path_postfix_final_results, mode='w');

        for i in range(0, len(shot_boundaries_np)):
            # print(sb_np[i])
            vidname = shot_boundaries_np[i][0];
            start = shot_boundaries_np[i][1];
            stop = shot_boundaries_np[i][2];

            tmp_str = str(vidname) + ";" + str(start) + ";" + str(stop);
            fp.write(tmp_str + "\n")
            # csv_writer.writerow(row);
        fp.close();



    def convertShotBoundaries2Shots(self, shot_boundaries_np: np.ndarray):
        # convert results to shot instances
        shot_l = [];
        vidname_curr = shot_boundaries_np[0][0];
        start_curr = shot_boundaries_np[0][1];
        shot_start = 0;
        shot_end = start_curr;
        shot = Shot(1, vidname_curr, shot_start, shot_end);
        shot_l.append(shot)

        for i in range(1, len(shot_boundaries_np)):
            stop_prev = shot_boundaries_np[i][2];
            vidname_curr = shot_boundaries_np[i][0];
            start_curr = shot_boundaries_np[i][1];
            shot_start = stop_prev;
            shot_end = start_curr;
            shot = Shot(i + 1, vidname_curr, shot_start, shot_end);
            shot_l.append(shot)

        vidname_curr = shot_boundaries_np[-1][0];
        stop_curr = shot_boundaries_np[-1][2];
        shot_start = stop_curr;
        shot_end = self.vid_instance.number_of_frames;
        shot = Shot(i + 1, vidname_curr, shot_start, shot_end);
        shot_l.append(shot)
        return shot_l;






