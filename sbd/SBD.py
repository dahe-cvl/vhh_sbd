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
from sbd.DeepSBD import CandidateSelection
import os

class SBD:
    def __init__(self, config_file: str):
        #printCustom("INFO: create instance of sbd ... ", STDOUT_TYPE.INFO);

        # parse configuration
        #cwd = os.getcwd()
        #print(cwd)

        if (config_file == ""):
            printCustom("No configuration file specified!", STDOUT_TYPE.ERROR)
            exit()

        #config_file = "../config/config.yaml";
        self.config_instance = Configuration(config_file);
        self.config_instance.loadConfig();

        self.vid_instance = None;
        self.pre_proc_instance = PreProcessing(self.config_instance);
        self.evaluation_instance = Evaluation(self.config_instance);
        self.candidate_selection_instance = CandidateSelection(self.config_instance)
        self.net = None;

        self.src_path = "";
        self.filename_l = "";

    def calculateDistance(self, x, y):
        dst = 0;

        # initial pre-trained model
        if (self.config_instance.similarity_metric == "cosine"):
            dst = distance.cosine(x, y)
        elif (self.config_instance.similarity_metric == "euclidean"):
            dst = distance.euclidean(x, y)
        else:
            dst = None;
            printCustom("No valid similarity metric selected!", STDOUT_TYPE.ERROR)
            exit();

        return dst;

    def run(self):
        self.src_path = self.config_instance.path_videos;
        self.filename_l = os.listdir(self.src_path)
        vid_name_l = self.filename_l
        #vid_name_l = ['OFM_Entree-du-Cinematographe-a-Vienne_H264-1440x1080-High-CAVLC-12Mbps-KFIAuto-mp-AudioNone_16fps_0001-03-0505.mp4']
        print(vid_name_l)
        shot_boundaries_l = []
        for vid_name in vid_name_l:
            printCustom("--------------------------", STDOUT_TYPE.INFO)
            printCustom("Process video: " + str(vid_name) + " ... ", STDOUT_TYPE.INFO)

            # load video
            self.vid_instance = Video();
            self.vid_instance.load(self.src_path + "/" + vid_name);

            # candidate selection
            printCustom("Process candidate selection: " + str(vid_name) + " ... ", STDOUT_TYPE.INFO)
            candidate_selection_result_np = self.candidate_selection_instance.run(self.src_path + "/" + vid_name)

            # shot boundary detection
            printCustom("Process shot boundary detection: " + str(vid_name) + " ... ", STDOUT_TYPE.INFO)
            shots_np = self.runWithCandidateSelection(candidate_selection_result_np)
            #print(shots_np)
            shot_boundaries_l.append(shots_np);
        shot_boundaries_np = np.squeeze(np.array(shot_boundaries_l));

        # convert shot boundaries to final shots
        print(shot_boundaries_np.shape)
        final_shot_l = self.convertShotBoundaries2Shots(shot_boundaries_np);

        # export final results
        if (self.config_instance.save_final_results == 1):
            self.exportResultsToCsv(final_shot_l, "all");

        return final_shot_l;

    def runOnSingleVideo(self, video_filename):
        shot_boundaries_l = []

        vid_name = video_filename.split('/')[-1];

        printCustom("--------------------------", STDOUT_TYPE.INFO)
        printCustom("Process video: " + str(vid_name) + " ... ", STDOUT_TYPE.INFO)

        # load video
        self.vid_instance = Video();
        self.vid_instance.load(video_filename);

        # candidate selection
        printCustom("Process candidate selection: " + str(vid_name) + " ... ", STDOUT_TYPE.INFO)
        candidate_selection_result_np = self.candidate_selection_instance.run(video_filename)

        # shot boundary detection
        printCustom("Process shot boundary detection: " + str(vid_name) + " ... ", STDOUT_TYPE.INFO)
        shots_np = self.runWithCandidateSelection(candidate_selection_result_np)
        shot_boundaries_l.append(shots_np);
        shot_boundaries_np = np.squeeze(np.array(shot_boundaries_l));

        # convert shot boundaries to final shots
        final_shot_l = self.convertShotBoundaries2Shots(shot_boundaries_np);

        # export final results
        if (self.config_instance.save_final_results == 1):
            self.exportResultsToCsv(final_shot_l, vid_name.split('.')[0]);

        return final_shot_l;

    def exportResultsToCsv(self, shot_l: list, name: str):
        printCustom("Export shot list to csv file ... ", STDOUT_TYPE.INFO);

        fp = open(self.config_instance.path_final_results + "/final_shots_" + str(name) + ".csv", 'w');
        fp.write("shot_id;vid_name;start;end" + "\n")
        for shot in shot_l:
            tmp_str = shot.convert2String();
            fp.write(tmp_str + "\n");
        fp.close()

    def runWithoutCandidateSelection(self, src_path, vid_name):
        #printCustom("process shot detection ... ", STDOUT_TYPE.INFO);

        # load video
        self.vid_instance = Video();
        self.vid_instance.load(src_path + "/" + vid_name);

        # initial pre-trained model
        self.net = PyTorchModel(model_arch=self.config_instance.backbone_cnn);

        number_of_frames = int(self.vid_instance.number_of_frames);
        #print(number_of_frames)
        results_l = [];
        shot_l = []
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
            result = self.calculateDistance(feature_prev, feature_curr)

            if (int(self.config_instance.save_raw_results) == 1):
                results_l.append([idx_prev, idx_curr, result])

            if (result > self.config_instance.threshold):
                printCustom("Abrupt Cut detected: " + str(idx_prev) + ", " + str(idx_curr), STDOUT_TYPE.INFO)
                shot_l.append([self.vid_instance.vidName, (idx_prev, idx_curr)])

        # save raw results to file
        if (int(self.config_instance.save_raw_results) == 1):
            print("save raw results ... ")
            results_np = np.array(results_l)
            self.exportRawResultsAsCsv_New(results_np)

        # convert shot boundaries to shots
        shots_np = np.array(shot_l)
        return shots_np;

    def runWithCandidateSelection(self, candidates_np):
        #printCustom("process shot detection ... ", STDOUT_TYPE.INFO);

        # initial pre-trained model
        self.net = PyTorchModel(model_arch=self.config_instance.backbone_cnn);

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

                frm_trans_prev = self.pre_proc_instance.applyTransformOnImg(frm_prev)
                frm_trans_curr = self.pre_proc_instance.applyTransformOnImg(frm_curr)

                # print("process core part ... ")
                feature_prev = self.net.getFeatures(frm_trans_prev)
                feature_curr = self.net.getFeatures(frm_trans_curr)
                result = self.calculateDistance(feature_prev, feature_curr)

                if (int(self.config_instance.save_raw_results) == 1):
                    results_per_range.append(result)

                if(result > self.config_instance.threshold):
                    printCustom("Abrupt Cut detected: " + str(idx_prev) + ", " + str(idx_curr), STDOUT_TYPE.INFO)
                    shot_l.append([self.vid_instance.vidName, (idx_prev, idx_curr)])

            if (int(self.config_instance.save_raw_results) == 1):
                results_l.append([start, end, results_per_range])

        # save raw results to file
        if (int(self.config_instance.save_raw_results) == 1):
            print("save raw results ... ")
            results_np = np.array(results_l)
            self.exportRawResultsAsCsv_New(results_np)

        # convert shot boundaries to shots
        shots_np = np.array(shot_l)
        return shots_np;

    def exportRawResultsAsCsv(self, results_np: np.ndarray):
        # save raw results to file
        fp = open(self.config_instance.path_raw_results +
                  self.config_instance.path_prefix_raw_results +
                  str(self.vid_instance.vidName.split('.')[0]) + "." +
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
                  str(self.vid_instance.vidName.split('.')[0]) + "." +
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
        start_curr, stop_curr = shot_boundaries_np[0][1];
        shot_start = 0;
        shot_end = start_curr;
        shot = Shot(1, vidname_curr, shot_start, shot_end);
        shot_l.append(shot)

        for i in range(1, len(shot_boundaries_np)):
            #print(i)
            start_prev, stop_prev = shot_boundaries_np[i-1][1];
            start_curr, stop_curr = shot_boundaries_np[i][1];
            vidname_curr = shot_boundaries_np[i][0];

            shot_start = int(stop_prev);
            shot_end = int(start_curr);
            shot = Shot(i + 1, vidname_curr, shot_start, shot_end);
            shot_l.append(shot)

        vidname_curr = shot_boundaries_np[-1][0];
        start_curr, stop_curr = shot_boundaries_np[-1][1];
        shot_start = int(stop_curr);
        shot_end = int(self.vid_instance.number_of_frames);
        shot = Shot(len(shot_boundaries_np), vidname_curr, shot_start, shot_end);
        shot_l.append(shot)


        if(self.config_instance.debug_flag == 1):
            # print shot infos
            for i in range(0, len(shot_l)):
                shot_l[i].printShotInfo()

        return shot_l;






