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

        # parse configuration
        #cwd = os.getcwd()
        #print(cwd)

        config_file = "../config/config.yaml";
        config_instance = Configuration(config_file);
        config_instance.loadConfig();

        #self.net = Squeezenet();
        self.net = VGG19a();

    def calculateCosineSimilarity(self, x, y):
        dst = distance.cosine(x, y)
        return dst;

    def run(self):
        #printCustom("process shot detection ... ", STDOUT_TYPE.INFO);

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
        shot_boundaries_np = self.calculateSimilarityMetric(results_np, threshold=0.8);

        #print("postprocess results ... ")

        # export shot boundaries as csv
        self.exportResultsAsCsv(shot_boundaries_np);

        # convert shot boundaries to shots
        shot_l = self.convertShotBoundaries2Shots(shot_boundaries_np);

        #printCustom("successfully finished!", STDOUT_TYPE.INFO);
        return shot_l;


    def exportRawResultsAsCsv(self, results_np: np.ndarray):
        # save raw results to file
        fp = open("./" + "results_raw_" + str(self.vid_instance.vidName.split('.')[0]) + ".csv", mode='w');
        for i in range(0, len(results_np)):
            # for j in range(0, len(results_np[0])):
            # print(sb_np[i])
            fp.write(self.vid_instance.vidName.split('.')[0] + ";" + str(results_np[i]).replace('.', ',') + "\n")
            # csv_writer.writerow(row);
        fp.close();

    def exportResultsAsCsv(self, shot_boundaries_np):
        # save final results to file
        fp = open("./" + "results_" + str(self.vid_instance.vidName.split('.')[0]) + ".csv", mode='w');
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






