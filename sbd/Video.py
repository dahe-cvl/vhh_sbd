import numpy as np
import cv2
import datetime
from sbd.utils import *


class Video:
    def __init__(self):
        printCustom("create instance of video class ... ", STDOUT_TYPE.INFO);
        self.vidFile = '';
        self.vidName = "";
        self.frame_rate = 0;
        self.channels = 0;
        self.height = 0;
        self.width = 0;
        self.format = '';
        self.length = 0;
        self.number_of_frames = 0;
        self.vid = None;
        self.convert_to_gray = False;
        self.convert_to_hsv = False;

    def load(self, vidFile: str):
        printCustom("load video information ... ", STDOUT_TYPE.INFO);
        self.vidFile = vidFile;
        if(self.vidFile == ""):
            printCustom("ERROR: you must add a video file path!", STDOUT_TYPE.ERROR);
            exit(1);
        self.vidName = self.vidFile.split('/')[-1]
        self.vid = cv2.VideoCapture(self.vidFile);

        if(self.vid.isOpened() == False):
            printCustom("ERROR: not able to open video file!", STDOUT_TYPE.ERROR);
            exit(1);

        status, frm = self.vid.read();

        self.channels = frm.shape[2];
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT);
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH);
        self.frame_rate = self.vid.get(cv2.CAP_PROP_FPS);
        self.format = self.vid.get(cv2.CAP_PROP_FORMAT);
        self.number_of_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT);

        self.vid.release();

    def printVIDInfo(self):
        print("---------------------------------");
        print("Video information");
        print("filename: " + str(self.vidFile));
        print("format: " + str(self.format));
        print("fps: " + str(self.frame_rate));
        print("channels: " + str(self.channels));
        print("width: " + str(self.width));
        print("height: " + str(self.height));
        print("nFrames: " + str(self.number_of_frames));
        print("---------------------------------");


    def getFrame(self, frame_id: int) -> np.ndarray:
        self.vid.open(self.vidFile);
        if(frame_id >= self.number_of_frames):
            print("ERROR: frame idx out of range!");
            return None;

        print("Read frame with id: " + str(frame_id));
        time_stamp_start = datetime.datetime.now().timestamp();

        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_id);
        status, frame_np = self.vid.read();
        self.vid.release();

        if(status == True):
            if(self.convert_to_gray == True):
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY);
                #print(frame_gray_np.shape);
            if (self.convert_to_hsv == True):
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2HSV);
                h, s, v = cv2.split(frame_np);

        time_stamp_end = datetime.datetime.now().timestamp();
        time_diff = time_stamp_end - time_stamp_start;
        print("time: " + str(round(time_diff, 4)) + " sec");

        return frame_np;


      #data: np.ndarray, labels: np.ndarray) -> float:




