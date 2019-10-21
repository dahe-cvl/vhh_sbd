from sbd.Evaluation import Evaluation
from sbd.utils import *
from sbd.Video import Video

printCustom("Welcome to the sbd evauation framework!", STDOUT_TYPE.INFO);

vid_instance = Video();
vid_instance.load("../Demo/EF-NS_026_OeFM.m4v");

sbd_eval_instance = Evaluation();
sbd_eval_instance.evaluation(vid_instance)