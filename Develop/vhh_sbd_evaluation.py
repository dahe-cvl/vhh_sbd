from sbd.Evaluation import Evaluation
from sbd.utils import *

printCustom("Welcome to the sbd evauation framework!", STDOUT_TYPE.INFO)
config_file = "config/config_holocaust_squeezenet_without_candidate.yaml"
sbd_eval_instance = Evaluation(config_file)
sbd_eval_instance.run()
