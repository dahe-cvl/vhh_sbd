from vhh_sbd.Evaluation import Evaluation
from vhh_sbd.utils import *

printCustom("Welcome to the sbd evauation framework!", STDOUT_TYPE.INFO)
config_file = "./config/config_vhh_mmsi_evaluation.yaml"
sbd_eval_instance = Evaluation(config_file)
sbd_eval_instance.run()
