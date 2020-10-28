from vhh_sbd.Evaluation import Evaluation
from vhh_sbd.utils import *

printCustom("Welcome to the sbd evauation framework!", STDOUT_TYPE.INFO)
config_file = "/caa/Homes01/dhelm/working/vhh/develop/vhh_sbd/config/config_vhh_test.yaml"
sbd_eval_instance = Evaluation(config_file)
sbd_eval_instance.run()
