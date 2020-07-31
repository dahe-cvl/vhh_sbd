from sbd.Evaluation import Evaluation
from sbd.utils import *

printCustom("Welcome to the sbd evauation framework!", STDOUT_TYPE.INFO)
config_file = "/home/dhelm/VHH_Develop/installed_pkg/vhh_pkgs/vhh_sbd/config/config_vhh_test.yaml"
sbd_eval_instance = Evaluation(config_file)
sbd_eval_instance.run()
