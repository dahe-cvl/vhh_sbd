from sbd.utils import *
import yaml


class Configuration:
    def __init__(self, config_file: str):
        printCustom("create instance of configuration ... ", STDOUT_TYPE.INFO)

        self.config_file = config_file;

        self.debug_flag = None;

        self.flag_convert2Gray = None;
        self.flag_downscale = None;
        self.opt_histogram_equ = None;

        self.save_raw_results = None;
        self.path_postfix_raw_results = None;
        self.path_prefix_raw_results = None;
        self.path_raw_results = None;

        self.save_final_results = None;
        self.path_prefix_final_results = None;
        self.path_postfix_final_results = None;
        self.path_final_results = None;

        self.path_videos = None;
        self.threshold = None;
        self.backbone_cnn = None;
        self.similarity_metric = None;

        self.path_eval_results = None;
        self.save_eval_results = None;

    def loadConfig(self):
        fp = open(self.config_file, 'r');
        config = yaml.load(fp);

        developer_config = config['Development'];
        pre_processing_config = config['PreProcessing'];
        post_processing_config = config['PostProcessing'];
        sbd_core_config = config['SbdCore'];
        visualization_config = config['Visualization'];
        evaluation_config = config['Evaluation'];

        # developer_config section
        self.debug_flag = developer_config['DEBUG_FLAG'];

        # pre-processing section
        self.flag_convert2Gray = pre_processing_config['CONVERT2GRAY'];
        self.flag_downscale = pre_processing_config['DOWNSCALE'];
        self.opt_histogram_equ = pre_processing_config['HISTOGRAM_EQU'];

        # sbd_core_config section
        self.save_raw_results = sbd_core_config['SAVE_RAW_RESULTS'];
        self.path_postfix_raw_results = sbd_core_config['POSTFIX_RAW_RESULTS'];
        self.path_prefix_raw_results = sbd_core_config['PREFIX_RAW_RESULTS'];
        self.path_raw_results = sbd_core_config['PATH_RAW_RESULTS'];

        self.save_final_results = sbd_core_config['SAVE_FINAL_RESULTS'];
        self.path_prefix_final_results = sbd_core_config['PREFIX_FINAL_RESULTS'];
        self.path_postfix_final_results = sbd_core_config['POSTFIX_FINAL_RESULTS'];
        self.path_final_results = sbd_core_config['PATH_FINAL_RESULTS'];

        self.path_videos = sbd_core_config['PATH_VIDEOS'];
        self.threshold = sbd_core_config['THRESHOLD'];
        self.backbone_cnn = sbd_core_config['BACKBONE_CNN'];
        self.similarity_metric = sbd_core_config['SIMILARITY_METRIC'];

        # evaluation section
        self.path_eval_results = evaluation_config['PATH_EVAL_RESULTS'];
        self.save_eval_results = evaluation_config['SAVE_EVAL_RESULTS'];




