from sbd.utils import *
import yaml


class Configuration:
    def __init__(self, config_file: str):
        printCustom("create instance of configuration ... ", STDOUT_TYPE.INFO)

        self.config_file = config_file;

    def loadConfig(self):
        fp = open(self.config_file, 'r');
        config = yaml.load(fp);

        for section in config:
            print(section)
        print(config['example_config_01']);
        print(config['example_config_02']);
