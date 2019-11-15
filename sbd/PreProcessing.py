from sbd.utils import *
import cv2
import numpy as np
from sbd.Configuration import Configuration


class PreProcessing:
    def __init__(self, config_instance: Configuration):
        printCustom("create instance of preprocessing ... ", STDOUT_TYPE.INFO);
        self.config_instance = config_instance;

    def applyTransformOnImg(self, image: np.ndarray) -> np.ndarray:
        image_trans = image;

        # convert to grayscale image
        if(int(self.config_instance.flag_convert2Gray) == 1):
            image_trans = self.convertRGB2Gray(image_trans);

        # resize image
        if(self.config_instance.flag_downscale == 1):
            dim = (int(image_trans.shape[0] * self.config_instance.scale_factor), int(image_trans.shape[1] * self.config_instance.scale_factor));
            image_trans = self.resize(image_trans, dim)

        # apply histogram equalization
        if(self.config_instance.opt_histogram_equ == 'classic'):
            image_trans = self.classicHE(image_trans);
        elif(self.config_instance.opt_histogram_equ == 'clahe'):
            image_trans = self.claHE(image_trans)
        #elif(self.config_instance.opt_histogram_equ == 'none'):
        #    image_trans

        return image_trans;

    def applyTransformOnImgSeq(self, img_seq: np.ndarray) -> np.ndarray:
        #printCustom("NOT IMPLEMENTED YET", STDOUT_TYPE.INFO);
        img_seq_trans_l = [];
        for i in range(0, len(img_seq)):
            img_seq_trans_l.append(self.applyTransformOnImg(img_seq[i]));
        img_seq_trans = np.array(img_seq_trans_l)
        return img_seq_trans

    def convertRGB2Gray(self, img: np.ndarray):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
        img_gray = np.expand_dims(img_gray, axis=-1)
        return img_gray;

    def crop(self, img: np.ndarray, dim: tuple):
        img_crop = img;
        return img_crop;

    def resize(self, img: np.ndarray, dim: tuple):
        img_resized = cv2.resize(img, dim);
        return img_resized;

    def classicHE(self, img: np.ndarray):
        # classic histogram equalization
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_che =  cv2.equalizeHist(gray);
        img_che = cv2.cvtColor(img_che, cv2.COLOR_GRAY2RGB)
        return img_che;

    def claHE(self, img: np.ndarray):
        # contrast Limited Adaptive Histogram Equalization
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE();
        img_clahe = clahe.apply(gray)
        img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        return img_clahe;

