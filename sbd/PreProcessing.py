from sbd.utils import *
import cv2
import numpy as np


class PreProcessing:
    def __init__(self):
        printCustom("create instance of preprocessing ... ", STDOUT_TYPE.INFO);

    def applyTransformOnImg(self, image: np.ndarray) -> np.ndarray:
        #printCustom("NOT IMPLEMENTED YET", STDOUT_TYPE.INFO);

        #dim = (int(self.vid_instance.width / 2), int(self.vid_instance.height / 2));

        #image_trans = self.convertRGB2Gray(image);
        image_trans = image;

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

