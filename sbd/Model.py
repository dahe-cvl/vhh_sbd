from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import preprocess_input, decode_predictions
from sbd.utils import *
from sbd.Video import Video
import cv2
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

class Squeezenet:
    def __init__(self, include_top=False):
        print("create instance of squeezenet ... ");

        self.model = SqueezeNet(include_top=False, weights='imagenet')
        #self.model.summary()

    def getFeatures(self, frm: np.ndarray):
        #print("calculate features ... ")
        try:
            x = image.img_to_array(frm)
            #print(x.shape)
            x = np.expand_dims(x, axis=0)
            #print(x.shape)
            x = preprocess_input(x)
        except:
            print("+++++++++++++++++++++++++++++")
            print(x.shape)
            exit(1);

        feature = self.model.predict(x).flatten();

        return feature;

    def getFeaturesFromRange(self, frames_np):
        # features_np = np.load("/home/dhelm/Working/testvideos/squeezenet_features.npy");
        # np.save('/caa/Homes01/dhelm/squeezenet_features.npy', features_np);

        # print(features_np)

        # model = SqueezeNet(include_top=False, weights='imagenet')
        # model.summary()

        features = [];
        frame_cnt = 0;

        for f in range(0, len(frames_np)):
            frame_cnt = frame_cnt + 1;
            frm = frames_np[f];

            dim = (int(frames_np.shape[2] / 2), int(frames_np.shape[1] / 2));
            # print(dim)
            resized = cv2.resize(frm, dim)

            x = image.img_to_array(resized)
            # print(x.shape)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            feature = self.model.predict(x).flatten();
            # print(feature.shape)
            features.append(feature);
            # print("frame_cnt: " + str(frame_cnt));

        features_np = np.array(features);

        return features_np;

class VGG19a:
    def __init__(self, weights='imagenet', include_top=False, classes=1000):
        print("create instance of vgg19");

        self.model = VGG19(weights=weights, include_top=include_top, classes=classes);

    def getFeatures(self, frm: np.ndarray):
        #print("calculate features ... ")
        try:
            x = image.img_to_array(frm)
            #print(x.shape)
            x = np.expand_dims(x, axis=0)
            #print(x.shape)
            x = preprocess_input(x)
        except:
            print("+++++++++++++++++++++++++++++")
            print(x.shape)
            exit(1);

        feature = self.model.predict(x).flatten();

        return feature;

    def getFeaturesFromRange(self, frames_np):
        features = [];
        frame_cnt = 0;

        for f in range(0, len(frames_np)):
            frame_cnt = frame_cnt + 1;
            frm = frames_np[f];

            dim = (int(frames_np.shape[2] / 2), int(frames_np.shape[1] / 2));
            # print(dim)
            resized = cv2.resize(frm, dim)

            x = image.img_to_array(resized)
            # print(x.shape)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = self.model.predict(x).flatten();
            # print(feature.shape)
            features.append(feature);
            # print("frame_cnt: " + str(frame_cnt));

        features_np = np.array(features);

        return features_np;



class KerasModel:
    def __init__(self):
        printCustom("create instance of KerasModel loader ... ", STDOUT_TYPE.INFO);

        model = VGG16()
        model.summary()

    def loadPreTrainedModel(self):
        printCustom("NOT IMPLEMENTED YET", STDOUT_TYPE.INFO);


    def predict(self):
        printCustom("NOT IMPLEMENTED YET", STDOUT_TYPE.INFO);


class PyTorchModel:
    def __init__(self):
        printCustom("create instance of PyTorchModel loader ... ", STDOUT_TYPE.INFO);

    def loadPreTrainedModel(self):
        printCustom("NOT IMPLEMENTED YET", STDOUT_TYPE.INFO);

    def predict(self):
        printCustom("NOT IMPLEMENTED YET", STDOUT_TYPE.INFO);
