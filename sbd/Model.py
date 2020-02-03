from sbd.utils import *
import numpy as np
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image

class PyTorchModel():
    def __init__(self, model_arch):
        printCustom("create instance of PyTorchModel ... ", STDOUT_TYPE.INFO);
        self.model_arch = model_arch
        if(self.model_arch == "squeezenet"):
            self.model = models.squeezenet1_0(pretrained=True);
        elif (self.model_arch == "vgg16"):
            self.model = models.vgg16(pretrained=True);
        else:
            self.model_arch = None;
            printCustom("No valid backbone cnn network selected!", STDOUT_TYPE.ERROR)
            exit();

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def getFeatures(self, frm: np.ndarray):
        # print("calculate features ... ")
        try:
            image = Image.fromarray(frm.astype('uint8'))
            loader = transforms.Compose([transforms.ToTensor()])
            image = loader(image).float()
            image = self.normalize(image)

            image = Variable(image, requires_grad=True)
            image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet

            self.model.features = self.model.features.to('cuda')
            self.model.features.eval()
            with torch.no_grad():
                CUDA = torch.cuda.is_available()
                if CUDA:
                    inputs = image.cuda()

                outputs = self.model.features(inputs)
                outputs_flatten = outputs.view(outputs.size(0), -1)
                #print(outputs_flatten.size())
        except:
            print("+++++++++++++++++++++++++++++")
            exit(1);

        return outputs_flatten.cpu().detach().numpy();

    def getFeaturesFromRange(self, batch):

        if(len(batch) == 0):
            return None;
        '''
        images = [];
        for i in range(0, len(frames)):
            images.append(Image.fromarray(frames[i].astype('uint8')))
        #images_np = np.array(images);
        print(images_np.shape)
        '''
        # print("calculate features ... ")
        #try:

        #loader = transforms.Compose([transforms.ToTensor()])
        #images_np = loader(batch).float()
        #images_np = self.normalize(images_np)

        #images_np = Variable(images_np, requires_grad=True)
        #images_np = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet

        self.model.features = self.model.features.to('cuda')
        self.model.features.eval()
        with torch.no_grad():
            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = batch.cuda()

            outputs = self.model.features(inputs)
            outputs_flatten = outputs.view(outputs.size(0), -1)
            print(outputs_flatten.size())
        #except:
        #    print("+++++++++++++++++++++++++++++")
        #    exit(1);

        return outputs_flatten.cpu().detach().numpy();
