import re, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
import shutil
from .ghostnet import ghostnet
from .MobileNetV3New import mobilenet
from .ResNet import resnet18
from .inceptionresnetv2 import inceptionresnetv2
from .vgg19 import vgg19_bn
from .loss import FocalLoss
from .inceptionV4 import inceptionv4

def _validate(modelOutput, labels, cls_dict, paths=None):
    _, maxindices = torch.max(modelOutput.data, 1)
    count = 0
    for i in range(0, labels.squeeze(1).size(0)):
        if maxindices[i] == labels.squeeze(1)[i]:
            count += 1
            cls_dict[maxindices[i]]["correct"] += 1
        else:
            cls_dict[labels.squeeze(1)[i]]["wrong"] += 1
            if paths is not None:
                cls_dict[labels.squeeze(1)[i]]["wrong_path"].append(paths[i])

    return count

class Net(nn.Module):
    def __init__(self,Config):
        super(Net,self).__init__()
        if Config.model_type == "GhostNet":
            self.backbone = ghostnet(num_classes=Config.num_classes)
        elif Config.model_type == "MobileNetV3_Small":
            self.backbone = mobilenet(num_classes=Config.num_classes, mymode='small')
        elif Config.model_type == "MobileNetV3_Large":
            self.backbone = mobilenet(num_classes=Config.num_classes, mymode='large')
        elif Config.model_type == "Resnet18":
            self.backbone = resnet18(num_classes=Config.num_classes)
        elif Config.model_type == "inceptionresnetv2":
            self.backbone = inceptionresnetv2(num_classes=Config.num_classes,pretrained=None)
        elif Config.model_type == "VGG19":
            self.backbone = vgg19_bn(num_classes=Config.num_classes)
        elif Config.model_type == "inceptionv4":
            self.backbone = inceptionv4(num_classes=Config.num_classes,pretrained=None)

        #function to initialize the weights and biases of each module. Matches the
        #classname with a regular expression to determine the type of the module, then
        #initializes the weights for it.
        def weights_init(m):
            classname = m.__class__.__name__
            if re.search("Conv[123]d", classname):
                m.weight.data.normal_(0.0, 0.02)
            elif re.search("BatchNorm[123]d", classname):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)
            elif re.search("Linear", classname):
                m.bias.data.fill_(0)

        #Apply weight initialization to every module in the model.
        if Config.model_type is "MobileNetV3_Small" or "MobileNetV3_Large":
            pass
        else:
            self.apply(weights_init)

    def forward(self, input):
        output = self.backbone(input)
        return output
    
    def loss(self):
        return nn.CrossEntropyLoss()
        # return FocalLoss()
    
    def validator_function(self):
        return _validate


# import sys,os
# sys.path.append(os.path.dirname(__file__) + os.sep + '../')
# from config import Config
# cfg = Config()
# model=Net(cfg)
# # from torchsummary import summary
# # summary(model, input_size=(1, 500, 500), device="cpu")
# input = torch.randn(2,1,500,500)
# output = model(input)
# print(output.size())#torch.Size([2, 2])

# loss_fc = model.loss()
# print(loss_fc(output.squeeze(1),output.squeeze(1)))