import os
import numpy as np
import glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def list_all_files(rootdir):
    # 返回某目录下的所以文件（包括子目录下的）
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files

class ReadValData(Dataset):
    def __init__(self, image_root, image_size, crop_size, data_augumentation=None):
        pic_paths = []
        labels    = []
        self.CLS_dict = []
        i = 0
        for class_dir in os.listdir(image_root):
            print(class_dir+" Class: "+str(i))
            Files = list_all_files(os.path.join(image_root,class_dir))
            pic_paths+=Files
            labels += [i for _ in range(len(Files))]
            i+=1
            dict = {'name': class_dir,'class': i, 'num': len(Files), 'correct':0, 'wrong':0, 'wrong_path':[]}
            self.CLS_dict.append(dict)

        print("==> [in LoadDataVal] len(pic): {}, len(labels): {}".format(len(pic_paths), len(labels)))
        for cls in range(len(self.CLS_dict)):
            print("Class: "+str(self.CLS_dict[cls]["name"])+" num: "+str(self.CLS_dict[cls]["num"]))

        self.data = [(pic_path, label) for pic_path, label in zip(pic_paths, labels)]
        self.data_augumentation = data_augumentation
        self.image_size = image_size
        self.crop_size = crop_size
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        (path, label) = self.data[idx]
        temp_img = Image.open(path)
        if self.data_augumentation:
            result = transforms.Compose([
                transforms.CenterCrop((self.crop_size, self.crop_size)),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                # transforms.Normalize([0.7906623], [0.16963087])#[0.7906623] [0.16963087]
            ])(temp_img)
        else:
            result = transforms.Compose([
                transforms.CenterCrop((self.crop_size, self.crop_size)),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                # transforms.Normalize([0.7906623], [0.16963087])#[0.7906623] [0.16963087]
            ])(temp_img)

        return {'result':result,'label':torch.LongTensor([label])}