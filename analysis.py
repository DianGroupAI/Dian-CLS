import os
import torch
import torchvision
import torch.nn as nn
import argparse
from tqdm import tqdm
from datetime import datetime
from config import Config
from torch.utils.data import DataLoader
from torch.autograd import Variable
import copy
import numpy as np

from model import Net
from dataloader.LoadDataTotal import ReadTotalData
cfg = Config()

def get_mean_std(dataloader, ratio=0.5):
    """Get mean and std by sample ratio
    """
    train = iter(dataloader).next()['result']   # 一个batch的数据
    mean = np.mean(train.numpy(), axis=(0,2,3))
    std = np.std(train.numpy(), axis=(0,2,3))
    return mean, std

total_image_file = cfg.total_image_file
total_dataset = ReadTotalData(total_image_file, cfg.test_image_size, cfg.crop_image_size)
data_loader = DataLoader(total_dataset, batch_size=int((len(total_dataset))), shuffle=True,
                              num_workers=cfg.num_workers, drop_last=False, pin_memory=cfg.pin_memory)

train_mean, train_std = get_mean_std(data_loader)
print(train_mean,train_std)

