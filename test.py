import os
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from config import Config
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import *
from model import Net
from dataloader.LoadDataTrain import ReadData
from dataloader.LoadDataVal import ReadValData
from dataloader.LoadDataTest import ReadTestData
import shutil
import pdb
cfg = Config()
args = init_args()
model_save_path = args.model
# ===============================================
#            1. Load Data 
# ===============================================
print("=====================================")
print(cfg.model_type)
test_image_file = cfg.test_image_file
test_dataset = ReadTestData(test_image_file, cfg.test_image_size, cfg.crop_image_size)
test_cls_dict = test_dataset.CLS_dict
test_data_loader = DataLoader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory)
print("==> finish loading test data")
# ===============================================
#            2. Load Model 
# ===============================================
model=Net(cfg)
loss_fc = model.loss().cuda()
validator_function = model.validator_function()
optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
model = nn.DataParallel(model, cfg.device_ids)
if model_save_path:
    if os.path.exists(model_save_path):
        model, optimizer, current_epoch, loss = load_model(model, optimizer, model_save_path)
model = model.cuda()
# ===============================================
#            3. Test model 
# ===============================================
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for sample_batched in tqdm(test_data_loader):
        input_data = Variable(sample_batched['result']).cuda()
        labels = Variable(sample_batched['label']).cuda()
        paths = sample_batched['path']
        outputs = model(input_data)
        count_tmp = validator_function(outputs, labels, test_cls_dict, paths)
        correct += count_tmp
        total += int(len(labels))

print("correct is ", correct)
max_test_acc = correct / total
message_test = "==> [TESTING] acc {} ".format(max_test_acc)
write_log(cfg.log_txt_path,message_test)
for cls in range(len(test_cls_dict)):
    message_res = ("Class: "+str(test_cls_dict[cls]["name"])+" correct: "+str(test_cls_dict[cls]["correct"])+" wrong: "+str(test_cls_dict[cls]["wrong"]))
    write_log(cfg.log_txt_path,message_res)

if cfg.save_badcase:
    for cls in range(len(test_cls_dict)):
        for i in range(len(test_cls_dict[cls]["wrong_path"])):
            if not os.path.exists(os.path.join(cfg.badcase_path,test_cls_dict[cls]["name"])):
                os.mkdir(os.path.join(cfg.badcase_path,test_cls_dict[cls]["name"]))
            shutil.copy(test_cls_dict[cls]["wrong_path"][i], os.path.join(cfg.badcase_path,test_cls_dict[cls]["name"],os.path.basename(test_cls_dict[cls]["wrong_path"][i])))
