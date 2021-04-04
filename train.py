import os
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from config import Config
from torch.utils.data import DataLoader
from torch.autograd import Variable
import copy
import pdb
from model import Net
from dataloader.LoadDataTrain import ReadData
from dataloader.LoadDataVal import ReadValData
from dataloader.LoadDataTest import ReadTestData
from tensorboardX import SummaryWriter
from utils import *

writer = SummaryWriter()
cfg = Config()
args = init_args()
model_save_path = args.model
# ===============================================
#            1. Load Data
# ===============================================
train_image_file = cfg.train_image_file
val_image_file = cfg.val_image_file
training_dataset = ReadData(train_image_file, cfg.train_image_size, cfg.crop_image_size)
train_cls_dict = training_dataset.CLS_dict
training_data_loader = DataLoader(training_dataset, batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory)
val_dataset = ReadValData(val_image_file, cfg.train_image_size, cfg.crop_image_size)
val_cls_dict = val_dataset.CLS_dict
val_data_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory)
print("==> finish loading Train&Val data")
# exit()
# ===============================================
#            2. Load Model
# ===============================================
model = Net(cfg)
loss_fc = model.loss().cuda()
validator_function = model.validator_function()
# optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
model = nn.DataParallel(model, cfg.device_ids)
current_epoch = 1
if model_save_path:
    if os.path.exists(model_save_path):
        model, optimizer, current_epoch, loss = load_model(model, optimizer, model_save_path)
model = model.cuda()
# ===============================================
#            3. Train model
# ===============================================
max_val_acc = 0
for epoch in range(current_epoch, cfg.NUM_EPOCHS+1):
    print("epoch ", epoch)
    model.train()
    correct = 0
    total_loss = 0
    total = 0
    for sample_batched in tqdm(training_data_loader):
        input_data = Variable(sample_batched['result']).cuda()
        labels = Variable(sample_batched['label']).cuda()
        optimizer.zero_grad()
        result = model(input_data)
        loss = loss_fc(result, labels.squeeze(1))
        total_loss += loss
        loss.backward()
        optimizer.step()

        count_tmp = validator_function(result, labels, train_cls_dict)
        correct += count_tmp
        total += int(len(labels))
    acc = correct/total
    writer.add_scalar('data/loss', total_loss, epoch)
    writer.add_scalar('data/train_acc', acc, epoch)

    message = "==> [train] epoch {}, total_loss {}, train_acc {}".format(
        epoch, total_loss, acc)
    write_log(cfg.log_txt_path, message)

    if epoch % cfg.evaluate_epoch == 0:
        model.eval()
        correct = 0
        total = 0
        for cls in range(len(val_cls_dict)):
            val_cls_dict[cls]["correct"] = 0
            val_cls_dict[cls]["wrong"] = 0
        with torch.no_grad():
            for sample_batched in tqdm(val_data_loader):
                input_data = Variable(sample_batched['result']).cuda()
                labels = Variable(sample_batched['label']).cuda()
                outputs = model(input_data)
                count_tmp = validator_function(outputs, labels, val_cls_dict)
                correct += count_tmp
                total += int(len(labels))
            val_acc = correct / total
            message = "==> [evaluate on val] epoch {}, acc {} ".format(epoch, val_acc)
            write_log(cfg.log_txt_path, message)
            for cls in range(len(val_cls_dict)):
                message_res = ("Class: "+str(val_cls_dict[cls]["name"])+" acc: "+str(val_cls_dict[cls]["correct"]/(val_cls_dict[cls]["correct"]+val_cls_dict[cls]["wrong"])))
                write_log(cfg.log_txt_path, message_res)

        writer.add_scalar('data/val_acc', val_acc, epoch)
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            max_train_acc = acc
            best_model_epoch = epoch
            best_model = copy.deepcopy(model)
            message = "[saving] epoch {}, acc {}, saving model".format(epoch, val_acc)
            write_log(cfg.log_txt_path, message)
            if max_val_acc > cfg.save_threshold:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    'loss': loss,
                }, cfg.save_path + cfg.model_type + "_epoch_" + str(epoch) + "_acc_" + str(val_acc) + ".tar")
writer.export_scalars_to_json("./boardlog/all_scalars.json")
writer.close()
# ===============================================
#            4. Test model
# ===============================================
test_image_file = cfg.test_image_file
test_dataset = ReadTestData(test_image_file, cfg.test_image_size, cfg.crop_image_size)
test_cls_dict = test_dataset.CLS_dict
test_data_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory)
print("==> finish loading test data")

model = best_model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for sample_batched in tqdm(test_data_loader):
        input_data = Variable(sample_batched['result']).cuda()
        labels = Variable(sample_batched['label']).cuda()
        outputs = model(input_data)
        count_tmp = validator_function(outputs, labels, test_cls_dict)
        correct += count_tmp
        total += int(len(labels))

print("correct is ", correct)
max_test_acc = correct / total
message_test = "==> [TESTING] epoch {}, acc {} ".format(epoch, max_test_acc)
message_best_model = ("==> [Best Model] epoch {}, train_acc {}, val_acc {}, test_acc {}".format(
    best_model_epoch, max_train_acc, max_val_acc, max_test_acc))
write_log(cfg.log_txt_path, message_test)
write_log(cfg.log_txt_path, message_best_model)

for cls in range(len(test_cls_dict)):
    message_res = ("Class: "+str(test_cls_dict[cls]["name"])+" correct: "+str(test_cls_dict[cls]["correct"])+" wrong: "+str(test_cls_dict[cls]["wrong"]))
    write_log(cfg.log_txt_path, message_res)

