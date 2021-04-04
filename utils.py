import torch
import argparse

def write_log(log_dir, message):
    print(message)
    with open(log_dir, 'a') as f:
        f.write(message + "\n")

def load_model(model, optimizer, ckp_path):
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, epoch, loss

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    return parser.parse_args()