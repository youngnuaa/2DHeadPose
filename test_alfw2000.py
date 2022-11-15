import os
import time
import torch
import json
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter, Resize, RandomCrop, CenterCrop

from net.loss import *
from net.network import Hopenet, HopeSenet, HopenetResAtt, HopenetResAtt_Scale
from config import Config
from dataloader.loader import *

import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

def prepare_device(device):
    n_gpu_use = len(device)
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
            n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    list_ids = device
    device = torch.device('cuda:{}'.format(
        device[0]) if n_gpu_use > 0 else 'cpu')

    return device, list_ids



config = Config()
config.logname = "resnet50_last1_198class_ori_lr_1.5-3_guss_2_alpha_10_l2_resize2-4_crop04-08_nolearnscale33_muli_RandomErasing"
config.train_path = './tools/train.txt'
config.test_path = './tools/val_f.txt'
config.gpu_ids = [0, 1, 2, 3,4,5, 6,7]
config.onegpu = 16

config.init_lr = 1.5e-3
config.num_epochs = 300
config.steps = [40, 55, 65]
config.offset = True
config.val = True
config.val_frequency = 1
config.cos_f = 50
config.alpha = 10




print("cuda num",torch.cuda.device_count())




testtransform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#testtransform = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
testdataset = AFLW2000(filename_path=config.test_path, transform=testtransform)
testloader = DataLoader(testdataset, batch_size=config.onegpu*len(config.gpu_ids),shuffle=False, num_workers=16)

# net
print('Net...')
device, device_ids = prepare_device(config.gpu_ids)


net = HopenetResAtt_Scale(198)
net.load_state_dict(torch.load("./ckpt/_epoch_53.pkl"))
net = net.to(device)



idx_tensor = [idx for idx in range(198)]
idx_tensor = torch.FloatTensor(idx_tensor).to(device)




def val():
    net.eval()
    print('val start')


    net.eval()

    yaw_error = .0
    pitch_error = .0
    roll_error = .0

    total = .0

    for i, data in enumerate(testloader, 0):
        images = data["img"]
        total += images.shape[0]

        yaw_value = data["yaw_value"]
        pitch_value = data["pitch_value"]
        roll_value = data["roll_value"]

        images = images.cuda()
        #print(images.shape)

        yaw_value = yaw_value.float().cuda()
        pitch_value = pitch_value.float().cuda()
        roll_value = roll_value.float().cuda()

        with torch.no_grad():
            _, _, _, yaw_predicted, pitch_predicted, roll_predicted = net(images)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 1 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 1 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 1 - 99

            yaw_error += torch.sum(torch.abs(yaw_predicted - yaw_value))
            pitch_error += torch.sum(torch.abs(pitch_predicted - pitch_value))
            roll_error += torch.sum(torch.abs(roll_predicted - roll_value))


    yaw_error = yaw_error / total
    pitch_error = pitch_error / total
    roll_error = roll_error / total
    mean_error = (yaw_error + pitch_error + roll_error)/3

    print('Test error in degrees of the model on the ' + str(total) +
          ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f, Mean: %.4f'
          % (yaw_error, pitch_error, roll_error, mean_error))



    best_val = mean_error
    best_yaw = yaw_error
    best_pitch = pitch_error
    best_roll = roll_error

    print(' best result %.4f. Yaw: %.4f, Pitch: %.4f, Roll: %.4f, Mean: %.4f'
          % (1,best_yaw, best_pitch, best_roll, best_val))





if __name__ == '__main__':

    val()
