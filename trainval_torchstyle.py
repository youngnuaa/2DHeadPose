import time
import torch.optim as optim
from torch.utils.data import DataLoader

from net.loss import *
from net.network import HopenetResAtt_Scale
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
config.logname = "resnet50_addourdata_alpha5"
config.train_path = './tools/train.txt'
config.test_path = './tools/val_f.txt'
config.gpu_ids = [0, 1, 2, 3,4,5, 6,7]
config.onegpu = 64

config.init_lr = 1.5e-3
config.num_epochs = 300
config.steps = [40, 60, 75]
config.offset = True
config.val = True
config.val_frequency = 1
config.cos_f = 50
config.alpha = 2

#write log
if not os.path.exists('./ckpt'):
    os.mkdir('./ckpt')
if not os.path.exists('./log'):
    os.mkdir('./log')

# open log file
log_file = './log/' + time.strftime('%Y%m%d', time.localtime(time.time())) + config.logname + '.log'
log = open(log_file, 'w')


print("cuda num",torch.cuda.device_count())



# dataset
print('Dataset...')
traintransform = Compose([ColorJitter(brightness=0.5), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
traindataset = Pose_300W_LP_random_add_mydata_ds(filename_path=config.train_path, transform=traintransform)
trainloader = DataLoader(traindataset, batch_size=config.onegpu*len(config.gpu_ids),shuffle=True, num_workers=16)


testtransform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
testdataset = AFLW2000(filename_path=config.test_path, transform=testtransform)
testloader = DataLoader(testdataset, batch_size=config.onegpu*len(config.gpu_ids),shuffle=False, num_workers=16)

# net
print('Net...')
device, device_ids = prepare_device(config.gpu_ids)


net = HopenetResAtt_Scale(198)

net = net.to(device)

criterion_guss = angle_focal_loss().to(device)
reg_criterion = nn.MSELoss().to(device)



idx_tensor = [idx for idx in range(198)]
idx_tensor = torch.FloatTensor(idx_tensor).to(device)


if(len(device_ids) > 1):
    net = torch.nn.DataParallel(net, device_ids=device_ids)

# optimizer
params = []
for n, p in net.named_parameters():
    if p.requires_grad:
        params.append({'params': p})
    else:
        print(n)



optimizer = optim.Adam(params, lr=config.init_lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.steps, gamma=0.1, last_epoch=-1)


batchsize = config.onegpu * len(config.gpu_ids)
train_batches = len(trainloader)

config.print_conf()



def train():
    print('Training start')
    best_val = np.Inf
    best_yaw = 0
    best_pitch = 0
    best_roll = 0
    best_epoch = 0

    for epoch in range(config.num_epochs):
        print('----------')

        net.train()


        for i, data in enumerate(trainloader, 0):

            images = data["img"]

            yaw_guss = data["yaw_guss"]
            pitch_guss = data["pitch_guss"]
            roll_guss = data["roll_guss"]

            yaw_value = data["yaw_value"]
            pitch_value = data["pitch_value"]
            roll_value = data["roll_value"]


            images = images.cuda()

            yaw_value = yaw_value.float().cuda()
            pitch_value = pitch_value.float().cuda()
            roll_value = roll_value.float().cuda()


            yaw_guss = yaw_guss.float().cuda()
            pitch_guss = pitch_guss.float().cuda()
            roll_guss = roll_guss.float().cuda()


            optimizer.zero_grad()
            yaw, pitch, roll, yaw_predicted, pitch_predicted, roll_predicted = net(images)


            loss_yaw = criterion_guss(yaw, yaw_guss)
            loss_pitch = criterion_guss(pitch, pitch_guss)
            loss_roll = criterion_guss(roll, roll_guss)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 1 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 1 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 1 - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, yaw_value)
            loss_reg_pitch = reg_criterion(pitch_predicted, pitch_value)
            loss_reg_roll = reg_criterion(roll_predicted, roll_value)

            loss_yaw = config.alpha * loss_reg_yaw + loss_yaw
            loss_pitch = config.alpha * loss_reg_pitch + loss_pitch
            loss_roll = config.alpha * loss_reg_roll + loss_roll

            loss = loss_yaw + loss_pitch + loss_roll


            loss.backward()

            optimizer.step()

            if (i + 1) % 50 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f, lr %.6f' % (
                epoch + 1, config.num_epochs, i + 1, len(traindataset) // (config.onegpu*len(config.gpu_ids)), loss_yaw.item(), loss_pitch.item(),
                loss_roll.item(), scheduler.get_last_lr()[0]))
                log.write('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f, lr %.6f' % (
                epoch + 1, config.num_epochs, i + 1, len(traindataset) // (config.onegpu*len(config.gpu_ids)), loss_yaw.item(), loss_pitch.item(),
                loss_roll.item(), scheduler.get_last_lr()[0]))
                log.write("\n")


        scheduler.step()
        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < config.num_epochs:
            print('Taking snapshot...')
            torch.save(net.module.state_dict(),'./ckpt/' + '_epoch_' + str(epoch + 1) + '.pkl')
        if 1:
            best_val, best_yaw, best_pitch, best_roll, best_epoch = val(best_val, best_yaw, best_pitch, best_roll, best_epoch, epoch)



def val(best_val, best_yaw, best_pitch, best_roll, best_epoch, epoch):
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
    log.write("\n")
    log.write('Test error in degrees of the model on the ' + str(total) +
          ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f, Mean: %.4f'
          % (yaw_error, pitch_error, roll_error, mean_error))


    if mean_error<best_val:
        best_val = mean_error
        best_yaw = yaw_error
        best_pitch = pitch_error
        best_roll = roll_error
        best_epoch = epoch
    print(' best result %.4f. Yaw: %.4f, Pitch: %.4f, Roll: %.4f, Mean: %.4f'
          % (best_epoch,best_yaw, best_pitch, best_roll, best_val))
    log.write(' best result %.4f. Yaw: %.4f, Pitch: %.4f, Roll: %.4f, Mean: %.4f'
          % (best_epoch,best_yaw, best_pitch, best_roll, best_val))
    log.write("\n")

    return best_val, best_yaw, best_pitch, best_roll, best_epoch





if __name__ == '__main__':
    train()
    #val()
    #val_search()
    #ap_val()