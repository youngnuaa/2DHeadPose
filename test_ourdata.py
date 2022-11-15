from torch.utils.data import DataLoader
from net.network import HopenetResAtt_Scale
from config import Config
from dataloader.loader import *
import torch


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


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
config.logname = "log"



config.test_path = ''


config.gpu_ids = [0]
config.onegpu = 1

config.init_lr = 4e-4
config.num_epochs = 300
config.steps = [151,251]
config.offset = True
config.val = True
config.val_frequency = 1
config.cos_f = 25
Config.alpha = 1

#write log
if not os.path.exists('./ckpt'):
    os.mkdir('./ckpt')
if not os.path.exists('./log'):
    os.mkdir('./log')




# dataset
print('Dataset...')
traintransform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
testdataset = OurData_test(filename_path=config.test_path, transform=traintransform)
testloader = DataLoader(testdataset, batch_size=config.onegpu*len(config.gpu_ids),shuffle=True, num_workers=4)



# net
print('Net...')
device, device_ids = prepare_device(config.gpu_ids)

net = HopenetResAtt_Scale(198)
net.load_state_dict(torch.load("./ckpt/_epoch_53.pkl"))
net = net.to(device)



idx_tensor = [idx for idx in range(198)]
idx_tensor = torch.FloatTensor(idx_tensor).to(device)


if(len(device_ids) > 1):
    net = torch.nn.DataParallel(net, device_ids=device_ids)


batchsize = config.onegpu * len(config.gpu_ids)
train_batches = len(testloader)

config.print_conf()



def val():

    print('val start')

    net.eval()
    yaw_error = .0
    pitch_error = .0
    roll_error = .0
    total = .0

    for i, data in enumerate(testloader, 0):
        images = data["img"]
        total += 1

        yaw_value = data["yaw_value"]
        pitch_value = data["pitch_value"]
        roll_value = data["roll_value"]


        images = images.cuda()

        yaw_value = yaw_value.float().cuda()
        pitch_value = pitch_value.float().cuda()
        roll_value = roll_value.float().cuda()

        with torch.no_grad():
            yaw_feature, pitch_feature, roll_feature, pre_yaw, pre_pitch, pre_roll = net(images)


            yaw_predicted = torch.sum(pre_yaw * idx_tensor, 1) * 1 - 99
            pitch_predicted = torch.sum(pre_pitch * idx_tensor, 1) * 1 - 99
            roll_predicted = torch.sum(pre_roll * idx_tensor, 1) * 1 - 99

            yaw_error += torch.sum(torch.abs(yaw_predicted - yaw_value))
            pitch_error += torch.sum(torch.abs(pitch_predicted - pitch_value))
            roll_error += torch.sum(torch.abs(roll_predicted - roll_value))



    yaw_error = yaw_error / total
    pitch_error = pitch_error / total
    roll_error = roll_error / total
    mean_error = (yaw_error + pitch_error + roll_error)/3
    print('Test error in degrees of the model on the ' + str(total) +
    ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f, mean: %.4f' % (yaw_error,pitch_error, roll_error, mean_error))


if __name__ == '__main__':

    val()