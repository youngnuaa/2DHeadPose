import torch
import torch.nn as nn
from net.resnet import *
import os


class HopenetResAtt_Scale(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, num_bins):

        super(HopenetResAtt_Scale, self).__init__()

        resnet = resnet50(pretrained=True, receptive_keep=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc_yaw = nn.Linear(2048, num_bins)
        self.fc_pitch = nn.Linear(2048, num_bins)
        self.fc_roll = nn.Linear(2048, num_bins)

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)


        yaw_feature = torch.sigmoid(pre_yaw)
        pitch_feature = torch.sigmoid(pre_pitch)
        roll_feature = torch.sigmoid(pre_roll)


        pre_yaw = torch.nn.functional.softmax(pre_yaw, dim=1)
        pre_pitch = torch.nn.functional.softmax(pre_pitch, dim=1)
        pre_roll = torch.nn.functional.softmax(pre_roll, dim=1)


        return yaw_feature, pitch_feature, roll_feature, pre_yaw, pre_pitch, pre_roll




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_ids = [0]
    torch.cuda.set_device(0)
    heads = {'hm_pos': 1, 'hm_scale': 1, 'hm_offet': 2}

    model = HopenetResAtt_Scale(198)
    model.cuda(0)
    model.eval()

    x = torch.rand(4, 3, 224, 224).cuda()
    for i in range(1):

        # out3, out4, out5 = model(x)
        pre_yaw, pre_pitch, pre_roll, pre_yaw_value, pre_pitch_value, pre_roll_value = model(x)

        print(pre_yaw.shape)
        print(pre_pitch.shape)
        print(pre_roll.shape)

        print(pre_yaw_value.shape)
        print(pre_pitch_value.shape)
        print(pre_roll_value.shape)




