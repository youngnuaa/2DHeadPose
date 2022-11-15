from __future__ import division
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
from PIL import Image, ImageFilter
import os
import math
from dataloader.data_augment import augment_data
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter
import json



class RandomErasing(object):
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def __call__(self, img):

        if np.random.rand() > self.p:
            return img


        while True:
            img_h, img_w, img_c = img.shape

            img_area = img_h * img_w
            mask_area = np.random.uniform(self.sl, self.sh) * img_area
            mask_aspect_ratio = np.random.uniform(self.r1, self.r2)
            mask_w = int(np.sqrt(mask_area / mask_aspect_ratio))
            mask_h = int(np.sqrt(mask_area * mask_aspect_ratio))

            mask = np.random.rand(mask_h, mask_w, img_c) * 255

            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            right = left + mask_w
            bottom = top + mask_h

            if right <= img_w and bottom <= img_h:
                break

        img[top:bottom, left:right, :] = mask

        return img


def gaussian_label(label, num_class=198, sig=1):
    # x = np.array(range(math.floor(-num_class/2), math.ceil(num_class/2), 1))
    x = np.array(range(math.floor(0), math.ceil(num_class), 1))
    # label = np.zeros(shape=(num_class))
    # print(label)
    y_sig = np.exp(-(x - label) ** 2 / (2 * sig ** 2))
    # y_sig = np.exp(-1 * ((x - label) ** 2) / (2 * (sig ** 2))) / (math.sqrt(2 * np.pi) * sig)
    # print(y_sig)
    return y_sig


def rotate(ps, m):
    pts = np.float32(ps)
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(m, pts)
    target_point = [ [target_point[0][x], target_point[1][x]] for x in range(len(target_point[0])) ]
    target_point = np.array(target_point)
    return target_point


def random_roatate(img, angle, pitch, yaw, roll, bbx):
    pitch = pitch / 180 * np.pi
    yaw = yaw / 180 * np.pi
    roll = roll / 180 * np.pi


    x1, y1, x2, y2 = bbx
    bbx = [[x1, y1],[x2, y2], [x2, y1], [x1, y2]]
    bbx = np.array(bbx)

    c_x = int((x1+x2)/2)
    c_y = int((y1 + y2) / 2)
    h,w,c = img.shape


    M = cv2.getRotationMatrix2D((c_x, c_y), angle, 1.0)

    img = cv2.warpAffine(img, M, (w, h))
    target_point = rotate(bbx, M)
    #print(target_point)
    min_x = min(target_point[:, 0])
    min_y = min(target_point[:, 1])
    max_x = max(target_point[:, 0])
    max_y = max(target_point[:, 1])

    angle = -angle / 180 * np.pi
    r1 = [[math.cos(roll), math.sin(roll), 0],
          [-math.sin(roll), math.cos(roll), 0],
          [0, 0, 1]]
    r2 = [[math.cos(yaw), 0, -math.sin(yaw)],
          [0, 1, 0],
          [math.sin(yaw),0,math.cos(yaw)]]
    r3 = [[1, 0, 0],
          [0, math.cos(pitch), math.sin(pitch)],
          [0, -math.sin(pitch), math.cos(pitch)]]
    r4 = [[math.cos(angle), math.sin(angle), 0],
          [-math.sin(angle), math.cos(angle), 0],
          [0, 0, 1]]
    r1 = np.array(r1)
    r2 = np.array(r2)
    r3 = np.array(r3)
    r4 = np.array(r4)
    r = np.matmul(r1, r2)
    r = np.matmul(r, r3)
    r = np.matmul(r, r4)
    #there are some error
    #p1 = -math.atan(r[2,1]/r[2,2])*180/np.pi
    p1 = -math.atan2(r[2, 1] , r[2, 2]) * 180 / np.pi
    y1 = math.asin(r[2,0])*180/np.pi
    r1 = -math.atan2(r[1,0] , r[0,0])*180/np.pi
    #r1 = -math.atan(r[1,0]/r[0,0])*180/np.pi
    return img, min_x, min_y, max_x, max_y, p1, y1, r1



class Pose_300W_LP_random_ds(Dataset):
    # 300W-LP dataset with random downsampling
    def __init__(self, filename_path, transform, img_ext='.jpg', annot_ext='.mat'):

        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        self.filename_list = self.get_list_from_filenames(filename_path)
        self.dataset = self.load_img()
        self.length = len(self.dataset)
        print("train num", self.length)

        self.random_E = RandomErasing()

    def __getitem__(self, index):
        data_ = self.dataset[index]
        img_path = data_["img_path"]
        box = data_["box"]
        pitch = data_["pitch"]
        yaw = data_["yaw"]
        roll = data_["roll"]

        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])

        h = y_max - y_min
        w = x_max - x_min

        #ad = 0.6
        rnd = np.random.random_sample()
        ad = 0.4*rnd + 0.4

        x_min = max(int(x_min - ad * w), 0)
        x_max = min(int(x_max + ad * w), img_w - 1)
        y_min = max(int(y_min - ad * h), 0)
        y_max = min(int(y_max + ad * h), img_h - 1)

        img = img[y_min:y_max, x_min:x_max]







        img_h, img_w, _ = img.shape
        rnd = np.random.random_sample()
        if rnd < 0.5:
            rate = random.randint(2, 4)
            rate_h = int(img_h / rate)
            rate_w = int(img_w / rate)
            img = cv2.resize(img, (rate_w, rate_h), interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)



        img = cv2.resize(img, (224, 224))
        img = augment_data(img)
        img = self.random_E(img)

        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = cv2.flip(img, 1)


        # Bin values
        bins = np.array(range(-99, 100, 1))
        binned_pose = np.digitize([yaw, pitch, roll], bins)-1

        yaw_label = self.gaussian_label(binned_pose[0], len(bins)-1)
        pitch_label = self.gaussian_label(binned_pose[1], len(bins)-1)
        roll_label = self.gaussian_label(binned_pose[2], len(bins)-1)

        data = {}


        data["yaw_class"] = binned_pose[0]
        data["pitch_class"] = binned_pose[1]
        data["roll_class"] = binned_pose[2]


        data["yaw_guss"] = yaw_label
        data["pitch_guss"] = pitch_label
        data["roll_guss"] = roll_label


        data["yaw_value"] = yaw
        data["pitch_value"] = pitch
        data["roll_value"] = roll



        #img = self.random_E(img)
        if self.transform is not None:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # Blur?
            #rnd = np.random.random_sample()
            if rnd < 0.03:
                img = img.filter(ImageFilter.BLUR)
            img = self.transform(img)

        data["img"] = img
        #print(img.shape)

        return data

    def __len__(self):
        # 122,450
        return self.length

    def gaussian_label(self, label, num_class=198, sig=5):
        # x = np.array(range(math.floor(-num_class/2), math.ceil(num_class/2), 1))
        x = np.array(range(math.floor(0), math.ceil(num_class), 1))
        # label = np.zeros(shape=(num_class))
        # print(label)
        y_sig = np.exp(-(x - label) ** 2 / (2 * sig ** 2))
        #y_sig = np.exp(-1 * ((x - label) ** 2) / (2 * (sig ** 2))) / (math.sqrt(2 * np.pi) * sig)
        # print(y_sig)
        return y_sig


    def get_list_from_filenames(self, path):
        f = open(path, "r")
        lines = f.readlines()
        f.close()
        data_line = []
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            lines[i] = lines[i].replace('\t', '')
            lines[i] = lines[i].replace('\r', '')
            # lines[i] = lines[i].replace(' ', '')
            if len(lines[i]) > 0:
                data_line.append(lines[i])
        return data_line


    def get_pt2d_from_mat(self, mat_path):
        # Get 2D landmarks
        mat = sio.loadmat(mat_path)
        pt2d = mat['pt2d']
        return pt2d


    def get_ypr_from_mat(self, mat_path):
        # Get yaw, pitch, roll from .mat annotation.
        # They are in radians
        mat = sio.loadmat(mat_path)
        # [pitch yaw roll tdx tdy tdz scale_factor]
        pre_pose_params = mat['Pose_Para'][0]
        # Get [pitch, yaw, roll]
        pose_params = pre_pose_params[:3]
        return pose_params


    def load_img(self):
        dataset = []
        for index in range(len(self.filename_list)):
            data={}

            img_path = self.filename_list[index]
            data["img_path"] = img_path

            mat_path = img_path[:-4] + ".mat"
            pt2d = self.get_pt2d_from_mat(mat_path)
            x_min = min(pt2d[0,:])
            y_min = min(pt2d[1,:])
            x_max = max(pt2d[0,:])
            y_max = max(pt2d[1,:])
            box = [x_min, y_min, x_max, y_max]
            data["box"] = np.array(box)
            pose = self.get_ypr_from_mat(mat_path)
            pitch = pose[0] * 180 / np.pi
            yaw = pose[1] * 180 / np.pi
            roll = pose[2] * 180 / np.pi
            if pitch<-99 or pitch>99 or yaw<-99 or yaw>99 or roll<-99 or roll>99:
                continue
            data["pitch"] = pitch
            data["yaw"] = yaw
            data["roll"] = roll
            dataset.append(data)

        return dataset



class Pose_300W_LP_random_add_mydata_ds(Dataset):
    # 300W-LP dataset with random downsampling
    def __init__(self, filename_path, transform, img_ext='.jpg', annot_ext='.mat'):

        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext


        self.my_datapaths = ["/home/data1/data_wy/widerface_pose/2",
                             "/home/data1/data_wy/widerface_pose/9",
                             "/home/data1/data_wy/widerface_pose/11",
                             ]

        self.my_databbxpath = "/home/data1/data_wy/widerface_pose/bbx"


        self.filename_list = self.get_list_from_filenames(filename_path)
        self.dataset = self.load_img()
        self.load_mydata()
        self.length = len(self.dataset)
        print("train num", self.length)

        self.random_E = RandomErasing()


    def __getitem__(self, index):
        data_ = self.dataset[index]
        img_path = data_["img_path"]
        box = data_["box"]
        pitch = data_["pitch"]
        yaw = data_["yaw"]
        roll = data_["roll"]

        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])

        h = y_max - y_min
        w = x_max - x_min

        #ad = 0.6
        rnd = np.random.random_sample()
        ad = 0.4*rnd + 0.4

        x_min = max(int(x_min - ad * w), 0)
        x_max = min(int(x_max + ad * w), img_w - 1)
        y_min = max(int(y_min - ad * h), 0)
        y_max = min(int(y_max + ad * h), img_h - 1)

        img = img[y_min:y_max, x_min:x_max]



        img_h, img_w, _ = img.shape

        if img_h>img_w:
            diff = int((img_h - img_w)/2)
            new_img = np.zeros(shape=(img_h, img_h, 3))
            new_img[:, diff:img_w+diff, :] = img

        else:
            diff = int((img_w - img_h) / 2)
            new_img = np.zeros(shape=(img_w, img_w, 3))
            new_img[diff:img_h + diff, :, :] = img

        img = new_img.astype(np.uint8)



        rnd = np.random.random_sample()
        if rnd < 0.5:
            rate = random.randint(2, 4)
            rate_h = int(img_h / rate)
            rate_w = int(img_w / rate)
            img = cv2.resize(img, (rate_w, rate_h), interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)




        img = cv2.resize(img, (224, 224))

        img = augment_data(img)
        img = self.random_E(img)

        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = cv2.flip(img, 1)


        # Bin values
        bins = np.array(range(-99, 100, 1))
        binned_pose = np.digitize([yaw, pitch, roll], bins)-1

        yaw_label = self.gaussian_label(binned_pose[0], len(bins)-1)
        pitch_label = self.gaussian_label(binned_pose[1], len(bins)-1)
        roll_label = self.gaussian_label(binned_pose[2], len(bins)-1)

        data = {}


        data["yaw_class"] = binned_pose[0]
        data["pitch_class"] = binned_pose[1]
        data["roll_class"] = binned_pose[2]


        data["yaw_guss"] = yaw_label
        data["pitch_guss"] = pitch_label
        data["roll_guss"] = roll_label


        data["yaw_value"] = yaw
        data["pitch_value"] = pitch
        data["roll_value"] = roll



        #img = self.random_E(img)
        if self.transform is not None:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # Blur?
            #rnd = np.random.random_sample()
            if rnd < 0.03:
                img = img.filter(ImageFilter.BLUR)
            img = self.transform(img)

        data["img"] = img
        #print(img.shape)

        return data

    def __len__(self):
        # 122,450
        return self.length

    def gaussian_label(self, label, num_class=198, sig=2):
        # x = np.array(range(math.floor(-num_class/2), math.ceil(num_class/2), 1))
        x = np.array(range(math.floor(0), math.ceil(num_class), 1))
        # label = np.zeros(shape=(num_class))
        # print(label)
        y_sig = np.exp(-(x - label) ** 2 / (2 * sig ** 2))
        #y_sig = np.exp(-1 * ((x - label) ** 2) / (2 * (sig ** 2))) / (math.sqrt(2 * np.pi) * sig)
        # print(y_sig)
        return y_sig


    def get_list_from_filenames(self, path):
        f = open(path, "r")
        lines = f.readlines()
        f.close()
        data_line = []
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            lines[i] = lines[i].replace('\t', '')
            lines[i] = lines[i].replace('\r', '')
            # lines[i] = lines[i].replace(' ', '')
            if len(lines[i]) > 0:
                data_line.append(lines[i])
        return data_line


    def get_pt2d_from_mat(self, mat_path):
        # Get 2D landmarks
        mat = sio.loadmat(mat_path)
        pt2d = mat['pt2d']
        return pt2d


    def get_ypr_from_mat(self, mat_path):
        # Get yaw, pitch, roll from .mat annotation.
        # They are in radians
        mat = sio.loadmat(mat_path)
        # [pitch yaw roll tdx tdy tdz scale_factor]
        pre_pose_params = mat['Pose_Para'][0]
        # Get [pitch, yaw, roll]
        pose_params = pre_pose_params[:3]
        return pose_params


    def load_img(self):
        dataset = []
        for index in range(len(self.filename_list)):
            data={}

            img_path = self.filename_list[index]
            data["img_path"] = img_path

            mat_path = img_path[:-4] + ".mat"
            pt2d = self.get_pt2d_from_mat(mat_path)
            x_min = min(pt2d[0,:])
            y_min = min(pt2d[1,:])
            x_max = max(pt2d[0,:])
            y_max = max(pt2d[1,:])
            box = [x_min, y_min, x_max, y_max]
            data["box"] = np.array(box)
            pose = self.get_ypr_from_mat(mat_path)
            pitch = pose[0] * 180 / np.pi
            yaw = pose[1] * 180 / np.pi
            roll = pose[2] * 180 / np.pi

            if pitch<-99 or pitch>99 or yaw<-99 or yaw>99 or roll<-99 or roll>99:
                continue

            data["pitch"] = pitch
            data["yaw"] = yaw
            data["roll"] = roll
            dataset.append(data)

        return dataset


    #read my data
    def read_bbx(self, path):
        f = open(path, "r")
        lines = f.readlines()
        f.close()
        data_line = []
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            lines[i] = lines[i].replace('\t', '')
            lines[i] = lines[i].replace('\r', '')
            lines[i] = lines[i].replace(' ', '')
            if len(lines[i]) > 0:
                data_line.append(lines[i])
        num = int(data_line[0])
        bbxs = []
        count = 1
        for i in range(num):
            bbxs.append([float(data_line[count + 1]), float(data_line[count + 2]), float(data_line[count + 3]),
                         float(data_line[count + 4])])
            count += 5
        return bbxs


    def read_bbx_pose(self, path):
        f = open(path, "r")
        lines = f.readlines()
        f.close()
        data_line = []

        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            lines[i] = lines[i].replace('\t', '')
            lines[i] = lines[i].replace('\r', '')
            if len(lines[i]) > 0:
                data_line.append(lines[i])

        bbx_pose = data_line[0].split(" ")
        bbx = [float(bbx_pose[0]), float(bbx_pose[1]), float(bbx_pose[2]), float(bbx_pose[3])]
        pose = [float(bbx_pose[4]), float(bbx_pose[5]), float(bbx_pose[6])]
        return bbx, pose



    def read_angle_coord(self, path):
        f = open(path, "r", encoding='utf-8')
        # lines = f.readline()
        # print(lines)
        lines = f.readlines()
        f.close()
        data_line = []
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            lines[i] = lines[i].replace('\t', '')
            lines[i] = lines[i].replace('\r', '')
            lines[i] = lines[i].replace(' ', '')
            if len(lines[i]) > 0:
                data_line.append(lines[i])
        lines = data_line[0].split("|")

        coord = []
        for line in lines[1:]:
            line = line.split(",")
            for l in line:
                coord.append(float(l))
        coord = np.array(coord)
        return coord


    def coord2eular(self, coord):
        a = coord.reshape(3, 3)

        b = np.array([[-1.0, 0, 0],
                      [0, -1.0, 0],
                      [0, 0, 1.0]])

        a1 = math.sqrt(a[0, 0] ** 2 + a[0, 1] ** 2 + a[0, 2] ** 2)
        a2 = math.sqrt(a[1, 0] ** 2 + a[1, 1] ** 2 + a[1, 2] ** 2)
        a3 = math.sqrt(a[2, 0] ** 2 + a[2, 1] ** 2 + a[2, 2] ** 2)

        b[0, 0] = -a1
        b[1, 1] = -a2
        b[2, 2] = a3

        mat_inv = np.linalg.inv(a)

        t = np.dot(mat_inv, b)
        # print(np.dot(t, a))
        # print(t)

        t = np.transpose(t)

        pitch = math.atan2(t[2, 1], t[2, 2]) * 180 / 3.14
        roll = -math.atan2(t[1, 0], t[0, 0]) * 180 / 3.14
        sy = math.sqrt(t[0, 0] * t[0, 0] + t[1, 0] * t[1, 0])
        y = math.atan2(-t[2, 0], sy)
        yaw = -y * 180 / 3.14

        #print("yaw:", yaw)
        #print("pitch:", pitch)
        #print("roll:", roll)
        return [yaw, pitch, roll]


    def load_mydata(self):
        for label_path in self.my_datapaths:
            for roots, _, files in os.walk(label_path):
                for file in files:
                    if file[-4:] == ".jpg":
                        pose_file = file[:-4] + "_pose.txt"
                        pose_file_path = os.path.join(roots, pose_file)
                        img_path = os.path.join(roots, file)
                        if os.path.exists(pose_file_path) and os.path.join(img_path):
                            bbx, pose_vale = self.read_bbx_pose(pose_file_path)
                            data = {}

                            # print(bbxs)
                            data["img_path"] = img_path

                            bbx = np.array(bbx)
                            data["box"] = bbx

                            pitch = pose_vale[1]
                            yaw = pose_vale[0]
                            roll = pose_vale[2]

                            if pitch < -99 or pitch > 99 or yaw < -99 or yaw > 99 or roll < -99 or roll > 99:
                                continue

                            data["pitch"] = pitch
                            data["yaw"] = yaw
                            data["roll"] = roll
                            self.dataset.append(data)



class OurData_test(Dataset):
    # 300W-LP dataset with random downsampling
    def __init__(self, filename_path, transform, img_ext='.jpg', annot_ext='.mat'):

        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext


        self.my_datapaths = ["/home/data1/data_wy/widerface_pose/5"]

        self.my_databbxpath = "/home/data1/data_wy/widerface_pose/bbx"


        self.dataset = []
        self.load_mydata()
        self.length = len(self.dataset)
        print("test num", self.length)


    def __getitem__(self, index):
        data_ = self.dataset[index]
        img_path = data_["img_path"]
        box = data_["box"]
        pitch = data_["pitch"]
        yaw = data_["yaw"]
        roll = data_["roll"]

        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])


        w, h = x_max-x_min, y_max-y_min



        if w<h:
            diff = int((h - w)/2)
            x_min -= diff
            x_max += diff

        ad = 0.6
        x_min = max(int(x_min - ad * w), 0)
        x_max = min(int(x_max + ad * w), img_w - 1)
        y_min = max(int(y_min - ad * h), 0)
        y_max = min(int(y_max + ad * h), img_h - 1)


        img = img[y_min:y_max, x_min:x_max]

        img_h, img_w, _ = img.shape

        if img_h > img_w:
            diff = int((img_h - img_w) / 2)
            new_img = np.zeros(shape=(img_h, img_h, 3))
            new_img[:, diff:img_w + diff, :] = img

        else:
            diff = int((img_w - img_h) / 2)
            new_img = np.zeros(shape=(img_w, img_w, 3))
            new_img[diff:img_h + diff, :, :] = img

        img = new_img.astype(np.uint8)


        try:
            img = cv2.resize(img, (224,224))
        except:
            print("###########################")
            print(x_min)
            print(x_max)
            print(y_min)
            print(y_max)
            print("###########################")

        data = {}

        data["img_path"] = img_path
        data["yaw_value"] = yaw
        data["pitch_value"] = pitch
        data["roll_value"] = roll


        # img = self.random_E(img)
        if self.transform is not None:
            try:
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except:
                print(x_min)
                print(x_max)
                print(y_min)
                print(y_max)
            img = self.transform(img)

        data["img"] = img

        return data


    def __len__(self):
        # 122,450
        return self.length


    def gaussian_label(self, label, num_class=198, sig=5):
        # x = np.array(range(math.floor(-num_class/2), math.ceil(num_class/2), 1))
        x = np.array(range(math.floor(0), math.ceil(num_class), 1))
        # label = np.zeros(shape=(num_class))
        # print(label)
        y_sig = np.exp(-(x - label) ** 2 / (2 * sig ** 2))
        #y_sig = np.exp(-1 * ((x - label) ** 2) / (2 * (sig ** 2))) / (math.sqrt(2 * np.pi) * sig)
        # print(y_sig)
        return y_sig


    def get_list_from_filenames(self, path):
        f = open(path, "r")
        lines = f.readlines()
        f.close()
        data_line = []
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            lines[i] = lines[i].replace('\t', '')
            lines[i] = lines[i].replace('\r', '')
            # lines[i] = lines[i].replace(' ', '')
            if len(lines[i]) > 0:
                data_line.append(lines[i])
        return data_line


    def get_pt2d_from_mat(self, mat_path):
        # Get 2D landmarks
        mat = sio.loadmat(mat_path)
        pt2d = mat['pt2d']
        return pt2d


    def get_ypr_from_mat(self, mat_path):
        # Get yaw, pitch, roll from .mat annotation.
        # They are in radians
        mat = sio.loadmat(mat_path)
        # [pitch yaw roll tdx tdy tdz scale_factor]
        pre_pose_params = mat['Pose_Para'][0]
        # Get [pitch, yaw, roll]
        pose_params = pre_pose_params[:3]
        return pose_params


    def load_img(self):
        dataset = []
        for index in range(len(self.filename_list)):
            data={}

            img_path = self.filename_list[index]
            data["img_path"] = img_path

            mat_path = img_path[:-4] + ".mat"
            pt2d = self.get_pt2d_from_mat(mat_path)
            x_min = min(pt2d[0,:])
            y_min = min(pt2d[1,:])
            x_max = max(pt2d[0,:])
            y_max = max(pt2d[1,:])
            box = [x_min, y_min, x_max, y_max]
            data["box"] = np.array(box)
            pose = self.get_ypr_from_mat(mat_path)
            pitch = pose[0] * 180 / np.pi
            yaw = pose[1] * 180 / np.pi
            roll = pose[2] * 180 / np.pi

            if pitch<-99 or pitch>99 or yaw<-99 or yaw>99 or roll<-99 or roll>99:
                continue

            data["pitch"] = pitch
            data["yaw"] = yaw
            data["roll"] = roll
            dataset.append(data)

        return dataset


    #read my data
    def read_bbx(self, path):
        f = open(path, "r")
        lines = f.readlines()
        f.close()
        data_line = []
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            lines[i] = lines[i].replace('\t', '')
            lines[i] = lines[i].replace('\r', '')
            lines[i] = lines[i].replace(' ', '')
            if len(lines[i]) > 0:
                data_line.append(lines[i])
        num = int(data_line[0])
        bbxs = []
        count = 1
        for i in range(num):
            bbxs.append([float(data_line[count + 1]), float(data_line[count + 2]), float(data_line[count + 3]),
                         float(data_line[count + 4])])
            count += 5
        return bbxs

    def read_angle_coord(self, path):
        f = open(path, "r", encoding='utf-8')
        # lines = f.readline()
        # print(lines)
        lines = f.readlines()
        f.close()
        data_line = []
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            lines[i] = lines[i].replace('\t', '')
            lines[i] = lines[i].replace('\r', '')
            lines[i] = lines[i].replace(' ', '')
            if len(lines[i]) > 0:
                data_line.append(lines[i])
        lines = data_line[0].split("|")

        coord = []
        for line in lines[1:]:
            line = line.split(",")
            for l in line:
                coord.append(float(l))
        coord = np.array(coord)
        return coord

    def coord2eular(self, coord):
        a = coord.reshape(3, 3)

        b = np.array([[-1.0, 0, 0],
                      [0, -1.0, 0],
                      [0, 0, 1.0]])

        a1 = math.sqrt(a[0, 0] ** 2 + a[0, 1] ** 2 + a[0, 2] ** 2)
        a2 = math.sqrt(a[1, 0] ** 2 + a[1, 1] ** 2 + a[1, 2] ** 2)
        a3 = math.sqrt(a[2, 0] ** 2 + a[2, 1] ** 2 + a[2, 2] ** 2)

        b[0, 0] = -a1
        b[1, 1] = -a2
        b[2, 2] = a3

        mat_inv = np.linalg.inv(a)

        t = np.dot(mat_inv, b)
        # print(np.dot(t, a))
        # print(t)

        t = np.transpose(t)

        pitch = math.atan2(t[2, 1], t[2, 2]) * 180 / 3.14
        roll = -math.atan2(t[1, 0], t[0, 0]) * 180 / 3.14
        sy = math.sqrt(t[0, 0] * t[0, 0] + t[1, 0] * t[1, 0])
        y = math.atan2(-t[2, 0], sy)
        yaw = -y * 180 / 3.14

        #print("yaw:", yaw)
        #print("pitch:", pitch)
        #print("roll:", roll)
        return [yaw, pitch, roll]


    def load_mydata(self):
        for label_path in self.my_datapaths:
            for _, _, files in os.walk(label_path):
                for file in files:
                    if file[-4:] == ".jpg":
                        data = {}
                        img_path = os.path.join(label_path, file)
                        pos_path = img_path[:-4] + ".txt"
                        # print(pos_path)
                        bbx_path = os.path.join(self.my_databbxpath, file[:-4] + ".txt")
                        try:
                            coord = self.read_angle_coord(pos_path)
                        except:
                            print(img_path)

                        pose_vale = self.coord2eular(coord)
                        bbxs = self.read_bbx(bbx_path)
                        # print(bbxs)
                        data["img_path"] = img_path

                        bbx = np.array(bbxs[0])
                        data["box"] = bbx

                        pitch = pose_vale[1]
                        yaw = pose_vale[0]
                        roll = pose_vale[2]

                        if pitch < -99 or pitch > 99 or yaw < -99 or yaw > 99 or roll < -99 or roll > 99:
                            continue

                        data["pitch"] = pitch
                        data["yaw"] = yaw
                        data["roll"] = roll
                        self.dataset.append(data)


class load_add_mydata_ds(Dataset):
    # 300W-LP dataset with random downsampling
    def __init__(self, img_ext='.jpg', annot_ext='.mat'):


        """
        self.my_datapaths = ["/home/data1/data_wy/widerface_pose/2",
                             "/home/data1/data_wy/widerface_pose/9",
                             "/home/data1/data_wy/widerface_pose/11"]
        """


        self.my_datapaths = ["/home/data1/data_wy/widerface_pose/2",
                             "/home/data1/data_wy/widerface_pose/9",
                             "/home/data1/data_wy/widerface_pose/11",
                             "/home/data1/data_wy/widerface_pose/5",
                             ]


        self.my_databbxpath = "/home/data1/data_wy/widerface_pose/bbx"



        self.dataset = []
        self.load_mydata()
        self.length = len(self.dataset)
        print("train num", self.length)


    #read my data
    def read_bbx(self, path):
        f = open(path, "r")
        lines = f.readlines()
        f.close()
        data_line = []
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            lines[i] = lines[i].replace('\t', '')
            lines[i] = lines[i].replace('\r', '')
            lines[i] = lines[i].replace(' ', '')
            if len(lines[i]) > 0:
                data_line.append(lines[i])
        num = int(data_line[0])
        bbxs = []
        count = 1
        for i in range(num):
            bbxs.append([float(data_line[count + 1]), float(data_line[count + 2]), float(data_line[count + 3]),
                         float(data_line[count + 4])])
            count += 5
        return bbxs

    def read_angle_coord(self, path):
        f = open(path, "r", encoding='utf-8')
        # lines = f.readline()
        # print(lines)
        lines = f.readlines()
        f.close()
        data_line = []
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            lines[i] = lines[i].replace('\t', '')
            lines[i] = lines[i].replace('\r', '')
            lines[i] = lines[i].replace(' ', '')
            if len(lines[i]) > 0:
                data_line.append(lines[i])
        lines = data_line[0].split("|")

        coord = []
        for line in lines[1:]:
            line = line.split(",")
            for l in line:
                coord.append(float(l))
        coord = np.array(coord)
        return coord

    def coord2eular(self, coord):
        a = coord.reshape(3, 3)

        b = np.array([[-1.0, 0, 0],
                      [0, -1.0, 0],
                      [0, 0, 1.0]])

        a1 = math.sqrt(a[0, 0] ** 2 + a[0, 1] ** 2 + a[0, 2] ** 2)
        a2 = math.sqrt(a[1, 0] ** 2 + a[1, 1] ** 2 + a[1, 2] ** 2)
        a3 = math.sqrt(a[2, 0] ** 2 + a[2, 1] ** 2 + a[2, 2] ** 2)

        b[0, 0] = -a1
        b[1, 1] = -a2
        b[2, 2] = a3

        mat_inv = np.linalg.inv(a)

        t = np.dot(mat_inv, b)
        # print(np.dot(t, a))
        # print(t)

        t = np.transpose(t)

        pitch = math.atan2(t[2, 1], t[2, 2]) * 180 / 3.14
        roll = -math.atan2(t[1, 0], t[0, 0]) * 180 / 3.14
        sy = math.sqrt(t[0, 0] * t[0, 0] + t[1, 0] * t[1, 0])
        y = math.atan2(-t[2, 0], sy)
        yaw = -y * 180 / 3.14

        #print("yaw:", yaw)
        #print("pitch:", pitch)
        #print("roll:", roll)
        return [yaw, pitch, roll]


    def load_mydata(self):
        for label_path in self.my_datapaths:
            for _, _, files in os.walk(label_path):
                for file in files:
                    if file[-4:] == ".jpg":
                        data = {}
                        img_path = os.path.join(label_path, file)
                        pos_path = img_path[:-4] + ".txt"
                        # print(pos_path)
                        bbx_path = os.path.join(self.my_databbxpath, file[:-4] + ".txt")
                        try:
                            coord = self.read_angle_coord(pos_path)
                        except:
                            print(img_path)

                        pose_vale = self.coord2eular(coord)
                        bbxs = self.read_bbx(bbx_path)
                        # print(bbxs)
                        data["img_path"] = img_path

                        bbx = np.array(bbxs[0])
                        data["box"] = bbx

                        pitch = pose_vale[1]
                        yaw = pose_vale[0]
                        roll = pose_vale[2]

                        if pitch < -99 or pitch > 99 or yaw < -99 or yaw > 99 or roll < -99 or roll > 99:
                            continue

                        data["pitch"] = pitch
                        data["yaw"] = yaw
                        data["roll"] = roll
                        self.dataset.append(data)



class AFLW2000(Dataset):
    def __init__(self, filename_path, transform, img_ext='.jpg', annot_ext='.mat'):

        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        self.filename_list = self.get_list_from_filenames(filename_path)

        self.dataset = self.load_img()
        self.length = len(self.dataset)
        print("test num", self.length)

    def __getitem__(self, index):
        data_ = self.dataset[index]
        img_path = data_["img_path"]
        box = data_["box"]
        pitch = data_["pitch"]
        yaw = data_["yaw"]
        roll = data_["roll"]

        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])


        w, h = x_max-x_min, y_max-y_min

        """
        h_w = max(h, w) * 1.4
        x_min -= (h_w-w)//2
        y_min -= (h_w-h)//2
        x_max = x_min + h_w
        y_max = y_min + h_w
        x_min = max(0, int(x_min))
        x_max = min(img_w, int(x_max))
        y_min = max(0, int(y_min))
        y_max = min(img_h, int(y_max))

        """

        if w<h:
            diff = int((h - w)/2)
            x_min -= diff
            x_max += diff

        ad = 0.6
        x_min = max(int(x_min - ad * w), 0)
        x_max = min(int(x_max + ad * w), img_w - 1)
        y_min = max(int(y_min - ad * h), 0)
        y_max = min(int(y_max + ad * h), img_h - 1)


        img = img[y_min:y_max, x_min:x_max]

        img_h, img_w, _ = img.shape

        if img_h > img_w:
            diff = int((img_h - img_w) / 2)
            new_img = np.zeros(shape=(img_h, img_h, 3))
            new_img[:, diff:img_w + diff, :] = img

        else:
            diff = int((img_w - img_h) / 2)
            new_img = np.zeros(shape=(img_w, img_w, 3))
            new_img[diff:img_h + diff, :, :] = img

        img = new_img.astype(np.uint8)



        img = cv2.resize(img, (224,224))


        data = {}

        data["ori_img"] = img
        data["img_path"] = img_path
        data["yaw_value"] = yaw
        data["pitch_value"] = pitch
        data["roll_value"] = roll


        # img = self.random_E(img)
        if self.transform is not None:
            try:
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except:
                print(x_min)
                print(x_max)
                print(y_min)
                print(y_max)
            img = self.transform(img)

        data["img"] = img

        return data


    def __len__(self):
        # 2,000
        return self.length


    def get_list_from_filenames(self, path):
        f = open(path, "r")
        lines = f.readlines()
        f.close()
        data_line = []
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            lines[i] = lines[i].replace('\t', '')
            lines[i] = lines[i].replace('\r', '')
            # lines[i] = lines[i].replace(' ', '')
            if len(lines[i]) > 0:
                data_line.append(lines[i])
        return data_line


    def get_pt2d_from_mat(self, mat_path):
        # Get 2D landmarks
        #print(mat_path)
        mat = sio.loadmat(mat_path)
        pt2d = mat['pt2d']
        return pt2d


    def get_ypr_from_mat(self, mat_path):
        # Get yaw, pitch, roll from .mat annotation.
        # They are in radians
        mat = sio.loadmat(mat_path)
        # [pitch yaw roll tdx tdy tdz scale_factor]
        pre_pose_params = mat['Pose_Para'][0]
        # Get [pitch, yaw, roll]
        pose_params = pre_pose_params[:3]
        return pose_params


    def load_img(self):
        dataset = []
        for index in range(len(self.filename_list)):
            data = {}

            img_path = self.filename_list[index]


            data["img_path"] = img_path



            img = cv2.imread(img_path)

            mat_path = img_path[:-4] + ".mat"

            pt2d = self.get_pt2d_from_mat(mat_path)

            #print(pt2d)
            xx = pt2d[0]
            xx = xx[xx>0]

            yy = pt2d[1]
            #print("yy ",yy)
            #print(img.shape[0])
            yy = yy[yy<img.shape[0]]



            x_min = min(xx)
            #x_min = min(pt2d[0, :])
            y_min = min(pt2d[1, :])
            x_max = max(pt2d[0, :])
            #y_max = max(pt2d[1, :])

            y_max = max(yy)


            box = [x_min, y_min, x_max, y_max]
            box = np.array(box)
            data["box"] = box
            pose = self.get_ypr_from_mat(mat_path)
            pitch = pose[0] * 180 / np.pi
            yaw = pose[1] * 180 / np.pi
            roll = pose[2] * 180 / np.pi
            if pitch < -99 or pitch > 99 or yaw < -99 or yaw > 99 or roll < -99 or roll > 99:
                continue
            data["pitch"] = pitch
            data["yaw"] = yaw
            data["roll"] = roll
            dataset.append(data)


            #print(box)
            #img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            #cv2.imshow("img", img)
            #cv2.waitKey(1000)

        return dataset


class AFLW2000_test(Dataset):
    def __init__(self, filename_path, transform, img_ext='.jpg', annot_ext='.mat'):

        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        self.filename_list = self.get_list_from_filenames(filename_path)

        self.dataset = self.load_img()
        self.length = len(self.dataset)
        print("test num", self.length)

    def __getitem__(self, index):
        data_ = self.dataset[index]
        img_path = data_["img_path"]

        box = data_["box"]
        pitch = data_["pitch"]
        yaw = data_["yaw"]
        roll = data_["roll"]

        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])

        h = y_max - y_min
        w = x_max - x_min

        if w<h:
            diff = int((h - w)/2)
            x_min -= diff
            x_max += diff


        if roll<30 and roll>-30:
            #angle = (np.random.random_sample()-0.5)*90
            angle = 40
            bbx = [x_min, y_min, x_max, y_max]
            img, x_min, y_min, x_max, y_max, p1, y1, r1 = random_roatate(img, angle, pitch, yaw, roll, bbx)
            ad = 0.6
            x_min = max(int(x_min - ad * w), 0)
            x_max = min(int(x_max + ad * w), img_w - 1)
            y_min = max(int(y_min - ad * h), 0)
            y_max = min(int(y_max + ad * h), img_h - 1)

            img = img[y_min:y_max, x_min:x_max]
            img = draw_axis(img, y1, p1, r1)
            cv2.imshow("img", img)
            cv2.waitKey(2000)




        data = {}
        """
        ad = 0.6
        x_min = max(int(x_min - ad * w), 0)
        x_max = min(int(x_max + ad * w), img_w - 1)
        y_min = max(int(y_min - ad * h), 0)
        y_max = min(int(y_max + ad * h), img_h - 1)

        img = img[y_min:y_max, x_min:x_max]

        img = cv2.resize(img, (224, 224))


        #img = augment_data(img)




        data["yaw_value"] = yaw
        data["pitch_value"] = pitch
        data["roll_value"] = roll
        data["img_path"] = img_path


        data["img"] = img

        """

        return data


    def __len__(self):
        # 2,000
        return self.length


    def get_list_from_filenames(self, path):
        f = open(path, "r")
        lines = f.readlines()
        f.close()
        data_line = []
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '')
            lines[i] = lines[i].replace('\t', '')
            lines[i] = lines[i].replace('\r', '')
            # lines[i] = lines[i].replace(' ', '')
            if len(lines[i]) > 0:
                data_line.append(lines[i])
        return data_line


    def get_pt2d_from_mat(self, mat_path):
        # Get 2D landmarks
        mat = sio.loadmat(mat_path)
        pt2d = mat['pt2d']
        return pt2d


    def get_ypr_from_mat(self, mat_path):
        # Get yaw, pitch, roll from .mat annotation.
        # They are in radians
        mat = sio.loadmat(mat_path)
        # [pitch yaw roll tdx tdy tdz scale_factor]
        pre_pose_params = mat['Pose_Para'][0]
        # Get [pitch, yaw, roll]
        pose_params = pre_pose_params[:3]
        return pose_params

    def load_img(self):
        dataset = []
        for index in range(len(self.filename_list)):
            data = {}



            img_path = self.filename_list[index]
            data["img_path"] = img_path

            img = cv2.imread(img_path)

            mat_path = img_path[:-4] + ".mat"
            pt2d = self.get_pt2d_from_mat(mat_path)
            #print(pt2d)
            xx = pt2d[0]
            xx = xx[xx>0]

            yy = pt2d[1]
            #print("yy ",yy)
            #print(img.shape[0])
            yy = yy[yy<img.shape[0]]



            x_min = min(xx)
            #x_min = min(pt2d[0, :])
            y_min = min(pt2d[1, :])
            x_max = max(pt2d[0, :])
            #y_max = max(pt2d[1, :])

            y_max = max(yy)


            box = [x_min, y_min, x_max, y_max]
            box = np.array(box)
            data["box"] = box
            pose = self.get_ypr_from_mat(mat_path)
            pitch = pose[0] * 180 / np.pi
            yaw = pose[1] * 180 / np.pi
            roll = pose[2] * 180 / np.pi
            if pitch < -99 or pitch > 99 or yaw < -99 or yaw > 99 or roll < -99 or roll > 99:
                continue
            data["pitch"] = pitch
            data["yaw"] = yaw
            data["roll"] = roll
            dataset.append(data)


            #print(box)
            #img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            #cv2.imshow("img", img)
            #cv2.waitKey(1000)

        return dataset






if __name__ == '__main__':
    import matplotlib.pyplot as plt



    traintransform = Compose(
        [ColorJitter(brightness=0.5), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    testdataset = AFLW2000(filename_path='../tools/val_f.txt', transform=None)



    for index in range(len(testdataset)):
        data = testdataset[index]
        ori_img = data["ori_img"]
        cv2.imshow("img", ori_img)
        cv2.waitKey(10)
        print(data)
