import os
import time
from copy import deepcopy
import numpy as np
import cv2
import shutil

save_path = "/home/data1/data_wy/myself_data/multi_face_data/"

path = "../data/head_pose/"

def readHelmetHeadTxtTxt(path, targetname=["helmet", "head"], classname={"helmet": 0, "head": 1}):
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
        data_line[count] = data_line[count].lower()

        if data_line[count] not in targetname:
            count += 5
            continue
        if classname == {}:
            label = data_line[count]
        else:
            label = classname[data_line[count]]
        if float(data_line[count + 4]) - float(data_line[count + 2]) < 5:
            count += 5
            continue

        if targetname == []:
            bbxs.append([label, float(data_line[count + 1]), float(data_line[count + 2]),
                         float(data_line[count + 3]), float(data_line[count + 4])])
        elif data_line[count] in targetname:
            bbxs.append([label, float(data_line[count + 1]), float(data_line[count + 2]),
                         float(data_line[count + 3]), float(data_line[count + 4])])
        count += 5

    return bbxs


def read_pose(path):
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    lines = lines[0]
    # print(lines)
    lines = lines.replace('\n', '')
    lines = lines.replace('\ufeff', '')
    lines = lines.split(",")

    yaw_value = float(lines[0])
    pitch_value = float(lines[1])
    roll_value = float(lines[2])

    return [pitch_value, roll_value, yaw_value]


def find_pose_txt(file_name, files):
    for file in files:
        if file == file_name:
            return file
    return None


def write_file(path, bbx, pose):
    f = open(path, "w")
    x1, y1, x2, y2 = bbx[1], bbx[2], bbx[3], bbx[4]
    pitch_value, roll_value, yaw_value = pose

    f.writelines(str(x1)+"\n")
    f.writelines(str(y1) + "\n")
    f.writelines(str(x2) + "\n")
    f.writelines(str(y2) + "\n")
    f.writelines(str(pitch_value) + "\n")
    f.writelines(str(yaw_value) + "\n")
    f.writelines(str(roll_value) + "\n")
    f.close()


def val():
    pose_file_path = "../data/9/"
    pose_files = os.listdir(pose_file_path)
    print(len(pose_files))
    index = 0
    for roots, _, files in os.walk(path):
        for file in files:
            if file[-4:] == ".jpg":
                img_path = os.path.join(roots, file)
                save_img_path = os.path.join(save_path, file)
                save_file_path = os.path.join(save_path, file[:-4] + ".txt")

                file_path = os.path.join(roots, file[:-4] + ".txt")
                file_name = file[:-4] + ".txt"
                file_pose_file = find_pose_txt(file_name, pose_files)
                if file_pose_file is None:
                    continue

                index += 1
                file_pose_file_path = os.path.join(pose_file_path, file_pose_file)
                anno_value = read_pose(file_pose_file_path)

                anno_pitch_value, anno_roll_value, anno_yaw_value = anno_value

                if anno_pitch_value==0 and anno_roll_value==0 and anno_yaw_value==0:
                    continue

                bbxes = readHelmetHeadTxtTxt(file_path)


                for bbx in bbxes:
                    x1, y1, x2, y2 = bbx[1], bbx[2], bbx[3], bbx[4]
                    write_file(save_file_path, bbx, anno_value)
                    shutil.copy(img_path, save_img_path)



if __name__ == '__main__':
    val()