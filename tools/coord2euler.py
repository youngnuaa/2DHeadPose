import cv2
import numpy as np
import os
import math


label_path = "D:\\head_pose_label\\10"
root_bbx_path = "D:\\head_pose_label\\bbx"


def read_bbx(path):
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


def read_txt(path):
    f = open(path, "r")
    lines = f.readlines()
    img_names = []
    bbxs = []
    for index in range(len(lines)):
        lines[index] = lines[index].replace("\n", "")

    for line in lines:
        line = line.split("%%")
        img_names.append(line[0])
        x1 = int(float(line[1]))
        y1 = int(float(line[2]))
        x2 = int(float(line[3]))
        y2 = int(float(line[4]))
        bbxs.append([x1, y1, x2, y2])

    return img_names, bbxs


def read_angle_coord(path):
    f = open(path, "r", encoding='utf-8')
    #lines = f.readline()
    #print(lines)
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
    print(lines)
    for line in lines[1:]:
        line = line.split(",")
        for l in line:
            coord.append(float(l))
    coord = np.array(coord)
    return  coord


def coord2eular(coord):
    a = coord.reshape(3,3)

    b = np.array([[-1.0, 0, 0],
                  [0, -1.0, 0],
                  [0, 0, 1.0]])

    a1 = math.sqrt(a[0, 0]**2 + a[0, 1]**2 + a[0, 2]**2)
    a2 = math.sqrt(a[1, 0]**2 + a[1, 1]**2 + a[1, 2]**2)
    a3 = math.sqrt(a[2, 0] ** 2 + a[2, 1] ** 2 + a[2, 2] ** 2)

    b[0, 0] = -a1
    b[1, 1] = -a2
    b[2, 2] = a3

    mat_inv = np.linalg.inv(a)

    t = np.dot(mat_inv, b)
    #print(np.dot(t, a))
    #print(t)

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


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100, thickness=(2, 2, 2)):
    """
    Function used to draw y (headpose label) on Input Image x.
    Implemented by: shamangary
    https://github.com/shamangary/FSA-Net/blob/master/demo/demo_FSANET.py
    Modified by: Omar Hassan
    """
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2-20

    # X-Axis pointing to right. drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),thickness[0])
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),thickness[1])
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),thickness[2])

    return img


def write_pose(file_path, values):
    f = open(file_path, "w")
    x1, y1, x2, y2, yaw, pitch, roll = values
    context = str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y2)+" "+str(yaw)+" "+str(pitch)+" "+str(roll)
    f.writelines(context + "\n")

    f.close()



def predict():

    img_and_coord_path = "../imgs/crop_face_img"
    path = "../imgs/face.txt"


    img_names, bbxs = read_txt(path)
    for index in range(len(img_names)):
        img_name = img_names[index]
        coord_file_name = img_name[:-4] + ".txt"

        pose_file_name = img_name[:-4] + "_pose.txt"

        img_path = os.path.join(img_and_coord_path, img_name)
        coord_file_path = os.path.join(img_and_coord_path, coord_file_name)
        pose_file_path = os.path.join(img_and_coord_path, pose_file_name)



        if os.path.exists(img_path) and os.path.exists(coord_file_path):
            #img = cv2.imread(img_path)

            bbx = bbxs[index]
            x1, y1, x2, y2 = bbx

            c_x = int((x1 + x2) / 2)
            c_y = int((y1 + y2) / 2)

            coord = read_angle_coord(coord_file_path)
            pose_vale = coord2eular(coord)

            write_pose(pose_file_path, [x1, y1, x2, y2, pose_vale[0], pose_vale[1], pose_vale[2]])

            #img = draw_axis(img, pose_vale[0], pose_vale[1], pose_vale[2], tdx=c_x, tdy=c_y - 20, size=150, thickness=(3,3,3))
            #cv2.imshow("img", img)
            #cv2.waitKey(1000)



def read_bbx_pose(path):
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




if __name__ == '__main__':
    #predict()

    path = "../imgs/crop_face_img"
    for roots, _, files in os.walk(path):
        for file in files:
            if file[-4:] == ".jpg":
                pose_file = file[:-4]+"_pose.txt"
                pose_file_path = os.path.join(roots, pose_file)
                img_path = os.path.join(roots, file)
                if os.path.exists(pose_file_path) and os.path.join(img_path):
                    bbx, pose = read_bbx_pose(pose_file_path)
                    img = cv2.imread(img_path)
                    x1, y1, x2, y2 = bbx

                    c_x = int((x1 + x2) / 2)
                    c_y = int((y1 + y2) / 2)

                    img = draw_axis(img, pose[0], pose[1], pose[2], tdx=c_x, tdy=c_y - 20, size=150, thickness=(3,3,3))
                    cv2.imshow("img", img)
                    cv2.waitKey(1000)




    """
    my_datapaths = ["/home/data1/data_wy/widerface_pose/2",
                         "/home/data1/data_wy/widerface_pose/9",
                         "/home/data1/data_wy/widerface_pose/11"]

    my_databbxpath = "/home/data1/data_wy/widerface_pose/bbx"

    for label_path in my_datapaths:
        for _, _, files in os.walk(label_path):
            for file in files:
                if file[-4:] == ".jpg":
                    data = {}
                    img_path = os.path.join(label_path, file)
                    pos_path = img_path[:-4] + ".txt"
                    pose_file_path = img_path[:-4] + "_pose.txt"
                    # print(pos_path)
                    bbx_path = os.path.join(my_databbxpath, file[:-4] + ".txt")
                    try:
                        coord = read_angle_coord(pos_path)
                    except:
                        print(img_path)

                    pose_vale = coord2eular(coord)
                    bbxs = read_bbx(bbx_path)
                    # print(bbxs)
                    data["img_path"] = img_path

                    bbx = np.array(bbxs[0])
                    data["box"] = bbx

                    write_pose(pose_file_path, [bbx[0], bbx[1], bbx[2], bbx[3], pose_vale[0], pose_vale[1], pose_vale[2]])
    """





