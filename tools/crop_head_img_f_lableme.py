import cv2
import os
import json
import numpy as np


save_head_paths = "../imgs/crop_face_img"


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



def display_img(path):
    img_names, bbxs = read_txt(path)
    for index in range(len(img_names)):
        img_name = img_names[index]
        bbx = bbxs[index]
        img_path = os.path.join(save_head_paths, img_name)
        img = cv2.imread(img_path)

        x1, y1, x2, y2 = bbx

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)

        cv2.imshow("img", img)
        cv2.waitKey(1000)


path = "../imgs/face.txt"

display_img(path)
"""



json_face_paths = "../imgs/661/"

save_head_paths = "../imgs/crop_face_img"



def read_json(path):
    bbxs = []
    with open(path, "r", encoding="gbk") as fp:
        json_data = json.load(fp)
        #print(json_data["shapes"])
        for lines in json_data["shapes"]:
            label = lines["label"]
            point = lines["points"]
            if label == "face" or label == "facce" :
                bbxs.append(point)

    return bbxs



def write_txt(f, img_name, bbx, stat_s):

    x1, y1, x2, y2 = bbx


    context = img_name+"%%"+str(x1)+"%%"+str(y1)+"%%"+str(x2)+"%%"+str(y2)

    for st in stat_s:
        context = context+"%%"+str(st)


    f.writelines(context + "\n")


pose_label_path = "../imgs/face.txt"
f = open(pose_label_path, "w")

face_num = 0


for roots, _, files in os.walk(json_face_paths):
    for file in files:
        if file[-4:] == ".jpg":
            img_path = os.path.join(roots, file)
            json_path = os.path.join(roots, file[:-4] + ".json")
            if not os.path.exists(json_path):
                continue
            bbxs = read_json(json_path)
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape

            index = 0



            print(bbxs)
            for bbx in bbxs:
                print(bbx)
                x1 = int(bbx[0][0])
                y1 = int(bbx[0][1])
                x2 = int(bbx[1][0])
                y2 = int(bbx[1][1])

                w = x2 - x1
                h = y2 - y1


                crop_img_name = file[:-4] + "_" + str(index) + ".jpg"
                crop_file_name = crop_img_name[:-4] + ".txt"

                crop_img_path = os.path.join(save_head_paths, crop_img_name)

                ad = 1
                x_min = x1
                y_min = y1
                x_max = x1 + w
                y_max = y1 + h

                x1 = max(int(x_min - ad * w), 0)
                x2 = min(int(x_max + ad * w), img_w - 1)
                y1 = max(int(y_min - ad * h), 0)
                y2 = min(int(y_max + ad * h), img_h - 1)

                crop_img = img[y1:y2, x1:x2, :]

                crop_h, crop_w, _ = crop_img.shape

                if crop_h > crop_w:
                    diff = int((crop_h - crop_w) / 2)
                    new_img = np.zeros(shape=(crop_h, crop_h, 3))
                    new_img[:, diff:diff+crop_w, :] = crop_img
                    shit_bbx = [x_min - x1+diff, y_min - y1, x_max - x1+diff, y_max - y1]


                else:
                    diff = int((crop_w - crop_h) / 2)
                    new_img = np.zeros(shape=(crop_w, crop_w, 3))
                    new_img[diff:diff + crop_h, :, :] = crop_img
                    shit_bbx = [x_min - x1, y_min - y1+diff, x_max - x1, y_max - y1+diff]

                xx1, yy1, xx2, yy2 = shit_bbx

                #new_img = new_img.astype(np.uint8)
                #cv2.rectangle(new_img, (int(xx1), int(yy1)), (int(xx2), int(yy2)), (0,0,255), 2)
                #cv2.imshow("img", new_img)
                #cv2.waitKey(1000)


                write_txt(f, crop_img_name, shit_bbx, [0, 0, 0, 0, 0, 0])
                cv2.imwrite(crop_img_path, new_img)

                index += 1
                # if h < 30 or w < 24 or pose_value==0:
                # continue

                face_num += 1



f.close()
print(face_num)
"""

























