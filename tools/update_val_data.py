import os




AF = open("./300W_LP_filename_filtered.txt", "r")
v = open("./train.txt", "r")

new_f = open("./train_f.txt", "w")


AFlines = AF.readlines()
AF.close()

vlines = v.readlines()
v.close()

for i in range(len(AFlines)):
    AFlines[i] = AFlines[i].replace('\n', '')
    AFlines[i] = AFlines[i].replace('\t', '')
    AFlines[i] = AFlines[i].replace('\r', '')


for i in range(len(vlines)):
    vlines[i] = vlines[i].replace('\n', '')
    vlines[i] = vlines[i].replace('\t', '')
    vlines[i] = vlines[i].replace('\r', '')


data_line = []
#/home/data1/data_wy/300W_LP/LFPW_Flip/LFPW_image_train_0263_4.jpg
#IBUG/IBUG_image_052_1_8
for index in range(len(AFlines)):
    f_r = AFlines[index]
    path = os.path.join("/home/data1/data_wy/300W_LP/", f_r+".jpg")
    data_line.append(path)



print(len(data_line))

for data in data_line:
    new_f.writelines(data)
    new_f.writelines("\n")
new_f.close()