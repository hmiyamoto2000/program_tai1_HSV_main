import cv2
from utils import *

path_anotation = ["./fish_img/1/", "./fish_img/2/", "./fish_img/3/"]
targets = ["1", "2", "3"]
hsv_list = ["h", "s", "v"]

for target in targets:
    for hsv_num in hsv_list:
        save_path = "./output/"+hsv_num+"/"+target+"/"
        my_makedirs(save_path)####フォルダの準備

for target, path_ano in zip(targets, path_anotation):
    for num in range(1,11):
        img = cv2.imread(path_ano + str(num) + ".tif")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]
        hsv_imgs = [h, s, v]

        for hsv_num, hsv_img in zip(hsv_list, hsv_imgs):
            save_path = "./output/"+hsv_num+"/"+target+"/"+ str(num) + ".tif"
            cv2.imwrite(save_path, hsv_img)

