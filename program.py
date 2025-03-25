import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy.stats as stats
import math
import seaborn as sns
import pandas as pd
from utils import *

targets = ["1","2","3"]
image_target = "hs_2d_histogram"
image_target2 = "hs_rectangular_coordinate_system"

def plot_masked_histogram(img_h,img_s, output_path, output_path2):
    #img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #img_h = img_hsv[:,:,0]
    #img_s = img_hsv[:,:,1]
    #img_v = img_hsv[:,:,2]
    print(img_h.dtype)
    print(img_s.dtype)

    # マスクを適用して、マスクされた部分の画像を得る
    mask2 = cv2.bitwise_not(mask)
    mask2 = mask2.astype(np.int32)
    mask2 = mask2 *10

    masked_img_h = cv2.bitwise_and(img_h, img_h, mask=mask)
    #print(np.amax(masked_img_h))
    masked_img_h = masked_img_h.astype(np.int32)
    masked_img_h += mask2
    masked_img_h = np.ravel(masked_img_h)
    masked_img_h = masked_img_h[masked_img_h<np.amax(mask2)]
    #print(np.amax(masked_img_h), np.amax(mask2))

    masked_img_s = cv2.bitwise_and(img_s, img_s, mask=mask)
    masked_img_s = masked_img_s.astype(np.int32)
    masked_img_s += mask2
    masked_img_s = np.ravel(masked_img_s)
    masked_img_s = masked_img_s[masked_img_s<np.amax(mask2)]
    
    df1 = pd.DataFrame({
        "masked_img_h" : masked_img_h,
        "masked_img_s" : masked_img_s
    })

    # ヒストグラムをプロット

    sns.jointplot(x = "masked_img_h", y = "masked_img_s", kind = 'hist', color = 'magenta', xlim=(0,180),ylim=(0,255),data=df1)
    #plt.scatter(x=masked_img_h,y=masked_img_s,s=1,alpha=0.3,marker='x',color='g')
    plt.xlabel('hue')
    plt.ylabel('satuation')
    plt.xlim(0,180)
    plt.ylim(0,255)
    plt.grid()
    plt.savefig(output_path)
    plt.clf()

    temp_img_h = masked_img_h.astype(np.float32)
    #print(np.amax(temp_img_h))
    #temp_img_h = temp_img_h[temp_img_h<=179]
    temp_img_s = masked_img_s.astype(np.float32)
    #print(np.amax(temp_img_s))
    #temp_img_s = temp_img_s[temp_img_s<=255]

    h_tyokkou = temp_img_s * np.cos(temp_img_h*(np.pi)/90)
    s_tyokkou = temp_img_s * np.sin(temp_img_h*(np.pi)/90)

    df1 = pd.DataFrame({
        "h_tyokkou" : h_tyokkou,
        "s_tyokkou" : s_tyokkou
    })




def plot_masked_histogram(img_h,img_v,mask, output_path, output_path2):
    #img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #img_h = img_hsv[:,:,0]
    #img_s = img_hsv[:,:,1]
    #img_v = img_hsv[:,:,2]
    print(img_h.dtype)
    print(img_v.dtype)
    
        # マスクを適用して、マスクされた部分の画像を得る
    mask2 = cv2.bitwise_not(mask)
    mask2 = mask2.astype(np.int32)
    mask2 = mask2 *10

    masked_img_h = cv2.bitwise_and(img_h, img_h, mask=mask)
    #print(np.amax(masked_img_h))
    masked_img_h = masked_img_h.astype(np.int32)
    masked_img_h += mask2
    masked_img_h = np.ravel(masked_img_h)
    masked_img_h = masked_img_h[masked_img_h<np.amax(mask2)]
    #print(np.amax(masked_img_h), np.amax(mask2))

    masked_img_v = cv2.bitwise_and(img_v, img_v, mask=mask)
    masked_img_v = masked_img_v.astype(np.int32)
    masked_img_v += mask2
    masked_img_v = np.ravel(masked_img_v)
    masked_img_v = masked_img_v[masked_img_v<np.amax(mask2)]
    
    df2 = pd.DataFrame({
        "masked_img_h" : masked_img_h,
        "masked_img_v" : masked_img_v
    })

    
    # ヒストグラムをプロット
    sns.jointplot(x = "masked_img_h", y = "masked_img_v", kind = 'hist', color = 'magenta', xlim=(0,180),ylim=(0,255),data=df1)
    #plt.scatter(x=masked_img_h,y=masked_img_s,s=1,alpha=0.3,marker='x',color='g')
    plt.xlabel('hue')
    plt.ylabel('value')
    plt.xlim(0,180)
    plt.ylim(0,255)
    plt.grid()
    plt.savefig(output_path)
    plt.clf()

    temp_img_h = masked_img_h.astype(np.float32)
    #print(np.amax(temp_img_h))
    #temp_img_h = temp_img_h[temp_img_h<=179]
    temp_img_v = masked_img_v.astype(np.float32)
    #print(np.amax(temp_img_s))
    #temp_img_s = temp_img_s[temp_img_s<=255]

    h_tyokkou = temp_img_v * np.cos(temp_img_h*(np.pi)/90)
    v_tyokkou = temp_img_v * np.sin(temp_img_h*(np.pi)/90)

    df2 = pd.DataFrame({
        "h_tyokkou" : h_tyokkou,
        "s_tyokkou" : s_tyokkou
    })


    # ヒストグラムをプロット
    sns.jointplot(x = "h_tyokkou", y = "s_tyokkou", kind = 'hist', color = 'magenta', xlim=(-255,255),ylim=(-255,255), data=df2)
    #plt.scatter(x=h_tyokkou,y=s_tyokkou,s=1,alpha=0.3,marker='x',color='r')
    plt.xlabel('hue')
    plt.ylabel('satuation')
    plt.xlim(-255,255)
    plt.ylim(-255,255)
    plt.grid()
    plt.savefig(output_path2)
    plt.clf()

    # ヒストグラムをプロット
    sns.jointplot(x = "h_tyokkou", y = "v_tyokkou", kind = 'hist', color = 'magenta', xlim=(-255,255),ylim=(-255,255), data=df2)
    #plt.scatter(x=h_tyokkou,y=s_tyokkou,s=1,alpha=0.3,marker='x',color='r')
    plt.xlabel('hue')
    plt.ylabel('value')
    plt.xlim(-255,255)
    plt.ylim(-255,255)
    plt.grid()
    plt.savefig(output_path2)
    plt.clf()



for target in targets:
    my_makedirs("./output/3-hsv_analysis/1-hs_2d_histogram/"+image_target+"/"+str(target))
    my_makedirs("./output/3-hsv_analysis/1-hs_2d_histogram/"+image_target2+"/"+str(target))

    print("target", target)
    path = "./output/anotation_split_img/"+target+"/"
    directories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    roop_num = len(directories) + 1

    for num in tqdm(range(1,roop_num)):
        input_path1 = "./output/fish_split_img/"+target+"/"+str(num)+"/img_crop.tif"
        mask_path =  "./output/anotation_split_img/"+target+"/"+str(num)+"/img_crop.tif"
        img = cv2.imread(input_path1)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_h = hsv_img[:,:,0]
        img_s = hsv_img[:,:,1]
        img_v = hsv_img[:,:,2]
        print(np.amax(img_h))
        mask_img = cv2.imread(mask_path)
        _, mask = make_img_mask(img_h, mask_img, 8)
        mask = mask*255
        #tifffile.imsave("./img.tif", mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #空間を消す
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        #tifffile.imsave("./img2.tif", mask)

        output_path1 = "./output/3-hsv_analysis/1-hs_2d_histogram/"+image_target+"/"+str(target)+"/"+str(num)+".png"
        output_path2 = "./output/3-hsv_analysis/1-hs_2d_histogram/"+image_target2+"/"+str(target)+"/"+str(num)+".png"
        plot_masked_histogram(img_h,img_s, mask, output_path1, output_path2)
        

        output_path1 = "./output/3-hsv_analysis/1-hs_2d_histogram/"+image_target+"/"+str(target)+"/"+str(num)+".png"
        output_path3 = "./output/3-hsv_analysis/1-hs_2d_histogram/"+image_target3+"/"+str(target)+"/"+str(num)+".png"
        plot_masked_histogram(img_h,img_v, mask, output_path1, output_path3)
        

