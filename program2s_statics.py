import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy.stats as stats
import math
from scipy import stats
import pandas as pd
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from utils import *

targets = ["1","2","3"]
image_target = "s" #h,sを入力
stack = True
statistical_test = True
label = [1,2,3,4,5,6,7,8,9,10]
cm = plt.get_cmap("Spectral")


def plot_masked_histogram(img, mask, output_path, mode):
    if(mode == "h"):
        # マスクを適用して、マスクされた部分の画像を得る
        masked_image = cv2.bitwise_and(img, img, mask=mask)
        #masked_image = img
        # ヒストグラムを計算
        hist_masked = cv2.calcHist([masked_image], [0], mask, [256], [0, 256]) #0とすると拝啓が多くなるので削除
        # ヒストグラムをプロット
        plt.plot(hist_masked, color='black')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.xlim(0,180)
        plt.ylim(0,17000)
        plt.savefig(output_path)
        plt.clf()

    if(mode == "s"):
        # マスクを適用して、マスクされた部分の画像を得る
        masked_image = cv2.bitwise_and(img, img, mask=mask)
        #masked_image = img
        # ヒストグラムを計算
        hist_masked = cv2.calcHist([masked_image], [0], mask, [256], [0, 256])
        # ヒストグラムをプロット
        plt.plot(hist_masked, color='black')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.xlim(0,255)
        plt.ylim(0, 10000)
        plt.savefig(output_path)
        plt.clf() 

    if(mode == "v"):
        # マスクを適用して、マスクされた部分の画像を得る
        masked_image = cv2.bitwise_and(img, img, mask=mask)
        #masked_image = img
        # ヒストグラムを計算
        hist_masked = cv2.calcHist([masked_image], [0], mask, [256], [0, 256])
        # ヒストグラムをプロット
        plt.plot(hist_masked, color='black')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.xlim(0,255)
        plt.ylim(0, 10000)
        plt.savefig(output_path)
        plt.clf()


def plot_stack_masked_histogram(img, mask, mode, num, stack_value):
    if(mode == "h"):
        # マスクを適用して、マスクされた部分の画像を得る
        masked_image = cv2.bitwise_and(img, img, mask=mask)
        #masked_image = img
        # ヒストグラムを計算
        hist_masked = cv2.calcHist([masked_image], [0], mask, [180], [0, 179]) #0とすると拝啓が多くなるので削除
        #print(len(hist_masked))
        # ヒストグラムをプロット
        plt.plot(hist_masked, color=cm((num-1)/10), label=label[num-1], linewidth=1)

        stack_value.append(hist_masked)

    if(mode == "s"):
        # マスクを適用して、マスクされた部分の画像を得る
        masked_image = cv2.bitwise_and(img, img, mask=mask)
        #masked_image = img
        # ヒストグラムを計算
        hist_masked = cv2.calcHist([masked_image], [0], mask, [256], [0, 256])
        #print(len(hist_masked))
        # ヒストグラムをプロット
        plt.plot(hist_masked, color=cm((num-1)/10), label=label[num-1], linewidth=1)

        stack_value.append(hist_masked)
        
    if(mode == "v"):
        # マスクを適用して、マスクされた部分の画像を得る
        masked_image = cv2.bitwise_and(img, img, mask=mask)
        #masked_image = img
        # ヒストグラムを計算
        hist_masked = cv2.calcHist([masked_image], [0], mask, [256], [0, 256])
        #print(len(hist_masked))
        # ヒストグラムをプロット
        plt.plot(hist_masked, color=cm((num-1)/10), label=label[num-1], linewidth=1)

        stack_value.append(hist_masked)

def statistical_test_function(group1, group2, group3):
    group_all = group1 + group2 + group3

    #shapiro-wilk検定(正規性検定) 0.05より大きければ正規分布を仮定
    print("shapiro-wilk検定(正規性検定)")
    s1, p1 = stats.shapiro(group_all)
    print("shapiro-wilk value ", p1)

    if (p1 > 0.05):
        print("\nBartlett検定(等分散検定)")
        s2,p2 = stats.bartlett(group1, group2, group3)
        print("bartlett value ", p2)

        if (p2 > 0.05):
            print("\nANOVA検定(一元配置分散分析)")
            s3, p3 = stats.f_oneway(group1, group2, group3)
            print("ANOVA value ", p3)

            if (p3 < 0.05):
                print("\nTukey検定, 0.05に近いと傾向有")
                group = ['group_1' for _ in range(len(group1))] + ['group_2' for _ in range(len(group2))] + \
                         ['group_3' for _ in range(len(group3))]
                score = group1 + group2 + group3

                print(pairwise_tukeyhsd(score, group))
            else:
                print("None")
        else:
            print("\nKruskal-Wallis検定(3群比較のノンパラメトリック検定)")
            s4,p4 = stats.kruskal(group1, group2, group3)
            print("Kruskal value ", p4)

            if (p4 < 0.05):
                print("\nSteel-Dwass検定(0.05より低いと有意差あり, 0.05に近いと傾向有)")
                group_all2 = [group1, group2, group3]
                print(sp.posthoc_dscf(group_all2))
            else:
                print("None")
    else:
        print("\nKruskal-Wallis検定(3群比較のノンパラメトリック検定)")
        s4,p4 = stats.kruskal(group1, group2, group3)
        print("Kruskal value ", p4)

        if (p4 < 0.05):
            print("\nSteel-Dwass検定(0.05より低いと有意差あり, 0.05に近いと傾向有り)")
            group_all2 = [group1, group2, group3]
            print(sp.posthoc_dscf(group_all2))
        else:
            print("None")


stack_value_all = []
for target in targets:
    stack_value = []
    my_makedirs("./output/3-hsv_analysis/2-histogram_analysis/"+image_target+"/"+str(target))
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
        mask_img = cv2.imread(mask_path)
        _, mask = make_img_mask(img, mask_img, 8)
        mask = mask*255
        #tifffile.imsave("./img.tif", mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #空間を消す
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        #tifffile.imsave("./img2.tif", mask)

        output_path = "./output/3-hsv_analysis/2-histogram_analysis/"+image_target+"/"+str(target)+"/"+str(num)+".png"

        if(image_target == "h"):
            if(stack==True):
                plot_stack_masked_histogram(img_h, mask, image_target, num, stack_value)
            else:
                plot_masked_histogram(img_h, mask, output_path, image_target)
            
        if(image_target == "s"):
            if(stack==True):
                plot_stack_masked_histogram(img_s, mask, image_target, num, stack_value)
            else:
                plot_masked_histogram(img_s, mask, output_path, image_target)
                
        if(image_target == "v"):
            if(stack==True):
                plot_stack_masked_histogram(img_v, mask, image_target, num, stack_value)
            else:
                plot_masked_histogram(img_v, mask, output_path, image_target)
                

    if(stack==True and image_target == "h"):
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.xlim(0,180)
        plt.ylim(0,17000)
        plt.legend(loc="upper right")
        output_path2 = "./output/3-hsv_analysis/2-histogram_analysis/"+image_target+"/"+str(target)+"/all_h.png"
        plt.savefig(output_path2)
        #print(len(stack_value))
        #print(stack_value)
        plt.clf()
        stack_value_all.append(stack_value)

    if(stack==True and image_target == "s"):
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.xlim(0,255)
        plt.ylim(0, 17000)
        plt.legend(loc="upper right")
        output_path2 = "./output/3-hsv_analysis/2-histogram_analysis/"+image_target+"/"+str(target)+"/all_s.png"
        plt.savefig(output_path2)
        plt.clf()
        stack_value_all.append(stack_value)
        
    if(stack==True and image_target == "v"):
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.xlim(0,255)
        plt.ylim(0, 17000)
        plt.legend(loc="upper right")
        output_path2 = "./output/3-hsv_analysis/2-histogram_analysis/"+image_target+"/"+str(target)+"/all_v.png"
        plt.savefig(output_path2)
        plt.clf()
        stack_value_all.append(stack_value)

if(statistical_test == True):
    stack_A = stack_value_all[0]
    stack_B = stack_value_all[1]
    stack_C = stack_value_all[2]

    stack_1_A, stack_2_A, stack_3_A, stack_4_A, stack_5_A, stack_6_A, stack_7_A, stack_8_A,stack_9_A, stack_10_A, stack_11_A, stack_12_A, stack_13_A, stack_14_A, stack_15_A, stack_16_A, stack_17_A, stack_18_A =\
        [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    stack_1_B, stack_2_B, stack_3_B, stack_4_B, stack_5_B, stack_6_B, stack_7_B, stack_8_B,stack_9_B, stack_10_B, stack_11_B, stack_12_B, stack_13_B,stack_14_B, stack_15_B, stack_16_B, stack_17_B, stack_18_B =\
        [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    stack_1_C, stack_2_C , stack_3_C, stack_4_C, stack_5_C, stack_6_C, stack_7_C, stack_8_C,stack_9_C, stack_10_C, stack_11_C, stack_12_C, stack_13_C,stack_14_C, stack_15_C, stack_16_C, stack_17_C, stack_18_C =\
        [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
      
    #データを幅ごとに分けて積算値をとる(幅:とりあえず2:22, 12:32, 22:42, 96:116, 106:126, 116:136を使用)
    for sA, sB in zip(stack_A, stack_B):
        stack_1_A.append(np.sum(sA[2:22]))
        stack_2_A.append(np.sum(sA[12:32]))
        stack_3_A.append(np.sum(sA[22:42]))
        stack_4_A.append(np.sum(sA[96:116]))
        stack_5_A.append(np.sum(sA[106:126]))
        stack_6_A.append(np.sum(sA[116:136]))
        stack_7_A.append(np.sum(sA[126:146]))
        stack_8_A.append(np.sum(sA[136:156]))
        stack_9_A.append(np.sum(sA[146:166]))
        stack_10_A.append(np.sum(sA[156:176]))
        stack_11_A.append(np.sum(sA[166:186]))
        stack_12_A.append(np.sum(sA[176:196]))
        stack_13_A.append(np.sum(sA[186:206]))
        stack_14_A.append(np.sum(sA[196:216]))
        stack_15_A.append(np.sum(sA[206:226]))
        stack_16_A.append(np.sum(sA[216:236]))
        stack_17_A.append(np.sum(sA[226:246]))
        stack_18_A.append(np.sum(sA[0:255]))

        stack_1_B.append(np.sum(sB[2:22]))
        stack_2_B.append(np.sum(sB[12:32]))
        stack_3_B.append(np.sum(sB[22:42]))
        stack_4_B.append(np.sum(sB[96:116]))
        stack_5_B.append(np.sum(sB[106:126]))
        stack_6_B.append(np.sum(sB[116:136]))
        stack_7_B.append(np.sum(sA[126:146]))
        stack_8_B.append(np.sum(sA[136:156]))
        stack_9_B.append(np.sum(sA[146:166]))
        stack_10_B.append(np.sum(sA[156:176]))
        stack_11_B.append(np.sum(sA[166:186]))
        stack_12_B.append(np.sum(sA[176:196]))
        stack_13_B.append(np.sum(sA[186:206]))
        stack_14_B.append(np.sum(sA[196:216]))
        stack_15_B.append(np.sum(sA[206:226]))
        stack_16_B.append(np.sum(sA[216:236]))
        stack_17_B.append(np.sum(sA[226:246]))
        stack_18_B.append(np.sum(sA[0:255]))
    
    for sC in stack_C:
        stack_1_C.append(np.sum(sC[2:22]))
        stack_2_C.append(np.sum(sC[12:32]))
        stack_3_C.append(np.sum(sC[22:42]))
        stack_4_C.append(np.sum(sC[96:116]))
        stack_5_C.append(np.sum(sC[106:126]))
        stack_6_C.append(np.sum(sC[116:136]))
        stack_7_C.append(np.sum(sA[126:146]))
        stack_8_C.append(np.sum(sA[136:156]))
        stack_9_C.append(np.sum(sA[146:166]))
        stack_10_C.append(np.sum(sA[156:176]))
        stack_11_C.append(np.sum(sA[166:186]))
        stack_12_C.append(np.sum(sA[176:196]))
        stack_13_C.append(np.sum(sA[186:206]))
        stack_14_C.append(np.sum(sA[196:216]))
        stack_15_C.append(np.sum(sA[206:226]))
        stack_16_C.append(np.sum(sA[216:236]))
        stack_17_C.append(np.sum(sA[226:246]))
        stack_18_C.append(np.sum(sA[0:255]))

    print("\ngroup1[2:22]")
    statistical_test_function(stack_1_A, stack_1_B, stack_1_C)
    print("\ngroup2[12:32]")
    statistical_test_function(stack_2_A, stack_2_B, stack_2_C)
    print("\ngroup3[22:42]")
    statistical_test_function(stack_3_A, stack_3_B, stack_3_C)
    print("\ngroup4[96:116]")
    statistical_test_function(stack_4_A, stack_4_B, stack_4_C)
    print("\ngroup5[106:126]")
    statistical_test_function(stack_5_A, stack_5_B, stack_5_C)
    print("\ngroup6[116:136]")
    statistical_test_function(stack_6_A, stack_6_B, stack_6_C)
    print("\ngroup7[126:146]")
    statistical_test_function(stack_7_A, stack_7_B, stack_7_C)
    print("\ngroup8[136:156]")
    statistical_test_function(stack_8_A, stack_8_B, stack_8_C)
    print("\ngroup9[146:166]")
    statistical_test_function(stack_9_A, stack_9_B, stack_9_C)
    print("\ngroup10[156:176]")
    statistical_test_function(stack_10_A, stack_10_B, stack_10_C)
    print("\ngroup11[166:186]")
    statistical_test_function(stack_11_A, stack_11_B, stack_11_C)
    print("\ngroup12[176:196]")
    statistical_test_function(stack_12_A, stack_12_B, stack_12_C)
    print("\ngroup13[186:206]")
    statistical_test_function(stack_13_A, stack_13_B, stack_13_C)
    print("\ngroup14[196:216]")
    statistical_test_function(stack_14_A, stack_14_B, stack_14_C)
    print("\ngroup15[206:226]")
    statistical_test_function(stack_15_A, stack_15_B, stack_15_C)
    print("\ngroup16[216:236]")
    statistical_test_function(stack_16_A, stack_16_B, stack_16_C)
    print("\ngroup17[226:246]")
    statistical_test_function(stack_17_A, stack_17_B, stack_17_C)
    print("\ngroup18[0:255]")
    statistical_test_function(stack_18_A, stack_18_B, stack_18_C)
