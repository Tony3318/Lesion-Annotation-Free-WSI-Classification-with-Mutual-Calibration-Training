import time
import os
import openslide
import torch
import numpy as np
import cv2
from torchvision import  models, transforms
import numpy as np
import time
import pandas as pd
from collections import Counter
import torch.nn.functional as F
from torch import nn

# for i in range(2,4):
i = 5 # level
print("Step:",i)
csv_path = "./skin_data_info_split.csv"
df = pd.read_csv(csv_path)  #data_info

category_values = sorted(set(df["category"]))  # 例如 "class"
print(category_values)
classes = {item.split('_', 1)[1]: item for item in category_values}
print(classes)
base_dir = f"./v622_level_5/"
sets = ["train", "val"]
# classes = {"malignant": "1_Malignant", "normal": "0_Normal"}

for s in sets:
    for c in classes.values():
        f_path = os.path.join(base_dir, s, c)
        os.makedirs(f_path, exist_ok=True)


new_dir  = f'./v622_level_5_view/'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)
wsi_data_dir = "D:/wsi_data/"

img_id = 0
img_total_cnt = 0
ss  = 256

###################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running device:', device)



data_dir= f'./v622_level_5/val/'
class_list = sorted(os.listdir(data_dir))
class_name2idx = { name:idx for idx,name in enumerate(class_list) }
class_idx2name = { idx:name for idx,name in enumerate(class_list) }


for index, row in df.iterrows():    #apply等等試試
    wsi_name = row['wsi_name']
    gt = row['category'].split('_', 1)[1]
    datasets = row['tag']
    # idx_gt = row['category'] if classes[gt][:1] == '1' else classes[gt][1:]
    true_gt = row['category']

    img_total_cnt += 1
    # if img_total_cnt % 2 == 0:#odd
    #     continue
    
    # print(base_dir, datasets, classes[gt])
    save_dir = os.path.join(base_dir, datasets, classes[gt])
    # other_dir = os.path.join(base_dir, datasets, '15_other')
    wsi_path = os.path.join(wsi_data_dir, classes[gt], wsi_name)


    if not os.path.exists(wsi_path):
        print(wsi_path)
        print("Path empty/n")
        continue

    #####################讀黨#############################################
    wsi_slide_img = openslide.OpenSlide(wsi_path)
    level_cnt = wsi_slide_img.level_count
    thumb_level = level_cnt - 1
    
    n, m = wsi_slide_img.level_dimensions[thumb_level]
    print(n,m)
    patch_bgr = np.array(wsi_slide_img.read_region((0,0), thumb_level, (n, m)))[:,:,[2,1,0]].copy()
    bg_bgr_arr = np.array(bg_bgr_list)
    h, w, _ = patch_bgr.shape
    flat_img = patch_bgr.reshape(-1, 3)
    # 建立背景遮罩：RGB 全部大於 230 → 背景
    # bg_mask = np.all(patch_bgr > 230 or patch_bgr < 100, axis=2)  # shape: (h, w)
    # bg_mask = np.all(np.logical_or(patch_bgr > 230, patch_bgr < 100), axis=2)
    # (三通道都大於 230) OR (三通道都小於 100)
    mask_extreme = np.all(patch_bgr > 245, axis=2) | np.all(patch_bgr < 100, axis=2)
    match_bgr = np.isin(flat_img.view([('', flat_img.dtype)] * 3),
                    bg_bgr_arr.view([('', bg_bgr_arr.dtype)] * 3)).all(axis=1)
    mask_from_list = match_bgr.reshape(h, w)
    bg_mask = mask_from_list | mask_extreme
    # 建立一張新的 patch（複製原圖）
    BW = (~bg_mask).astype(np.uint8) * 255
    
    # 將背景區設為黑色（0, 0, 0），或你要的顏色
    # BW[bg_mask] = [0, 0, 0]  # 或設為 [255, 255, 255] 為白背景
    patch = patch_bgr
    # gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    
    # T, BW = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    ker = np.array([[0,1,0],[1,1,1],[0,1,0]]).astype(np.uint8)
    BW = cv2.dilate(cv2.erode(BW, ker), ker)
    
    ########################把長寬比過大的刪除############################
    Contours, hirearchy = cv2.findContours(BW, 0, 2)
    # BW = np.zeros((BW.shape[0], BW.shape[1]), dtype=np.uint8)
    for CNT in Contours:
        area = cv2.contourArea(CNT)
        x, y, w, h = cv2.boundingRect(CNT)
        if area < 20 and 0.1 > w/h > 10:
            cv2.drawContours(BW, [CNT], 0, 0, -1)

    ####################################################################
    white_points = np.argwhere(BW > 0)  # 返回 (y, x) 座標
    white_points_list = [tuple(point) for point in white_points]

    BW = cv2.cvtColor(BW, cv2.COLOR_GRAY2BGR)
    mask = BW.copy()
    if len(white_points_list) == 0:
        print(f"skip_{wsi_name}")
        continue


    max_Sample = Sampling_ls[classes[gt]]
    cnt_Sample = 0
    # total_cnt = 0
    ss  = 256

    scale_factor = wsi_slide_img.level_downsamples[thumb_level] #* 2
    # print(scale_factor)
    while(max_Sample > 0):
        if len(white_points_list) == 0:
            break
        
        idx = np.random.randint(0, len(white_points_list))
        y_, x_ = white_points_list[idx]
        white_points_list.remove((y_, x_))
        if y_ + 31 > m or x_ + 31 > n:
            # print(123)
            continue

        BW[y_, x_, :] = (0,255,255) 
        x = int(x_ * scale_factor)
        y = int(y_ * scale_factor)

        sample = np.array(wsi_slide_img.read_region((x, y), i, (ss, ss)))[:,:,[2,1,0]]
        # gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)  # 轉換為灰階
        # Thresh, map_BW = cv2.threshold(gray_sample, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Contours, hirearchy = cv2.findContours(map_BW, 0, 2)
        # if len(Contours) < 50:
        #     print(1)
        #     continue
        ######################################
        # 檢查 32x32 區塊的黑色像素數
        check_block_size = 32
        black_threshold = 400
        block = BW[y_:y_+check_block_size, x_:x_+check_block_size, 0]  # 用第一個 channel（灰階一樣）
        black_pixels = np.sum(block == 0)
        if black_pixels > black_threshold:
            print(f"Too many black pixels: {black_pixels}")
            continue
        ######################################
        radius = 7
        white_points_set = set(white_points_list)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                point = (y_ + dy, x_ + dx)
                if point in white_points_set:
                    white_points_set.remove(point)
        white_points_list = list(white_points_set)
        ########################################

        save_path = os.path.join(save_dir, f"{img_total_cnt}_{wsi_name}_({x},{y}).png")
        max_Sample -= 1
        cv2.imwrite(save_path, sample)
        BW[y_, x_, :] = (0,0,255) 
        patch[y_, x_, :] = (0,0,255) 

        if (max_Sample == 0):
            break

    combined_img = np.hstack((patch, BW))
    cv2.imwrite(new_dir+f'{img_total_cnt}_{wsi_name}'+'.png',combined_img)


