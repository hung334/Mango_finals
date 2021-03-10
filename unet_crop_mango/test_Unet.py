import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets, models
import torchvision.transforms as tfs
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from Unet import *
import cv2
from PIL import Image


classes = ['background','mango']
color_map = [[0,0,0],[0,0,128]]

# 定义预测函数
cm = np.array(color_map).astype('uint8')

def predict(im, label): # 预测结果
    im = Variable(im.unsqueeze(0)).cuda()
    out = net(im)
    pred = out.max(1)[1].squeeze().cpu().data.numpy()
    pred = cm[pred]
    return pred, cm[label.numpy()]

# 定义旋转rotate函数
def rotate(image, angle, center=None, scale=1.0,flag=None):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
 
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
 
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    if flag is None:
        rotated = cv2.warpAffine(image, M, (w, h))
    else:
        rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_NEAREST)
 
    # 返回旋转后的图像
    return rotated

def detect_mango(save_file,image_path):
    
    list_path = os.path.split(image_path)
    print(image_path)
    crop_size=(256,256)
    img = cv2.imread(image_path)
    H,W,C = img.shape
    re_img = cv2.resize(img, crop_size)
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        #tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = im_tfs(re_img)
    batch = img_tensor.unsqueeze(0)
    test = Variable(batch).to(device)
    outputs = net(test)
    label_pred = outputs.max(1)[1].squeeze().cpu().data.numpy()
    ans = cm[label_pred]
    re_ans = cv2.resize(ans, (W, H),interpolation=cv2.INTER_NEAREST)
    re_g_ans = cv2.cvtColor(re_ans, cv2.COLOR_BGR2GRAY)
    ret,re_thresh1 = cv2.threshold(re_g_ans,15,255,cv2.THRESH_BINARY)
    re_masked = cv2.bitwise_and(img, img, mask=re_thresh1)
    ontours, hierarchy = cv2.findContours(re_thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in ontours:
        (x, y, w, h) = cv2.boundingRect(c)
        if(w>=150 and h>=150):
            crop_img = re_masked[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(save_file,list_path[1]), crop_img)
            
        

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = ResNetUNet(n_class=2).to(device)
    net.load_state_dict(torch.load('save_unet_best.pkl'))
    
    print(net)
    
    train_path=r"C:\Users\AICT-TITAN_RTX\Desktop\chen_hung\mango\python_finals\datasets\crop_img_Train"
    dev_path=r"C:\Users\AICT-TITAN_RTX\Desktop\chen_hung\mango\python_finals\datasets\crop_img_Dev"
    
    save_train_path=r"C:\Users\AICT-TITAN_RTX\Desktop\chen_hung\mango\python_finals\datasets\unet_crop_img_Train"
    save_dev_path=r"C:\Users\AICT-TITAN_RTX\Desktop\chen_hung\mango\python_finals\datasets\unet_crop_img_Dev"
    
    if not os.path.isdir(save_train_path):
        os.mkdir(save_train_path)
    if not os.path.isdir(save_dev_path):
        os.mkdir(save_dev_path)
    
    for class_img in (os.listdir(train_path)):
        save_class_img_path=os.path.join(save_train_path,class_img)
        if not os.path.isdir(save_class_img_path):
            os.mkdir(save_class_img_path)
        for img_path in (os.listdir(os.path.join(train_path,class_img))):
            detect_mango(save_class_img_path,os.path.join(train_path,class_img,img_path))
    
    
    for class_img in (os.listdir(dev_path)):
        save_class_img_path=os.path.join(save_dev_path,class_img)
        if not os.path.isdir(save_class_img_path):
            os.mkdir(save_class_img_path)
        for img_path in (os.listdir(os.path.join(dev_path,class_img))):
            detect_mango(save_class_img_path,os.path.join(dev_path,class_img,img_path))
        
    #for img_path in (os.listdir(file_path)):
        #detect_mango(save_file,os.path.join(file_path,img_path))
    
    # crop_size=(224,224)
    # img = cv2.imread("./test/C_00556.jpg")#("./test/img.png")
    # H,W,C = img.shape
    # plt_show(img)
    # re_img = cv2.resize(img, crop_size)
    # plt_show(re_img)
    
    # im_tfs = tfs.Compose([
    #     tfs.ToTensor(),
    #     #tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    
    # img_tensor = im_tfs(re_img)
    # batch = img_tensor.unsqueeze(0)
    
    # test = Variable(batch).to(device)
    # outputs = net(test)
    
    # label_pred = outputs.max(1)[1].squeeze().cpu().data.numpy()

    # print(np.unique(label_pred))
    
    # ans = cm[label_pred]
    # plt_show(ans)
    
    # #*************************************************************************
    # g_ans = cv2.cvtColor(ans, cv2.COLOR_BGR2GRAY)
    # ret,thresh1 = cv2.threshold(g_ans,15,255,cv2.THRESH_BINARY)
    
    # masked = cv2.bitwise_and(re_img, re_img, mask=thresh1)
    # plt_show(masked)
    
    # ontours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # for c in ontours:
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     cv2.rectangle(masked, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # plt_show(masked)
    
    # #*************************************************************************
    # re_ans = cv2.resize(ans, (W, H),interpolation=cv2.INTER_NEAREST)
    # print(np.unique(re_ans))
    # plt_show(re_ans)
    
    # re_g_ans = cv2.cvtColor(re_ans, cv2.COLOR_BGR2GRAY)
    # ret,re_thresh1 = cv2.threshold(re_g_ans,15,255,cv2.THRESH_BINARY)
    
    # re_masked = cv2.bitwise_and(img, img, mask=re_thresh1)
    # plt_show(re_masked)
    
    # ontours, hierarchy = cv2.findContours(re_thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # for c in ontours:
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     cv2.rectangle(re_masked, (x, y), (x + w, y + h), (0, 255, 0), 5)

    # plt_show(re_masked)
    # #*************************************************************************
    
    #**************************************************************************
    # random_angle=45
    # ro_img = rotate(re_img, random_angle)
    # ro_ans = rotate(ans, random_angle,flag=1)
    # plt_show(ro_img)
    # plt_show(ro_ans)
    # print(np.unique(ro_ans))
    
    # re_masked = cv2.resize(masked, (w, h))
    # plt_show(re_masked)
    # cv2.imwrite('test/output.png',re_masked)
    #*************************************************************************



