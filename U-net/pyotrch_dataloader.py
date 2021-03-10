from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import cv2
import random


def make_Mango_txt(root,val_threshold):
    
    
    t_fi= open('train_Images.txt','w')
    
    v_fi= open('val_Images.txt','w')
    
    
    total=[]
    for file in os.listdir(root):
        print(file)
        reg_total=[]
        for data in os.listdir(os.path.join(root,file)):
            if(data == "img.png"):
                print(os.path.join(root,file,data))
                reg_total.append(os.path.join(root,file,data))
            if(data == "label.png"):
                print(os.path.join(root,file,data))
                reg_total.append(os.path.join(root,file,data))
        total.append(reg_total)
    random.shuffle(total)#打亂
    for i,data in enumerate(total):
        if(i<len(total)*val_threshold):
            t_fi.writelines("{} {}\n".format(data[0],data[1]))
        else:
            v_fi.writelines("{} {}\n".format(data[0],data[1]))

    
    t_fi.close()
    v_fi.close()

class Mango_segmentation(Dataset): 
    def __init__(self,root, datatxt, transform=None, target_transform=None,crop_size=(224,224)): 
        #super(MyDataset,self).__init__()
        
        self.transform = transform
        self.target_transform = target_transform
        fh = open(datatxt, 'r') 
        imgs = []
        for line in fh :
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],words[1]))#圖片，label
        self.imgs = imgs

    def __getitem__(self, index):
        
            fn, fl = self.imgs[index] 
            img,label = cv2.imread(fn),cv2.imread(fl)
            img = cv2.resize(img, crop_size)
            label = cv2.resize(label, crop_size,interpolation=cv2.INTER_NEAREST)
            #img = Image.open(fn).convert('RGB') 
            if self.transform is not None:
                img, label = self.transform(img, label, self.crop_size) 
            return img,label  

    def __len__(self): 
        return len(self.imgs)



if __name__ == '__main__':
    
    
    val_threshold=0.9
    root = "labelme_mango"
    make_Mango_txt(root,val_threshold)
    
    

    
    
    
    
    #root = "./crop_img_train"
    #make_Mango_txt(root)
    
    
    # trc = transforms.Compose([ transforms.Resize(size=(224,224)),
    #                           transforms.ToTensor()])
    
    train_data = Mango_Data(root,"train_Images.txt",transform=trc)
    # train_dataloader = DataLoader(train_data,batch_size=4,shuffle=True,num_workers=8)
    
    # for step,(batch_x,batch_y) in enumerate(train_dataloader):
        
    #     print(step,batch_x.size(),batch_y)


    
    pass