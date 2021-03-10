from torch.utils.data import DataLoader,Dataset
from torchvision import transforms,utils
from torchvision import  datasets
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import random

# def make_mango_txt(root,val_threshold):
    
#     save_file="save_txt"
#     if not os.path.isdir(save_file):
#         os.mkdir(save_file)
    
#     ft= open('{}/train.txt'.format(save_file),'w')
#     fv= open('{}/val.txt'.format(save_file),'w')
#     fl = open('{}/Labels.txt'.format(save_file),'w')
    
#     label_count=0
#     for label in os.listdir(root):
#         print(label)
#         fl.writelines("{}\n".format(label))
#         file_img=os.listdir(os.path.join(root,label))
#         random.shuffle(file_img)
#         for i,image_path in enumerate(file_img):
#              print(os.path.join(root,label,image_path),label_count)
#              if(i<len(file_img)*val_threshold):
#                  ft.writelines("{} {}\n".format(os.path.join(root,label,image_path),label_count))
#              else:
#                  fv.writelines("{} {}\n".format(os.path.join(root,label,image_path),label_count))
#         label_count+=1
    
#     ft.close()
#     fv.close()
#     fl.close()
    

class Mango_Data(Dataset): 
    def __init__(self,root, datatxt, transform=None, target_transform=None): 
        #super(MyDataset,self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        fh = open(datatxt, 'r') 
        imgs = []
        for line in fh :
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))#圖片，label
        self.imgs = imgs

        
    def __getitem__(self, index):
        
            fn, label = self.imgs[index] 
            img = Image.open(fn).convert('RGB') 
            if self.transform is not None:
                img = self.transform(img) 
            return img,label  

    def __len__(self): 
        return len(self.imgs)

def get_labels(root):
    f = open(root, 'r') 
    labels = []
    for line in f :
            line = line.rstrip()
            #print(line)
            labels.append(line)
    return labels
    
if __name__ == '__main__':
    
    
    val_threshold = 0.9
    root = "./datasets/crop_img_Dev"
    root_label = "./save_txt/Labels.txt"
    
    #make_mango_txt(root,val_threshold)
    
    path=r"D:\label_mango\python_finals\datasets\crop_img_Dev"
    
    train_data = datasets.ImageFolder(path,transform=transforms.Compose([
        transforms.Resize(size=(256,256)),#(h,w)
        transforms.RandomCrop(size=(256,256), padding=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.5,saturation=0.5),
        # transforms.RandomHorizontalFlip(p=0.5),#依據p概率水平翻轉
        # transforms.RandomVerticalFlip(p=0.5),#依據p概率垂直翻轉
        transforms.RandomRotation((-45,45)),#隨機角度旋轉
        transforms.ToTensor()
    ]))
    print(train_data.classes)#获取标签
    print(train_data.class_to_idx)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=128,shuffle=True,num_workers=6)
    
    print(len(train_loader))
    for i_batch, img in enumerate(train_loader):
        #if i_batch == 0:
            print(img[1])   #标签转化为编码
            fig = plt.figure()
            grid = utils.make_grid(img[0])
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.show()
    
    #labels = get_labels(root_label)

    
    # trc = transforms.Compose([ transforms.Resize(size=(224,224)),transforms.ToTensor()])
    
    # train_data = Dog_breed_Data(root,"./save_txt/train.txt",transform=trc)
    # train_dataloader = DataLoader(train_data,batch_size=4,shuffle=True,num_workers=8)
    
    # for step,(batch_x,batch_y) in enumerate(train_dataloader):
        
    #      print(step,batch_x.size(),batch_y)


    
    pass