import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as tfs
import random
from torch.utils.data import DataLoader,Dataset
import os
from Unet import *

classes = ['background','mango']
color_map = [[0,0,0],[0,0,128]] #np.asarray([[0,0,0],[0,0,128]])

cm2lbl = np.zeros(256**3) # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
for i,cm in enumerate(color_map):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i # 建立索引

def make_Mango_txt(root,val_threshold):
    
    
    save_file="save_txt"
    if not os.path.isdir(save_file):
        os.mkdir(save_file)
    
    t_fi= open('{}/train_Images.txt'.format(save_file),'w')
    
    v_fi= open('{}/val_Images.txt'.format(save_file),'w')
    
    
    p_total=[]
    f_total=[]
    for file in os.listdir(root):
        print(file)
        prefix=file.split("_")[0]
        
        reg_total=[]
        for data in os.listdir(os.path.join(root,file)):
            if(data == "img.png"):
                #print(os.path.join(root,file,data))
                reg_total.append(os.path.join(root,file,data))
            if(data == "label.png"):
                #print(os.path.join(root,file,data))
                reg_total.append(os.path.join(root,file,data))
        if(prefix=="Preliminary"):
            p_total.append(reg_total)
        elif(prefix=="finals"):
            f_total.append(reg_total)
    
    random.shuffle(f_total)#打亂
    random.shuffle(p_total)#打亂
    
    for i,data in enumerate(f_total):
        if(i<len(f_total)*val_threshold):
            t_fi.writelines("{} {}\n".format(data[0],data[1]))
        else:
            v_fi.writelines("{} {}\n".format(data[0],data[1]))
    for i,data in enumerate(p_total):
        if(i<len(p_total)*val_threshold):
            t_fi.writelines("{} {}\n".format(data[0],data[1]))
        else:
            v_fi.writelines("{} {}\n".format(data[0],data[1]))

    
    t_fi.close()
    v_fi.close()

class Mango_segmentation(Dataset): 
    def __init__(self,root, datatxt, transform=None, target_transform=None,crop_size=(256,256)): 
        #super(MyDataset,self).__init__()
        self.crop_size = crop_size
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
            #img = cv2.resize(img, self.crop_size)
            #label = cv2.resize(label, self.crop_size,interpolation=cv2.INTER_NEAREST)
            #img = Image.open(fn).convert('RGB') 
            if self.transform is not None:
                img, label = self.transform(img, label, self.crop_size) 
            return img,label  

    def __len__(self): 
        return len(self.imgs)

def rotate(image, angle, center=None, scale=1.0,flag=None):
    
    (h, w) = image.shape[:2]
 
    if center is None:
        center = (w / 2, h / 2)
 
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    if flag is None:
        rotated = cv2.warpAffine(image, M, (w, h))
    else:
        rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_NEAREST)
 
    return rotated

def img_transforms(img, label, crop_size):
    #im, label = rand_crop(im, label, *crop_size)
    re_img = cv2.resize(img, crop_size)
    re_label = cv2.resize(label, crop_size,interpolation=cv2.INTER_NEAREST)
    
    random_angle=random.randint(0,360)
    ro_img = rotate(re_img, random_angle)
    ro_label = rotate(re_label, random_angle,flag=1)
    #print(random_angle,np.unique(ro_label))
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        #tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    im = im_tfs(ro_img)
    label = image2label(ro_label)
    label = torch.from_numpy(label)
    return im, label

def image2label(im):
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64') # 根据索引得到 label 矩阵

def labelimage(im):
    im=np.array(im,dtype=int)
    print(im.shape)            
    return color_map[im]

def plt_show(image):
    
    image = image[:,:,::-1]
    plt.imshow(image,cmap ='gray')
    plt.xticks([])
    plt.yticks([])
    #plt.savefig("test/"+titles[i]+".jpg")
    plt.show()

if __name__ == '__main__':
    
    
    net = ResNetUNet(n_class=2)
    
    val_threshold=0.9
    
    root = "datasets/Total"#"labelme_mango"
    make_Mango_txt(root,val_threshold)
    
    # train_data = Mango_segmentation(root,"train_Images.txt",transform=img_transforms)
    # train_dataloader = DataLoader(train_data,batch_size=4,shuffle=True,num_workers=0)
    
    # for step,(batch_x,batch_y) in enumerate(train_dataloader):
        
    #     output = net(batch_x)
    #     print(step,batch_x.size(),batch_y)
    
    # color_map = np.asarray([[0,0,0],[0,0,128]])
    
    # img = cv2.imread("./test/img.png")
    # plt_show(img)
    # label = cv2.imread("./test/label.png")
    # plt_show(label)
    
    # h,w,c = img.shape
    
    # a,b = img_transforms(img, label, (224,224))
    
    # re_img = cv2.resize(img, (224, 224))
    # plt_show(re_img)
    
    # re_label = cv2.resize(label, (224, 224),interpolation=cv2.INTER_NEAREST)
    # plt_show(re_label)
    
    
    # re_label = cv2.resize(label, (w, h),interpolation=cv2.INTER_NEAREST)
    # plt_show(re_label)
    #print(np.unique(re_label))
    pass
















# import torch.utils.data as data

# class DataLoaderSegmentation(data.Dataset):
#     def __init__(self, folder_path):
#         super(DataLoaderSegmentation, self).__init__()
#         self.img_files = glob.glob(os.path.join(folder_path,'image','*.png')
#         self.mask_files = []
#         for img_path in img_files:
#              self.mask_files.append(os.path.join(folder_path,'mask',os.path.basename(img_path)) 

#     def __getitem__(self, index):
#             img_path = self.img_files[index]
#             mask_path = self.mask_files[index]
#             data = use opencv or pil read image using img_path
#             label =use opencv or pil read label  using mask_path
#             return torch.from_numpy(data).float(), torch.from_numpy(label).float()

#     def __len__(self):
#         return len(self.img_files)