import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets, models
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import cv2
from dog_breed_dataloader import get_labels

def detect_dog_breed(img_path):
    

    crop_size=(224,224)
    img = cv2.imread(image_path)
    H,W,C = img.shape
    re_img = cv2.resize(img, crop_size)
    im_tfs = transforms.Compose([
        transforms.ToTensor(),
        #tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = im_tfs(re_img)
    batch = img_tensor.unsqueeze(0)
    test = Variable(batch).to(device)
    outputs = net(test)
    label_pred = outputs.max(1)[1].squeeze().cpu().data.numpy()
    plt_show(img)
    print(class_name[label_pred])
    
def plt_show(image):
    
    if(len(image.shape)==3):
        image = image[:,:,::-1]
    elif(len(image.shape)==2):
        image = image
    plt.imshow(image,cmap ='gray')
    plt.xticks([])
    plt.yticks([])
    #plt.savefig("test/"+titles[i]+".jpg")
    plt.show()
    
    
if __name__ == '__main__':
    
    print(torch.cuda.get_device_properties(0))
    
    root_label = "./save_txt/Labels.txt"
    class_name = get_labels(root_label)
    num_class = len(class_name)
    
    net = models.resnet50(pretrained=False)

    net.fc = torch.nn.Sequential(torch.nn.Linear(2048, 1000),
                                 torch.nn.Linear(1000, 500),
                                 torch.nn.Linear(500, 100),
                                 torch.nn.Linear(100, num_class),)
    print(net)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = net.to(device)
    net.eval()
    net.load_state_dict(torch.load('save_train/save_net_best_150.150_0.0484375.pkl'))
    
    image_path="./test/dog_5.jpg"
    detect_dog_breed(image_path)
    