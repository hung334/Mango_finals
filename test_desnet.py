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
    torch.backends.cudnn.benchmark = True
    
    dev_path=r"./datasets/unet_crop_img_Dev"
    
    val_data = datasets.ImageFolder(dev_path,transform=transforms.Compose([
        transforms.Resize(size=(256,256)),#(h,w)
        transforms.ToTensor()
    ]))

    #train_Loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    val_Loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=4,shuffle=False,num_workers=0)
    #*******************************************************************************************
    
    net = models.densenet121(pretrained=False)
    
    net.classifier = torch.nn.Sequential(torch.nn.Linear(1024, 1000),
                                torch.nn.Linear(1000, 500),
                                 torch.nn.Linear(500, 100),
                                 torch.nn.Linear(100, 3),)
    print(net)
    
    loss_func = torch.nn.CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = net.to(device)
    
    net.load_state_dict(torch.load('save_train_densenet/save_net_best_154.1000_0.0125.pkl'))
    
    training_loss,validation_loss=[],[]
    training_accuracy,validation_accuracy =[],[]
    best_acc,best_epoch=0,0
    
    with torch.no_grad():
            net.eval()
            val_loss_reg,total_val,step_count,correct_val =0.0, 0,0,0
            for step, (batch_x,label_y) in enumerate(val_Loader):
                print("({}/{})".format(step+1,len(val_Loader)))
                #h,w,c= batch_x.shape[1],batch_x.shape[2],batch_x.shape[3]
                #input_shape = (-1,c,h,w)
                #val = Variable(batch_x.view(input_shape)).to(device)
                val = Variable(batch_x).to(device)
                labels = Variable(label_y).to(device)
                outputs = net(val)
                val_loss = loss_func(outputs,labels)
    
                val_loss_reg +=val_loss.cpu().data
                step_count += 1
                
                ans=torch.max(outputs,1)[1].squeeze()
                correct_val += (ans.cpu() == labels.cpu()).float().sum()
            
            val_accuracy = 100 * correct_val / float(len(val_data))
            validation_accuracy.append(val_accuracy)
            
            avg_val_loss = val_loss_reg/len(val_Loader)#step_count
            validation_loss.append(avg_val_loss)
            print("{}[Avg_val_loss:{:1.3f} Acc_val:{:3.2f}]".format(("*"*30),avg_val_loss,val_accuracy))
            #if(val_accuracy>=best_acc):
                #best_acc=val_accuracy
                #torch.save(net.state_dict(), '{}/save_net_best_{}.{}_{}.pkl'.format(save_file,epoch+1,EPOCHS,LR))
                #t_fi= open('{}/Best_Acc.txt'.format(save_file),'w')
                #t_fi.writelines("epoch:{} acc:{}\n".format(epoch+1,best_acc))
                #t_fi.close()
            torch.cuda.empty_cache()
        
    #print("best_acc:{}%".format(best_acc))