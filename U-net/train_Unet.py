import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets, models
#from data_augmentation_2 import get_train,get_dev
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from Unet import *
from unet.unet_model import *
from main import *

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def acc_plt_show(num_epochs,training_accuracy,validation_accuracy,LR,save_file):
    plt.plot(range(num_epochs), training_accuracy, 'b-', label='Training_accuracy')
    plt.plot(range(num_epochs), validation_accuracy, 'g-', label='Validation_accuracy')
    plt.title('Train & Val accuracy,epoch:{}'.format(num_epochs))
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("{}/{}_acc.jpg".format(save_file,LR))
    plt.show()

def loss_plt_show(num_epochs,training_loss,validation_loss,LR,save_file):
    plt.plot(range(num_epochs), training_loss, 'b-', label='Training_loss')
    plt.plot(range(num_epochs), validation_loss, 'g-', label='validation_loss')
    plt.title('Train & Val loss,epoch:{}'.format(num_epochs))
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("{}/{}_loss.jpg".format(save_file,LR))
    plt.show()
    
def plt_show(image):
    
    plt.imshow(image,cmap ='gray')
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig("test/"+titles[i]+".jpg")
    plt.show()
    
    
if __name__ == '__main__':
    
    print(torch.cuda.get_device_properties(0))
    
    root = "labelme_mango"
    train_data = Mango_segmentation(root,"train_Images.txt",transform=img_transforms)
    val_data = Mango_segmentation(root,"val_Images.txt",transform=img_transforms)
    
    print("train_image_label ok")
    
    BATCH_SIZE = 20
    LR =0.0125#(0.62 / 1024 * BATCH_SIZE)#0.01
    EPOCHS = 300
    num_class = 2#len(class_name)
    num_classes = 2
    
    train_Loader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
    val_Loader = DataLoader(dataset=val_data,batch_size=4,shuffle=True)
    print("train and val loader is ok")
    #del train_data
    
    
    gc.collect()#回收
    
    net =ResNetUNet(n_class=2) #UNet(3,2)
    
    print(net)
    
    #optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.5, patience=20, verbose=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = net.to(device)
    #net.train()
    
    #net.load_state_dict(torch.load('save/save_resnet152_best_78.250_0.01.pkl'))
    
    loss_func = torch.nn.CrossEntropyLoss()
    
    training_loss,validation_loss=[],[]
    training_accuracy,validation_accuracy =[],[]
    
    best_acc,best_epoch=0,0
    best_acc_cls,best_MIoU,best_FWIoU=0,0,0
    
    save_file="save"
    if not os.path.isdir(save_file):
        os.mkdir(save_file)
    
    for epoch in range(EPOCHS):
        net.train()
        train_loss_reg,total_train,step_count,correct_train =0.0, 0,0,0
        train_acc = 0
        train_acc_cls = 0
        train_mean_iu = 0
        train_fwavacc = 0

        for step, (batch_x,label_y) in enumerate(train_Loader):
            
            train = Variable(batch_x).to(device)
            labels = Variable(label_y).to(device)
            outputs = net(train)
            train_loss = loss_func(outputs,labels)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            train_loss_reg +=train_loss.cpu().data
            step_count += 1
            
            label_pred = outputs.max(dim=1)[1].data.cpu().numpy()
            label_true = labels.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
                train_acc += acc
                train_acc_cls += acc_cls
                train_mean_iu += mean_iu
                train_fwavacc += fwavacc
            
            #ans=torch.max(outputs,1)[1].squeeze()
            #total_train += len(labels)
            #correct_train += (ans.cpu() == labels.cpu()).float().sum()
            
            print("Epoch:{}/{} Step:{}/{} Train_loss:{:1.3f} ".format(epoch+1,EPOCHS,step+1,len(train_Loader),train_loss))
            
        train_accuracy = 100 * train_acc / len(train_data)
        training_accuracy.append(train_accuracy)

        avg_train_loss = train_loss_reg/len(train_Loader)
        training_loss.append(avg_train_loss)
        
        print("{}[Epoch:{}/{}  Avg_train_loss:{:1.3f} ]".format(("*"*30),epoch+1,EPOCHS,avg_train_loss))#loss.item()
        print("{}[Train_Acc:{:.5f},Train_MIoU:{:.5f}]".format(("*"*30),train_acc / len(train_data), train_mean_iu / len(train_data)))
        print("{}[Train_Acc_cls:{:.5f},Train_FWIoU:{:.5f}]".format(("*"*30),train_acc_cls / len(train_data),train_fwavacc / len(train_data)))
        
        with torch.no_grad():
            net.eval()
            val_loss_reg,total_val,step_count,correct_val =0.0, 0,0,0
            eval_acc = 0
            eval_acc_cls = 0
            eval_mean_iu = 0
            eval_fwavacc = 0
            for step, (batch_x,label_y) in enumerate(val_Loader):
                val = Variable(batch_x).to(device)
                labels = Variable(label_y).to(device)
                outputs = net(val)
                val_loss = loss_func(outputs,labels)
    
                val_loss_reg +=val_loss.cpu().data
                step_count += 1
                
                label_pred = outputs.max(dim=1)[1].data.cpu().numpy()
                label_true = labels.data.cpu().numpy()
                for lbt, lbp in zip(label_true, label_pred):
                    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
                    eval_acc += acc
                    eval_acc_cls += acc_cls
                    eval_mean_iu += mean_iu
                    eval_fwavacc += fwavacc
            
            val_accuracy = 100 * eval_acc / len(val_data)
            validation_accuracy.append(val_accuracy)
            
            avg_val_loss = val_loss_reg/len(val_Loader)
            validation_loss.append(avg_val_loss)
            print("{}[Epoch:{}/{}  Avg_val_loss:{:1.3f} ]".format(("*"*30),epoch+1,EPOCHS,avg_val_loss))
            print("{}[Val_Acc:{:.5f},Val_Mean IU:{:.5f}]".format(("*"*30),eval_acc / len(val_data), eval_mean_iu / len(val_data)))
            print("{}[Val_Acc_cls:{:.5f},Val_FWIoU:{:.5f}]".format(("*"*30),eval_acc_cls / len(val_data),eval_fwavacc/len(val_data)))
                  
            val_acc_cls = eval_acc_cls / len(val_data) *100
            val_MIoU = eval_mean_iu / len(val_data) *100
            val_FWIoU = eval_fwavacc/len(val_data)*100
            

            if((val_accuracy>=best_acc)and(val_MIoU>=best_MIoU)and(val_acc_cls>=best_acc_cls)and(val_FWIoU>=best_FWIoU)):
                best_acc=val_accuracy
                best_MIoU=val_MIoU
                best_acc_cls=val_acc_cls
                best_FWIoU=val_FWIoU
                
                torch.save(net.state_dict(), '{}/save_unet_best_{}.{}_{}.pkl'.format(save_file,epoch+1,EPOCHS,LR))
            torch.cuda.empty_cache()
                
        scheduler.step(val_accuracy)
        lr = optimizer.param_groups[0]['lr']
        print("best_acc:{}，best_acc_cls:{}".format(best_acc,best_acc_cls))
        print("best_MIoU:{}，best_FWIoU:{}".format(best_MIoU,best_FWIoU))
        print("LR:{}".format(lr))
        loss_plt_show(epoch+1,training_loss,validation_loss,LR,save_file)
        acc_plt_show(epoch+1,training_accuracy,validation_accuracy,LR,save_file)
    #EPOCHS=35
    #loss_plt_show(EPOCHS,training_loss,validation_loss,LR,save_file)
    #acc_plt_show(EPOCHS,training_accuracy,validation_accuracy,LR,save_file)
    torch.save(net.state_dict(), '{}/save_unet_{}_{}.pkl'.format(save_file,EPOCHS,LR))
    