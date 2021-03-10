import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets, models
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import gc


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
    #plt.ylim(0, 1)
    plt.legend()
    plt.savefig("{}/{}_loss.jpg".format(save_file,LR))
    plt.show()
    
def plt_show(image):
    
    if(len(image.shape)==3):
        image = image[:,:,::-1]
    elif(len(image.shape)==2):
        image = image
    
    plt.imshow(image,cmap ='gray')
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig("test/"+titles[i]+".jpg")
    plt.show()
    
    
if __name__ == '__main__':
    
    
    print(torch.cuda.get_device_properties(0))
    
    root = "./datasets/class_10"
    root_label = "./save_txt/Labels.txt"
    #class_name = get_labels(root_label)
    
    BATCH_SIZE = 80
    LR =0.0125#(0.62 / 1024 * BATCH_SIZE)#0.01
    EPOCHS = 1000
    #num_class = len(class_name)
    
    #******************************************************************************************
    train_path=r"C:\Users\AICT-TITAN_RTX\Desktop\chen_hung\mango\python_finals\datasets\unet_crop_img_Train"
    dev_path=r"C:\Users\AICT-TITAN_RTX\Desktop\chen_hung\mango\python_finals\datasets\unet_crop_img_Dev"
    
    train_data = datasets.ImageFolder(train_path,transform=transforms.Compose([
        transforms.Resize(size=(256,256)),#(h,w)
        transforms.RandomCrop(size=(256,256), padding=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.5,saturation=0.5),
        transforms.RandomHorizontalFlip(p=0.5),#依據p概率水平翻轉
        transforms.RandomVerticalFlip(p=0.5),#依據p概率垂直翻轉
        transforms.RandomRotation((-45,45)),#隨機角度旋轉
        transforms.RandomGrayscale(p=0.4),
        transforms.ToTensor()
    ]))
    val_data = datasets.ImageFolder(dev_path,transform=transforms.Compose([
        transforms.Resize(size=(256,256)),#(h,w)
        transforms.ToTensor()
    ]))
    print(train_data.classes)#获取标签
    print(train_data.class_to_idx)
    train_Loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    val_Loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=4,shuffle=False,num_workers=0)
    #*******************************************************************************************
    
    net = models.resnet152(pretrained=False)
    '''
    pretrained_dict=torch.load('save/save_desnet_best_12.50_0.0001.pkl')
    model_dict=net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    #net.classifier = torch.nn.Sequential(torch.nn.Linear(1920, num_class))
    '''
    net.fc = torch.nn.Sequential(torch.nn.Linear(2048, 1000),
                                torch.nn.Linear(1000, 500),
                                 torch.nn.Linear(500, 100),
                                 torch.nn.Linear(100, 3),)
    print(net)
    #net.load_state_dict(torch.load('save_train/save_net_best.pkl'))
    '''
    for i,(name, parma) in enumerate(net.named_parameters()):
        if(i<141):
            parma.requires_grad=False
        #if(int(name.split('.')[1])==2 and name.split('.')[0]=='layer4'):
        #    parma.requires_grad=False
        print(i,name,parma.requires_grad)#name.split('.')
    '''
    

    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=6e-4)
    #optimizer =torch.optim.RMSprop(net.parameters(), lr=LR, alpha=0.9)
    #optimizer =torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
    # optimizer = torch.optim.SGD([{'params':net.fc.parameters()},
    #                              {'params':net.layer4[1:3].parameters()}], 
    #                             lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.5, patience=40, verbose=True)
    loss_func = torch.nn.CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = net.to(device)
    net.train()
    #net.load_state_dict(torch.load('save_train_++/save_net_best_87.1000_0.0125.pkl'))#('save_train_3/save_net_best_45.1000_0.0484375.pkl'))
    
    training_loss,validation_loss=[],[]
    training_accuracy,validation_accuracy =[],[]
    best_acc,best_epoch=0,0
    
    save_file="save_train"
    if not os.path.isdir(save_file):
        os.mkdir(save_file)
    
    for epoch in range(EPOCHS):
        net.train()
        train_loss_reg,total_train,step_count,correct_train =0.0, 0,0,0
        #if(epoch in [50,70]):
        #    learn_rate=learn_rate/10
        #    print("****************learn_rate=",learn_rate,"*****************")
        #    optimizer = torch.optim.Adam(net.classifier.parameters(),lr=learn_rate)
        
        for step, (batch_x,label_y) in enumerate(train_Loader):
            #batch_x = torch.FloatTensor(batch_x.type(torch.FloatTensor)/255)
            #label_y = torch.LongTensor(label_y)
            #h,w,c= batch_x.shape[1],batch_x.shape[2],batch_x.shape[3]
            #input_shape = (-1,c,h,w)
            #train = Variable(batch_x.view(input_shape)).to(device)
            train = Variable(batch_x).to(device)
            labels = Variable(label_y).to(device)
            outputs = net(train)
            train_loss = loss_func(outputs,labels)
            
            optimizer.zero_grad()               # clear gradients for this training step
            train_loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients
            
            train_loss_reg +=train_loss.cpu().data
            #step_count += 1
            
            ans=torch.max(outputs,1)[1].squeeze()
            #total_train += len(labels)
            correct_train += (ans.cpu() == labels.cpu()).float().sum()
            
            print("Epoch:{}/{} Step:{}/{} Train_loss:{:1.3f} ".format(epoch+1,EPOCHS,step+1,len(train_Loader),train_loss))
            
        train_accuracy = 100 * correct_train / float(len(train_data))
        training_accuracy.append(train_accuracy)
        
        #print(step,step_count)
        avg_train_loss = train_loss_reg/len(train_Loader)
        training_loss.append(avg_train_loss)
        print("{}[Epoch:{}/{}  Avg_train_loss:{:1.3f} Acc_train:{:3.2f}%]".format(("*"*30),epoch+1,EPOCHS,avg_train_loss,train_accuracy))#loss.item()
        
        with torch.no_grad():
            net.eval()
            val_loss_reg,total_val,step_count,correct_val =0.0, 0,0,0
            for step, (batch_x,label_y) in enumerate(val_Loader):
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
            print("{}[Epoch:{}/{}  Avg_val_loss:{:1.3f} Acc_val:{:3.2f}]".format(("*"*30),epoch+1,EPOCHS,avg_val_loss,val_accuracy))
            if(val_accuracy>=best_acc):
                best_acc=val_accuracy
                torch.save(net.state_dict(), '{}/save_net_best_{}.{}_{}.pkl'.format(save_file,epoch+1,EPOCHS,LR))
                t_fi= open('{}/Best_Acc.txt'.format(save_file),'w')
                t_fi.writelines("epoch:{} acc:{}\n".format(epoch+1,best_acc))
                t_fi.close()
            torch.cuda.empty_cache()
        
        print("best_acc:{}%".format(best_acc))
        scheduler.step(val_accuracy)
        lr = optimizer.param_groups[0]['lr']
        print("LR:{}".format(lr))
        loss_plt_show(epoch+1,training_loss,validation_loss,LR,save_file)
        acc_plt_show(epoch+1,training_accuracy,validation_accuracy,LR,save_file)
    
    # #EPOCHS=35
    # loss_plt_show(EPOCHS,training_loss,validation_loss,LR,save_file)
    # acc_plt_show(EPOCHS,training_accuracy,validation_accuracy,LR,save_file)
    torch.save(net.state_dict(), '{}/save_net_{}_{}.pkl'.format(save_file,EPOCHS,LR))
    