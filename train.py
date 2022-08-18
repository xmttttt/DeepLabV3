import numpy as np
import pandas as pd
import os
import time
import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from model import *
from loss_func import *
from dataloader import *
import matplotlib.pyplot as plt
from tqdm import *
from eval import *

img_folder = './Datasets/image'
gt_folder = './Datasets/gt'
ratio = 0.85
batch_size = 32

train_dataset = MyDataset(img_folder,gt_folder,ratio,train=True,seed=1)
test_dataset = MyDataset(img_folder,gt_folder,ratio,train=False,seed=1)

Train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
Test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

model = DeepLabv3()

model.cuda()

loss_func = Loss_func()

lr = 2e-4

MAX_EPOCH = 300

# optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

lambda1 = lambda epoch:0.99 ** epoch              
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda1)


def train():

    torch.cuda.empty_cache()

    Train_loss = []
    Test_loss = []
    Test_iou = []
    Test_biou = []

    for epoch in range(MAX_EPOCH):

        loss_mean = 0.

        model.train()

        for iter, data in enumerate(tqdm(Train_loader)):

            # forward
            inputs, labels = data
            # print(inputs.shape)

            # 通过 torchvision.dataset.ImageFolder 装载的数据时, 迭代器返回的为一列表, 第一位为数据, 第二位为labels, 无label时只取第一维即可
            outputs = model(inputs.cuda())

            # print(outputs.shape,labels.shape)

            # Compute loss
            optimizer.zero_grad()

            loss = loss_func(outputs, (labels)[:,0,:,:].cuda())

            # backward
            loss.backward()

            # updata weights
            optimizer.step()
            # print(loss.item())

            loss_mean += loss.item()

        Train_loss.append(loss_mean/(iter+1))
        # print(loss_mean,iter)
        print(epoch, 'Train mean loss: ', loss_mean/(iter+1))

        scheduler.step()

        loss_mean = 0.

        # 每 5 个 epoch 测试一次
        if epoch % 5 == 0:

            model.eval()
            with torch.no_grad():
  
                # 由于存在 dropout, 在测试时应将模型设为评估模式

                iou_sum = 0
                biou_sum = 0

                for iter, test_data in enumerate(Test_loader):

                    # forward
                    inputs, labels = test_data

                    # 通过 torchvision.dataset.ImageFolder 装载的数据时, 迭代器返回的为一列表, 第一位为数据, 第二位为labels, 无label时只取第一维即可
                    outputs = model(inputs.cuda())

                    loss = loss_func(outputs, (labels)[:,0,:,:].cuda())

                    loss_mean += loss.item()



                    for pred in outputs:

                        pred = np.reshape(pred.cpu().detach().numpy(),(224,224,1))
                        pred -= np.min(pred)
                        pred /= np.max(pred)

                        # 二值化
                        # 使用otsu分割时, 要求输入为uint8

                        ret,pred = cv2.threshold((pred*255).astype(np.uint8),0.7,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                        img = inputs[0,:,:,:].permute(1,2,0).cpu().detach().numpy()
                        labels = (np.reshape(labels.cpu().detach().numpy(),(224,224))*255).astype(np.uint8)

                        biou = Boundary_IOU(pred,labels)

                        iou = IOU_v2((pred/255).astype(np.float16),(labels/255).astype(np.float16))
                        # print("iou: ",iou)
                        # print("biou: ",biou)

                        iou_sum += iou
                        biou_sum += biou

                Test_loss.append(loss_mean/(iter+1))
                Test_iou.append(iou_sum/(iter+1))
                Test_biou.append(biou_sum/(iter+1))

                print("miou: ",iou_sum/(iter+1))
                print("mbiou: ",biou_sum/(iter+1))

                print(epoch, 'Test mean loss: ', loss_mean/(iter+1))

                loss_mean = 0.

        if epoch % 10 == 0:
            torch.save(model.state_dict(),'./weights/' + 'epoch' + str(epoch) +'test_loss'+ str(loss_mean/(iter+1)) + '.pth')
            plt.plot(Train_loss)
            plt.plot(5 * np.array(range(len(Test_loss))),Test_loss)
            plt.plot(5 * np.array(range(len(Test_loss))),Test_iou)
            plt.plot(5 * np.array(range(len(Test_loss))),Test_biou)
            # plt.show()
            plt.savefig('result.jpg')
            plt.clf()

    
train()