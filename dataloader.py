from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader
import random
import torch
import numpy as np

transform_img = transforms.Compose([
    transforms.Resize(256),    # 将图片短边缩放至256，长宽比保持不变：
    transforms.CenterCrop(224),   #将图片从中心切剪成3*224*224大小的图片
    transforms.ToTensor(),          #把图片进行归一化，并把数据转换成Tensor类型
])

transform_gt = transforms.Compose([
    transforms.Resize(256),    # 将图片短边缩放至256，长宽比保持不变：
    transforms.CenterCrop(224),   #将图片从中心切剪成3*224*224大小的图片
    transforms.ToTensor(),          #把图片进行归一化，并把数据转换成Tensor类型
    transforms.Normalize((0.0), (0.003921569))

])



class MyDataset(Dataset):
    def __init__(self,img_folder,gt_folder,ratio,train=True,seed=None) -> None:
        super().__init__()

        self.data = datasets.ImageFolder(img_folder,transform=transform_img)
        self.label = datasets.ImageFolder(gt_folder,transform=transform_gt)

        # print(np.array(self.label[0][0])[:,:,0].shape)

        self.train_img = []
        self.test_img = []
        self.train_gt = []
        self.test_gt = []

        self.train = train

        num = list(range(len(self.data)))
        random.seed(seed)
        random.shuffle(num)
        train_num = num[:int(len(num) * ratio)]
        test_num = num[int(len(num) * ratio):]
        print(test_num)


        for index in range(len(self.data)):
            if index in train_num and train:
                self.train_img.append(self.data[index][0])
                self.train_gt.append((self.label[index][0])[0,:,:].unsqueeze(0))
            elif index in test_num and not train:
                self.test_img.append(self.data[index][0])
                self.test_gt.append((self.label[index][0])[0,:,:].unsqueeze(0))



        # for gt in self.train_gt:
            
        #     gt = np.array(gt)
        #     mask = np.array(gt[:,:,0] == gt[:,:,1]).flatten()
        #     for i in mask:
        #         if not i:
        #             print(i)

        #     mask = np.array(gt[:,:,1] == gt[:,:,2]).flatten()
        #     for i in mask:
        #         if not i:
        #             print(i)

            
            
            # print(gt)
            



    def __getitem__(self, index):

        if self.train:
            return self.train_img[index], self.train_gt[index]

        else:
            return self.test_img[index],self.test_gt[index]   

    def __len__(self):
        if self.train:
            return len(self.train_img)

        else:
            return len(self.test_img)




if __name__=='__main__':

    img_folder = './Datasets/image'
    gt_folder = './Datasets/gt'
    ratio = 0.85
    batch_size = 32

    train_dataset = MyDataset(img_folder,gt_folder,ratio,True)
    test_dataset = MyDataset(img_folder,gt_folder,ratio,False)
    
    train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=False)


    # for data in test_dataloader:
    #     print(data)