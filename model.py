import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# https://blog.csdn.net/qq_36530992/article/details/102628455?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166029678316782388042242%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166029678316782388042242&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-102628455-null-null.142^v40^pc_rank_34_queryrelevant0,185^v2^control&utm_term=aspp&spm=1018.2226.3001.4187


# 借鉴 torchvision 库手写的 ResNet50, 备用(文件中要求从零实现)
# ResNet 中不使用 bias, 是由于 bn 层的存在, bias 不起作用
class ResNet_Head(nn.Module):
    def __init__(self,v1c=True):
        super().__init__()

        if v1c:
            self.conv = nn.Sequential(
                nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1,bias=False),
                nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False),
                nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False)
            )

        else:
            self.conv = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)

        self.bn_conv = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # ceil_mode 会在pooling时, 将最后小于卷积核大小的边界保留下来另外计算, 等同于自适应padding
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=0,ceil_mode=True)


    def forward(self,x):
        output = self.conv(x)
        output = self.bn_conv(output)
        output = self.relu(output)
        output = self.maxpool(output)
        
        return output

class ResNet_bottleneck(nn.Module):
    def __init__(self,input_c,middle_c,output_c,stride=1,dilation=1):
        super().__init__()

        # 若需下采样 或 输入输出通道数不等，则残差链路需要1*1卷积控制通道数
        self.branch_conv = (stride != 1) or (input_c != output_c)

        if stride != 1:
            self.point_conv1 = nn.Conv2d(input_c,middle_c,kernel_size=1,stride=stride,bias=False)
        else:
            self.point_conv1 = nn.Conv2d(input_c,middle_c,kernel_size=1,bias=False)

        self.bn1 = nn.BatchNorm2d(middle_c)
        self.conv = nn.Conv2d(middle_c,middle_c,kernel_size=3,stride=1,padding=dilation,bias=False,dilation=dilation)
        self.bn2 = nn.BatchNorm2d(middle_c)
        self.point_conv2 = nn.Conv2d(middle_c,output_c,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(output_c)
        self.act = nn.ReLU(inplace=True)

        if self.branch_conv:
            if stride != 1:
                self.point_conv_ = nn.Conv2d(input_c,output_c,kernel_size=1,stride=stride,bias=False)
            else:
                self.point_conv_ = nn.Conv2d(input_c,output_c,kernel_size=1,bias=False)
            self.bn_ = nn.BatchNorm2d(output_c)


    def forward(self,x):
        y = self.point_conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv(y)
        y = self.bn2(y)
        y = self.act(y)
        y = self.point_conv2(y)
        y = self.bn3(y)

        if self.branch_conv:
            y_ = self.point_conv_(x)
            y_ = self.bn_(y_)
            y = y + y_

        else:
            y = y + x

        y = self.act(y)
        return y

class ResNet_block(nn.Module):
    def __init__(self,input_c,output_c,duplicate,stride=1,dilation=1):
        super().__init__()

        self.list = nn.ModuleList()
        self.list.append(ResNet_bottleneck(input_c,output_c//4,output_c,stride,dilation))

        for i in range(duplicate - 1):
            self.list.append(ResNet_bottleneck(output_c,output_c//4,output_c,1,dilation))

    def forward(self,x):
        for module in self.list:
            x = module(x)
        return x

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ResNet_Head(True)
        self.conv2 = ResNet_block(64,256,3,2)
        self.conv3 = ResNet_block(256,512,4,2)
        self.conv4 = ResNet_block(512,1024,6,2)
        self.conv5 = ResNet_block(1024,2048,3,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048,1000)

    def forward(self,x):
        print(0,x.shape)
        y = self.conv1(x)
        print(1,y.shape)
        y = self.conv2(y)
        print(2,y.shape)
        y = self.conv3(y)
        print(3,y.shape)
        y = self.conv4(y)
        print(4,y.shape)
        y = self.conv5(y)
        print(5,y.shape)
        y = self.avgpool(y)
        y = self.flatten(y)
        y = self.fc(y)
        return y


# backbone = ResNet50()
# conv1:4
# conv2:8
# conv3:16
# conv4:32
# conv5:32


class ResNet50_FCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ResNet_Head(True)
        self.conv2 = ResNet_block(64,256,3,2)
        self.conv3 = ResNet_block(256,512,4,1)
        self.conv4 = ResNet_block(512,1024,6,1,2)
        self.conv5 = ResNet_block(1024,2048,3,1,4)

    def forward(self,x):
        # print(0,x.shape)
        y = self.conv1(x)
        # print(1,y.shape)
        y = self.conv2(y)
        # print(2,y.shape)
        y = self.conv3(y)
        # print(3,y.shape)
        y = self.conv4(y)
        # print(4,y.shape)
        y = self.conv5(y)
        # print(5,y.shape)

        return y


class ASPP(nn.Module):
    def __init__(self,input_c=2048,middle_c=512,num_classes=10,scale=60,primal=480,dropout=0.3):
        super(ASPP,self).__init__()

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(input_c,middle_c,kernel_size=1,bias=False),
            nn.Dropout2d(p=dropout),
            nn.BatchNorm2d(middle_c),
            nn.ReLU(inplace=True),
            nn.Upsample((scale,scale),mode='bilinear')
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_c,middle_c,kernel_size=1,bias=False),
            nn.Dropout2d(p=dropout),
            nn.BatchNorm2d(middle_c),
            nn.ReLU(inplace=True)
        )
        
        self.dilated_conv2 = nn.Sequential(
            nn.Conv2d(input_c,middle_c,kernel_size=3,padding=6,dilation=6,bias=False),
            nn.Dropout2d(p=dropout),
            nn.BatchNorm2d(middle_c),
            nn.ReLU(inplace=True)
        )

        self.dilated_conv3 = nn.Sequential(
            nn.Conv2d(input_c,middle_c,kernel_size=3,padding=12,dilation=12,bias=False),
            nn.Dropout2d(p=dropout),
            nn.BatchNorm2d(middle_c),
            nn.ReLU(inplace=True)
        )

        self.dilated_conv4 = nn.Sequential(
            nn.Conv2d(input_c,middle_c,kernel_size=3,padding=18,dilation=18,bias=False),
            nn.Dropout2d(p=dropout),
            nn.BatchNorm2d(middle_c),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Sequential(
            nn.Conv2d(5*middle_c,middle_c,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(middle_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(middle_c,num_classes,kernel_size=1,bias=False),
            nn.Upsample((primal,primal),mode='bilinear')
        )

    def forward(self, x):

        y1 = self.conv1(x)
        y2 = self.dilated_conv2(x)
        y3 = self.dilated_conv3(x)
        y4 = self.dilated_conv4(x)
        y5 = self.image_pool(x)

        y = torch.cat([y1,y2,y3,y4,y5],dim=1)

        y = self.output(y)
        # print(y.shape)

        return y


class DeepLabv3(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = ResNet50_FCN()
        self.ASPP = ASPP(num_classes=1,scale=28,primal=224)
    
    def forward(self,x):
        y = self.backbone(x)
        y = self.ASPP(y)

        return y

if __name__ == '__main__':
    model = DeepLabv3()
    model(torch.ones((2,3,480,480)))

