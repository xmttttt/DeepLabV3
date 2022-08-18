import torch
from torch.utils.data import DataLoader
from dataloader import MyDataset
import numpy as np
from model import *
import cv2


# 请从“视觉感知中的认知”ppt中37-57页中任选一个语义分割模型，要求：
# 在PyTorch框架下实现其网络架构及基本训练流程（模型越新，基础分越高），要求代码从零开始自己实现（如果实现有困难，可以参考github上相关代码，但不允许直接照搬，如发现，直接按零分处理）；
# 在Weizmann Horse数据集（https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database/metadata）上汇报验证集上的指标（训练验证集按0.85：0.15的比例随机划分），指标包括mIoU和Boundary IoU（Boundary IoU: Improving Object-Centric Image Segmentation Evaluation，CVPR’21）（参考指标：模型汇报的mIoU指标应在0.9左右属于正常）；
# 将作业整理成报告提交，相关代码提交到github，报告中附上github repository的链接，同时在ReadMe中使用MarkDown写好代码运行说明（可参照https://github.com/poppinace/indexnet_matting 的格式），应附带训练好的模型的下载链接，且该模型运行的结果与报告中保持一致。


# 使用 for 循环暴力求解 IOU
def IOU(pred,gt):
    # pred,gt = pred[:,:,0],gt[:,:,0]
    # print(pred.shape,gt.shape)
    width,height = pred.shape

    # Intersection over Union

    intersection = 0
    union = 0 
    for w in range(width):
        for h in range(height):
            # print(pred[w,h],gt[w,h])

            if pred[w,h] == 1 and gt[w,h] == 1:
                intersection += 1

            if pred[w,h] == 1 or gt[w,h] == 1:
                union += 1

    return intersection/union

# 使用 np 库函数提速, 经验证与前者计算结果相同
def IOU_v2(pred,gt):

    inter = pred * gt
    intersection = np.sum(inter)
    union = np.sum(pred + gt - inter)

    return intersection/union

def Boundary_IOU(pred,gt,show=True,img=None):

    # canny边缘检测
    pred_canny = cv2.Canny(pred,0,255)
    gt_canny = cv2.Canny(gt,0,255)

    # 膨胀, 增大边缘面积
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (d, d))
    pred_dilate = cv2.dilate(pred_canny,kernel,1)
    gt_dilate = cv2.dilate(gt_canny,kernel,1)

    inter_pred = (pred/255).astype(np.float16) * (pred_dilate/255).astype(np.float16)
    inter_gt = (gt/255).astype(np.float16) * (gt_dilate/255).astype(np.float16)

    inter = inter_pred * inter_gt
    intersection = np.sum(inter)
    union = np.sum(inter_pred + inter_gt - inter)

    if show:
        cv2.imshow('primal',img)
        cv2.imshow('pred',pred)
        cv2.imshow('gt',gt)

        cv2.imshow('pred_canny',pred_canny)
        cv2.imshow('gt_canny',gt_canny)

        cv2.imshow('pred_dilate',pred_dilate)
        cv2.imshow('gt_dilate',gt_dilate)

        cv2.imshow('inter_pred',(inter_pred*255).astype(np.uint8))
        cv2.imshow('inter_gt',(inter_gt*255).astype(np.uint8))

        cv2.waitKey(0)

    return intersection / union

    

img_folder = './Datasets/image'
gt_folder = './Datasets/gt'
ratio = 0.85
batch_size = 1
iou_thres = 0.7

# evaluation中, 指定随机数种子区分测试集
test_dataset = MyDataset(img_folder,gt_folder,ratio,train=False,seed=1)

Test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

model = DeepLabv3()
model.load_state_dict(torch.load("./weights/best.pth"))
model.eval().cuda()

# best:0.83,0.45
# epoch40:0.84,0.46
# epoch100:0.94,0.78(?)
# epoch290:0.85,0.49

iou_sum = 0
biou_sum = 0
d = 10

with torch.no_grad():
    for iter, data in enumerate(Test_loader):

        # forward
        inputs, labels = data

        outputs = model(inputs.cuda())

        # print(inputs.shape,outputs.shape,labels.shape)

        for pred in outputs:

            pred = np.reshape(pred.cpu().detach().numpy(),(224,224,1))

            pred -= np.min(pred)
            pred /= np.max(pred)

            # 二值化
            # 使用otsu分割时, 要求输入为uint8
            ret,pred = cv2.threshold((pred*255).astype(np.uint8),iou_thres,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            img = inputs[0,:,:,:].permute(1,2,0).cpu().detach().numpy()
            labels = (np.reshape(labels.cpu().detach().numpy(),(224,224))*255).astype(np.uint8)

            # cv2.imshow('img',img)
            # cv2.imshow('pred',pred)
            # cv2.waitKey(10)
            # cv2.imshow('gt',labels)

            biou = Boundary_IOU(pred,labels,show=True,img=img)

            

            iou = IOU_v2((pred/255).astype(np.float16),(labels/255).astype(np.float16))
            print("iou: ",iou)
            print("biou: ",biou)

            iou_sum += iou
            biou_sum += biou

miou = iou_sum / len(Test_loader)
mbiou = biou_sum / len(Test_loader)

print("miou: ",miou)
print("mbiou: ",mbiou)    

            
